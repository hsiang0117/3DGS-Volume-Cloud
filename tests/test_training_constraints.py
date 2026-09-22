"""Regression tests for bounded density training and elapsed-step prune grace.

Run in the project environment:
    python -m unittest discover -s tests -p test_training_constraints.py -v
The lifecycle and training-loop tests require the project's CUDA environment.
"""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel


def options():
    parser = argparse.ArgumentParser()
    group = OptimizationParams(parser)
    opt = group.extract(parser.parse_args([]))
    opt.densify_adaptive = False
    opt.resurrect_interval = 0
    opt.contribution_reset_interval = 0
    return opt


def tiny_model(opt):
    model = GaussianModel()
    model.spatial_lr_scale = 1.0
    model._xyz = nn.Parameter(torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3))
    model._sigma_t = nn.Parameter(model._softplus_inverse(torch.full((4, 1), 0.1, device="cuda")))
    model._omega = nn.Parameter(torch.zeros(4, 3, device="cuda"))
    model._g = nn.Parameter(torch.zeros(4, 1, device="cuda"))
    model._w = nn.Parameter(torch.zeros(4, 6, device="cuda"))
    scales = torch.tensor([0.001, 0.02, 0.002, 0.01], device="cuda")
    model._scaling = nn.Parameter(scales.log().unsqueeze(1).repeat(1, 3))
    model._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(4, 1))
    model.max_radii2D = torch.ones(4, device="cuda")
    model.training_setup(opt)
    # Initialize Adam state without changing the fixture's parameters.
    sum(group["params"][0].sum() * 0 for group in model.optimizer.param_groups).backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    model.contribution_accum.fill_(5)
    model.contribution_denom.fill_(5)
    return model


class DensityProjectionTests(unittest.TestCase):
    def test_projection_preserves_forward_and_restores_boundary_gradient(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = GaussianModel()
                model._sigma_t = nn.Parameter(torch.tensor(
                    [[-10.], [0.], [4.], [6.], [100.]], device=device))
                parameter = model._sigma_t
                before = model.get_sigma_t.detach().clone()
                model.get_sigma_t.sum().backward()
                self.assertTrue(torch.equal(parameter.grad[-2:], torch.zeros_like(parameter.grad[-2:])))
                parameter.grad = None
                model.project_sigma_t()
                self.assertIs(parameter, model._sigma_t)
                torch.testing.assert_close(model.get_sigma_t, before, rtol=0, atol=0)
                model.get_sigma_t.sum().backward()
                self.assertTrue(bool((parameter.grad[-2:] > 0.99).all()))
                # A descent step can immediately leave the upper boundary.
                torch.optim.SGD([parameter], lr=0.01).step()
                model.project_sigma_t()
                self.assertTrue(bool((model.get_sigma_t[-2:] < 5).all()))

    def test_adam_projection_keeps_state_and_bound(self):
        model = GaussianModel()
        model._sigma_t = nn.Parameter(torch.tensor([[model.SIGMA_T_RAW_MAX]]))
        parameter = model._sigma_t
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            (model.get_sigma_t - 10).square().sum().backward()
            self.assertLess(parameter.grad.item(), 0)
            optimizer.step()
            state = optimizer.state[parameter]
            moment = state["exp_avg"].clone()
            model.project_sigma_t()
            self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
            self.assertIs(optimizer.state[parameter], state)
            torch.testing.assert_close(state["exp_avg"], moment, rtol=0, atol=0)
            self.assertLessEqual(model.get_sigma_t.item(), 5)
        # Existing outward momentum may delay recovery, but it no longer loses gradients.
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            (model.get_sigma_t - 1).square().sum().backward()
            optimizer.step()
            model.project_sigma_t()
        self.assertLess(model.get_sigma_t.item(), 4.9)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TrainingLifecycleTests(unittest.TestCase):
    def assert_aligned(self, model):
        count = len(model.get_xyz)
        for name in ("prune_grace", "contribution_accum", "contribution_denom",
                     "xyz_gradient_accum", "scale_gradient_accum", "denom", "max_radii2D"):
            self.assertEqual(getattr(model, name).shape[0], count, name)
        for group in model.optimizer.param_groups:
            param = group["params"][0]
            self.assertEqual(param.shape[0], count)
            for name in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(model.optimizer.state[param][name].shape, param.shape)
        self.assertTrue(bool((model.get_sigma_t <= 5).all()))

    def test_training_setup_projects_legacy_raw_parameters(self):
        opt = options()
        model = tiny_model(opt)
        with torch.no_grad():
            model._sigma_t.fill_(20)
        model.training_setup(opt)
        self.assertLessEqual(model._sigma_t.max().item(), model._sigma_t.new_tensor(model.SIGMA_T_RAW_MAX).item())
        model.get_sigma_t.sum().backward()
        self.assertTrue(bool((model._sigma_t.grad > 0.99).all()))

    def test_load_legacy_ply_preserves_forward_without_rewriting_file(self):
        model = tiny_model(options())
        with torch.no_grad():
            model._sigma_t[:2].fill_(20)
        before = model.get_sigma_t.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point_cloud.ply"
            model.save_ply(str(path))
            original_bytes = path.read_bytes()
            loaded = GaussianModel()
            loaded.load_ply(str(path))
            self.assertEqual(path.read_bytes(), original_bytes)
        torch.testing.assert_close(loaded.get_sigma_t, before, rtol=0, atol=0)
        loaded.get_sigma_t.sum().backward()
        self.assertTrue(bool((loaded._sigma_t.grad[:2] > 0.99).all()))

    def test_elapsed_clock_is_idempotent_and_rejects_backwards_time(self):
        model = tiny_model(options())
        model.advance_prune_grace(1000)
        model.prune_grace.fill_(500)
        model.advance_prune_grace(1000)
        self.assertEqual(model.prune_grace.tolist(), [500] * 4)
        model.advance_prune_grace(1300)
        self.assertEqual(model.prune_grace.tolist(), [200] * 4)
        model.advance_prune_grace(1499)
        self.assertEqual(model.prune_grace.tolist(), [1] * 4)
        model.advance_prune_grace(1500)
        self.assertEqual(model.prune_grace.tolist(), [0] * 4)
        model.advance_prune_grace(1600)
        self.assertEqual(model.prune_grace.tolist(), [0] * 4)
        with self.assertRaises(ValueError):
            model.advance_prune_grace(1599)

    def test_clone_split_and_both_prune_passes_keep_full_grace(self):
        opt = options()
        model = tiny_model(opt)
        model.advance_prune_grace(1000)
        model.prune_grace[2] = 170
        model.xyz_gradient_accum[:2] = 1
        model.denom.fill_(1)
        with torch.no_grad():
            model.physical_densify_and_prune(opt, 1000, torch.ones(4, device="cuda"), 1.0)
        # Original point 1 split into two; point 0 produced one clone.
        self.assertEqual(model.prune_grace.tolist(), [0, 170, 0, 500, 500, 500])
        model.tick_post_densify_maintenance(opt, 1000)
        self.assertEqual(model.prune_grace.tolist(), [0, 170, 0, 500, 500, 500])
        self.assert_aligned(model)
        model.advance_prune_grace(1499)
        self.assertEqual(model._prune_by_contribution(opt), 0)
        self.assertEqual(model.prune_grace[-3:].tolist(), [1, 1, 1])
        model.advance_prune_grace(1500)
        self.assertEqual(model._prune_by_contribution(opt), 3)
        self.assert_aligned(model)

    def test_resurrection_survives_same_tick_prune_and_reset(self):
        opt = options()
        opt.resurrect_interval = 3000
        opt.resurrect_fraction = 0.25
        opt.contribution_reset_interval = 1000
        model = tiny_model(opt)
        model.contribution_accum[0] = 0
        model.tick_post_densify_maintenance(opt, 3000)
        self.assertEqual(model.prune_grace.tolist(), [500, 0, 0, 0])
        self.assertEqual(model.contribution_denom.tolist(), [0] * 4)
        # Other points are observed again; the revived point remains invisible.
        model.contribution_accum[1:] = 5
        model.contribution_denom[1:] = 5
        model.advance_prune_grace(3499)
        self.assertEqual(model._prune_by_contribution(opt), 0)
        model.advance_prune_grace(3500)
        self.assertEqual(model._prune_by_contribution(opt), 1)
        self.assert_aligned(model)

    def test_needle_children_get_500_steps_independent_of_densify_start(self):
        opt = options()
        opt.densify_from_iter = 37
        model = tiny_model(opt)
        with torch.no_grad():
            model._scaling[0] = torch.tensor([1., 0.001, 0.01], device="cuda").log()
            model.advance_prune_grace(16000)
            self.assertEqual(model.split_needles(30, opt), 1)
        model.tick_post_densify_maintenance(opt, 16000)
        self.assertEqual(model.prune_grace.tolist(), [0, 0, 0, 500, 500])
        self.assert_aligned(model)
        model.advance_prune_grace(16499)
        self.assertEqual(model._prune_by_contribution(opt), 0)
        model.advance_prune_grace(16500)
        self.assertEqual(model._prune_by_contribution(opt), 2)
        self.assert_aligned(model)

    def test_training_loop_projects_each_step_and_ticks_outside_densify_window(self):
        import train

        opt = options()
        opt.iterations = 4
        opt.densify_until_iter = 0
        opt.needle_split_interval = 0
        model = tiny_model(opt)
        with torch.no_grad():
            model._sigma_t.fill_(model.SIGMA_T_RAW_MAX)
        camera = SimpleNamespace(original_image=torch.ones(3, 16, 16, device="cuda"))
        grace_seen = []
        raw_seen = []

        def fake_render(_camera, gaussians, _pipe, _background):
            raw_seen.append(gaussians._sigma_t.detach().clone())
            grace_seen.append(gaussians.prune_grace[0].item())
            if len(grace_seen) == 1:
                gaussians.prune_grace[0] = 500
            value = gaussians.get_sigma_t.mean() / 10
            return {"render": value.expand(3, 16, 16).contiguous(),
                    "viewspace_points": torch.zeros(4, 3, device="cuda", requires_grad=True),
                    "visibility_filter": torch.ones(4, device="cuda", dtype=torch.bool),
                    "radii": torch.ones(4, device="cuda"),
                    "contribution": torch.ones(4, device="cuda")}

        with (patch.object(train, "GaussianModel", return_value=model),
              patch.object(train, "Scene", return_value=SimpleNamespace()),
              patch.object(train, "prepare_output_and_logger", return_value=None),
              patch.object(train, "CameraPrefetcher") as prefetcher,
              patch.object(train, "training_report"),
              patch.object(train, "render", side_effect=fake_render)):
            prefetcher.return_value.next.return_value = camera
            train.training(SimpleNamespace(white_background=False), opt,
                           SimpleNamespace(tonemap_learnable=False), [], [])
        self.assertEqual(grace_seen, [0, 499, 498, 497])
        for raw in raw_seen:
            self.assertTrue(bool((raw <= raw.new_tensor(model.SIGMA_T_RAW_MAX)).all()))
        self.assertIsNotNone(model._sigma_t.grad)
        self.assertTrue(bool((model._sigma_t.grad < 0).all()))


if __name__ == "__main__":
    unittest.main()
