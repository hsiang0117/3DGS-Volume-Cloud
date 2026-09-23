"""Default training, saved checkpoint compatibility, and actual CUDA dispatch."""
import argparse
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.lightpass_config import saved_lightpass_settings, training_lightpass_settings


class LightpassDefaultsTests(unittest.TestCase):
    def test_training_config_round_trip(self):
        import train
        parser = argparse.ArgumentParser()
        model = ModelParams(parser)
        pipeline = PipelineParams(parser)
        optimization = OptimizationParams(parser)
        args = parser.parse_args([])
        pipe = pipeline.extract(args)
        opt = optimization.extract(args)
        fields = training_lightpass_settings()
        for key, value in fields.items():
            self.assertNotIn(key, vars(args))  # no experimental CLI switches
            self.assertEqual(getattr(pipe, key), value)
        self.assertNotIn('needle_split_interval', vars(opt))
        self.assertNotIn('needle_split_ratio', vars(opt))
        self.assertEqual(opt.lambda_aniso, .001)
        with tempfile.TemporaryDirectory() as folder:
            dataset = model.extract(args)
            dataset.model_path = folder
            with patch.object(train, 'TENSORBOARD_FOUND', False):
                train.prepare_output_and_logger(dataset, pipe, opt)
            self.assertEqual(saved_lightpass_settings((Path(folder)/'cfg_args').read_text()), fields)

    def test_legacy_and_experimental_checkpoints(self):
        self.assertEqual(saved_lightpass_settings('Namespace(tlight_voxel=False)'),
                         dict(tlight_tau_filter=False, tlight_full_grad=False, tlight_filter_variance=.3))
        self.assertEqual(saved_lightpass_settings('Namespace(tlight_tau_filter=True, tlight_full_grad=True)'),
                         dict(tlight_tau_filter=True, tlight_full_grad=True, tlight_filter_variance=.3))
        self.assertEqual(saved_lightpass_settings(training_lightpass_settings()), training_lightpass_settings())
        with self.assertRaises(ValueError):
            saved_lightpass_settings({'tlight_filter_variance': float('nan')})
        with self.assertRaises(ValueError):
            saved_lightpass_settings('__import__("os").getcwd()')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_default_lightpass_matches_zero_dilation_full_vjp(self):
        from gaussian_renderer import compute_T_light_raster
        means = torch.tensor([[-.19,.07,2.5],[.21,-.08,3.],[.03,.14,3.5]], device='cuda', requires_grad=True)
        tau = torch.tensor([[.8],[1.2],[.6]], device='cuda', requires_grad=True)
        scales = torch.tensor([[.55,.47,.37],[.62,.44,.41],[.57,.53,.32]], device='cuda', requires_grad=True)
        rotations = torch.tensor([[1.,0.,0.,0.]]*3, device='cuda', requires_grad=True)
        sun = torch.tensor([0.,0.,1.], device='cuda')
        inputs = [means, tau, scales, rotations]
        actual = compute_T_light_raster(*inputs, sun, image_size=48)
        expected = compute_T_light_raster(*inputs, sun, image_size=48,
                                         tau_filter=False, full_grad=True, filter_variance=0.)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual.sum(), inputs)
        ge = torch.autograd.grad(expected.sum(), inputs)
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-5)
            self.assertTrue(torch.isfinite(a).all())
        self.assertGreater(float(ga[0].abs().sum()), 0)
        self.assertGreater(float(ga[2].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
