"""Run training with shape/light/gradient monitoring.

The training camera RNG is independent and a full prefetch queue never drops a
frame, so every variant sees an identical sequence. The physical trainer, its
loss, optimizer and topology rules are otherwise unchanged.

Example (from repository root):
    python tools/train_lightpass_ablation.py -s D:/dataset/CloudDatasetZenith \
        -m output/<new-run> --eval

No losses, renderer kernels, optimizer updates, or topology rules are replaced.
The monitor runs before this iteration's topology changes and optimizer step.
"""
import argparse
from datetime import datetime
import json
import math
import queue
import random
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, data, optional=False):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            return True
        except PermissionError:
            if attempt == 5:
                if optional:
                    return False
                raise
            time.sleep(0.01 * (attempt+1))


def quantiles(values):
    return dict(zip(['min', 'p50', 'p90', 'p99', 'p999', 'max'],
                    np.quantile(values, [0, .5, .9, .99, .999, 1]).tolist()))


def shape_statistics(raw_scales, rotations, raw_density, contribution=None):
    """Pure CPU statistics. Quaternion convention matches build_rotation (wxyz)."""
    scales = np.exp(np.asarray(raw_scales, dtype=np.float64))
    ordered = np.sort(scales, axis=1)
    ratio = ordered[:, 2] / ordered[:, 0]
    surgery_ratio = ordered[:, 2] / np.maximum(ordered[:, 0], 1e-6)
    long_mid = ordered[:, 2] / ordered[:, 1]
    mid_short = ordered[:, 1] / ordered[:, 0]
    sigma = np.minimum(np.logaddexp(0, np.asarray(raw_density, dtype=np.float64).reshape(-1)), 5)
    q = np.array(rotations, dtype=np.float64, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError('Zero quaternion norm')
    q /= norm
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1-2*(y*y+z*z)
    R[:, 0, 1] = 2*(x*y-w*z)
    R[:, 0, 2] = 2*(x*z+w*y)
    R[:, 1, 0] = 2*(x*y+w*z)
    R[:, 1, 1] = 1-2*(x*x+z*z)
    R[:, 1, 2] = 2*(y*z-w*x)
    R[:, 2, 0] = 2*(x*z-w*y)
    R[:, 2, 1] = 2*(y*z+w*x)
    R[:, 2, 2] = 1-2*(x*x+y*y)
    major_gl = R[np.arange(len(q)), :, scales.argmax(1)]
    major_ue = major_gl[:, [2, 0, 1]] * [-1, 1, 1]
    thresholds = {}
    mass_proxy = sigma * scales.prod(1)
    contrib_total = float(np.sum(contribution)) if contribution is not None else 0
    for threshold in [5, 10, 20, 30, 50, 100]:
        mask = ratio > threshold
        n = int(mask.sum())
        row = dict(count=n, percent=100*n/len(q),
                   integrated_extinction_share=float(mass_proxy[mask].sum()/max(mass_proxy.sum(), 1e-30)))
        if contrib_total > 0:
            row['accumulated_alphaT_share'] = float(np.sum(contribution[mask])/contrib_total)
        if n:
            u = major_ue[mask]
            row.update(long_mid_median=float(np.median(long_mid[mask])),
                       mid_short_median=float(np.median(mid_short[mask])),
                       flat_percent=float(100*np.mean(long_mid[mask] < 1.2)),
                       needle_percent=float(100*np.mean((long_mid[mask] > 3) & (mid_short[mask] < 2))),
                       major_axis_squared_components_ue=(u*u).mean(0).tolist())
        thresholds[str(threshold)] = row
    return dict(gaussians=len(q), anisotropy=quantiles(ratio),
                surgery_eligible=int((surgery_ratio > 30).sum()),
                major_scale=quantiles(ordered[:, 2]), minor_scale=quantiles(ordered[:, 0]),
                long_mid=quantiles(long_mid), mid_short=quantiles(mid_short),
                sigma_t=quantiles(sigma), sigma_t_mean=float(sigma.mean()),
                density_at_cap_count=int((sigma >= 4.99999).sum()),
                major_axis_squared_components_ue=(major_ue*major_ue).mean(0).tolist(),
                thresholds=thresholds)


def self_test():
    s = np.log(np.array([[40., 1., 1.], [40., 40., 1.], [1., 1., 1.]]))
    q = np.array([[1.,0,0,0], [math.sqrt(.5),0,math.sqrt(.5),0], [1.,0,0,0]])
    q_before = q.copy()
    out = shape_statistics(s, q, np.zeros(3), np.array([2., 3., 5.]))
    assert np.array_equal(q_before, q), 'Diagnostics mutated quaternion inputs'
    assert out['thresholds']['30']['count'] == 2
    assert out['thresholds']['30']['flat_percent'] == 50
    assert out['thresholds']['30']['needle_percent'] == 50
    assert np.isclose(out['thresholds']['30']['accumulated_alphaT_share'], .5)
    assert np.allclose(out['thresholds']['30']['major_axis_squared_components_ue'], [.5, .5, 0])
    assert np.isclose(out['anisotropy']['max'],40)
    assert out['surgery_eligible'] == 2
    print('PASS: rotated axes, needle/flat distinction, threshold counts, contribution share, no input mutation')


def main():
    if '--self-test' in sys.argv:
        self_test()
        return
    import torch
    import train
    from arguments import ModelParams, PipelineParams, OptimizationParams
    parser = argparse.ArgumentParser(description=__doc__)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--monitor_interval', type=int, default=500)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7000,15000,20000,25000,30000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7000,15000,20000,25000,30000])
    args = parser.parse_args()
    if not args.model_path or args.monitor_interval < 1:
        parser.error('An explicit new model path and positive monitor interval are required')
    output = Path(args.model_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'cfg_args').exists():
        raise FileExistsError('Refusing to overwrite an existing training run')
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    args.test_iterations = sorted(set(args.test_iterations + [args.iterations]))
    state = dict(phase='initializing', iteration=0, total_iterations=args.iterations,
                 training_pid=os.getpid(), started_at=datetime.now().isoformat())
    started = time.monotonic()
    cumulative_monitor = 0.0
    cumulative_validation = 0.0
    last_camera = {}
    sampled = 0
    prefetchers = []
    history = (output/'shape_history.jsonl').open('x',encoding='utf-8',buffering=1)
    samples = (output/'training_camera_sequence.jsonl').open('x',encoding='utf-8',buffering=1)
    original_report = train.training_report
    original_prefetcher = train.CameraPrefetcher
    original_render = train.render
    latest_light = {}

    def recorded_render(*a, **kw):
        package = original_render(*a, **kw)
        latest_light.clear()
        latest_light.update(T=package['T_light'].detach(), Lk=package['Lk'].detach(), contribution=package['contribution'].detach())
        return package


    def status(**updates):
        state.update(updates, updated_at=datetime.now().isoformat(),
                     elapsed_seconds=time.monotonic()-started,
                     monitoring_seconds=cumulative_monitor, validation_seconds=cumulative_validation)
        write_json(output/'status.json', state, optional=True)

    class RecordingPrefetcher(original_prefetcher):
        def __init__(self,*a,**kw):
            # An independent RNG and no dropped queue items guarantee the same
            # training camera sequence despite different CUDA step/eval times.
            self._experiment_rng = random.Random(20260923)
            super().__init__(*a,**kw)
            prefetchers.append(self)

        def _pick_one(self):
            with self._lock:
                if not self._stack:
                    self._refill()
                index = self._experiment_rng.randrange(len(self._stack))
                cam = self._stack.pop(index)
                self._indices.pop(index)
                return cam

        def _run(self):
            while not self._stop.is_set():
                cam = self._pick_one()
                try:
                    _ = cam.original_image
                except Exception as error:
                    print(f'[prefetch] {error}', flush=True)
                while not self._stop.is_set():
                    try:
                        self._q.put(cam, timeout=.1)
                        break
                    except queue.Full:
                        continue

        def next(self):
            nonlocal sampled, last_camera
            camera = super().next()
            sampled += 1
            last_camera = dict(iteration=sampled, file_path=Path(camera.image_path).relative_to(Path(args.source_path)).as_posix())
            samples.write(json.dumps(last_camera)+'\n')
            return camera

    def monitored_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                         scene, renderFunc, renderArgs, final_iteration=None):
        nonlocal cumulative_monitor, cumulative_validation
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            raise FloatingPointError(f'Nonfinite training loss at {iteration}')
        if iteration == 1 or iteration % args.monitor_interval == 0 or iteration == args.iterations:
            start_monitor = time.monotonic()
            g = scene.gaussians
            fields = {name: getattr(g,name).detach().cpu().numpy() for name in
                      ['_xyz','_scaling','_rotation','_sigma_t','_omega','_g','_w']}
            invalid = [name for name,array in fields.items() if not np.isfinite(array).all()]
            if invalid:
                write_json(output/'nonfinite_failure.json',dict(iteration=iteration, fields=invalid))
                raise FloatingPointError(f'Nonfinite model fields: {invalid}')
            if fields['_sigma_t'].max() > math.log(math.expm1(5))+1e-5:
                raise ValueError('Density raw projection upper bound violated')
            contribution = g.contribution_accum.detach().cpu().numpy()
            snapshot = shape_statistics(fields['_scaling'],fields['_rotation'],fields['_sigma_t'],contribution)
            snapshot.update(iteration=iteration, elapsed_seconds=time.monotonic()-started,
                            loss=loss_value, l1=float(Ll1), camera=last_camera,
                            all_fields_finite=True, stage='before_topology_and_optimizer',
                            learning_rates={group['name']:group['lr'] for group in g.optimizer.param_groups},
                            cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                            coordinate_convention='OpenGL=(UE_y,UE_z,-UE_x)',
                            contribution_note='Historical accumulated sum(alpha*T), not current-view RGB importance; resets follow unchanged training policy')
            gradients = {}
            for name in fields:
                grad = getattr(g, name).grad
                if grad is not None:
                    value = grad.detach().cpu().numpy().astype(np.float64)
                    if not np.isfinite(value).all():
                        raise FloatingPointError(f'Nonfinite {name} gradient at {iteration}')
                    gradients[name] = dict(l2=float(np.linalg.norm(value)), max_abs=float(np.abs(value).max()))
            light = latest_light['T'].detach().cpu().numpy().reshape(-1)
            lk = latest_light['Lk'].detach().cpu().numpy()
            weights = latest_light['contribution'].detach().cpu().numpy().reshape(-1)
            if not np.isfinite(light).all() or not np.isfinite(lk).all():
                raise FloatingPointError('Nonfinite light/shading')
            omega_activated = 1/(1+np.exp(-fields['_omega'].astype(np.float64)))
            snapshot.update(gradients=gradients, T_light=quantiles(light), T_light_mean=float(light.mean()),
                            T_light_contribution_weighted=float(np.dot(weights,light)/max(weights.sum(),1e-30)),
                            Lk=quantiles(lk), Lk_mean=float(lk.mean()),
                            omega_mean=float(omega_activated.mean()),
                            w_mean=float(np.logaddexp(0,fields['_w'].astype(np.float64)).mean()))
            cumulative_monitor += time.monotonic()-start_monitor
            snapshot['monitoring_seconds_cumulative'] = cumulative_monitor
            history.write(json.dumps(snapshot,allow_nan=False)+'\n')
            write_json(output/'shape_latest.json',snapshot,optional=True)
            print(f"\n[SHAPE {iteration}] N={snapshot['gaussians']} "
                  f"r_p99={snapshot['anisotropy']['p99']:.3f} r_max={snapshot['anisotropy']['max']:.3f} "
                  f">30={snapshot['thresholds']['30']['count']} >100={snapshot['thresholds']['100']['count']} "
                  f"sigma_max={snapshot['sigma_t']['max']:.5f}",flush=True)
        if iteration % 100 == 0 or iteration == 1:
            status(phase='training',iteration=iteration,loss=loss_value)
        if iteration in testing_iterations:
            status(phase='validation',iteration=iteration)
        start_validation = time.monotonic()
        original_report(tb_writer,iteration,Ll1,loss,l1_loss,elapsed,testing_iterations,
                        scene,renderFunc,renderArgs,final_iteration=final_iteration)
        if iteration in testing_iterations:
            cumulative_validation += time.monotonic()-start_validation
            status(phase='training',iteration=iteration)

    train.render = recorded_render
    train.CameraPrefetcher = RecordingPrefetcher
    train.training_report = monitored_report
    status()
    try:
        # Identical seed setup to train.py; monitoring uses no random draws.
        train.safe_state(False)
        train.training(lp.extract(args),op.extract(args),pp.extract(args),args.test_iterations,args.save_iterations)
        status(phase='training_complete',iteration=args.iterations,training_seconds=time.monotonic()-started,
               sampled_frames=sampled)
        print('\nMonitored training complete.',flush=True)
    except BaseException as error:
        status(phase='failed',error=repr(error))
        traceback.print_exc()
        raise
    finally:
        for prefetcher in prefetchers:
            prefetcher.shutdown()
        history.close()
        samples.close()
        train.training_report = original_report
        train.CameraPrefetcher = original_prefetcher
        train.render = original_render


if __name__ == '__main__':
    main()
