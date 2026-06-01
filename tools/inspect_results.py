#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick inspector for CoIRL-AD / LAW results.pkl.

This is a *zero-dependency-on-nuScenes* helper: it only needs the
``results.pkl`` produced by inference (or downloaded from the authors'
HuggingFace repo). It

  1. prints the structure of results.pkl,
  2. re-aggregates the planning metrics (L2 / collision) exactly the way the
     official evaluator does (sum over samples with fut_valid_flag==True,
     divided by the number of valid samples), so you can sanity-check the
     numbers against the paper / eval_metrics.xlsx,
  3. prints a few example predicted ego trajectories,
  4. optionally draws the predicted trajectories in BEV (matplotlib) and, if
     you pass --info-path, overlays the ground-truth ego trajectory.

results.pkl format (see CoIRL.py:insert_sample_results / save_results):
    dict { sample_token (str) : {
        'scene_token'        : str,
        'ego_fut_traj'       : np.ndarray [6, 2]  (cumsum'd LiDAR-ego deltas),
        'plan_L2_1s/2s/3s'   : float,
        'plan_obj_col_1s/2s/3s'     : float,
        'plan_obj_box_col_1s/2s/3s' : float,
        'fut_valid_flag'     : bool,
        # if eval_method='decouple', same metric keys again with '_rl_actor' suffix
    } }

Usage:
    python tools/inspect_results.py path/to/results.pkl
    python tools/inspect_results.py path/to/results.pkl --plot out.png --num-traj 80
    python tools/inspect_results.py path/to/results.pkl \
        --info-path data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
        --sample-token <token> --plot one_sample.png
"""

import argparse
import pickle
from collections import defaultdict


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_structure(results):
    print('=' * 60)
    print('STRUCTURE')
    print('=' * 60)
    print('top-level type   :', type(results).__name__)
    print('num samples      :', len(results))
    if not isinstance(results, dict):
        print('WARNING: expected a dict keyed by sample_token; got '
              f'{type(results).__name__}. The rest assumes the dict format.')
        return
    first_token = next(iter(results))
    entry = results[first_token]
    print('example key      :', first_token)
    print('per-entry keys   :', list(entry.keys()))
    print('\n-- example entry --')
    for k, v in entry.items():
        try:
            import numpy as np
            if isinstance(v, np.ndarray):
                print(f'  {k:24s}: ndarray shape={v.shape} dtype={v.dtype}')
                continue
        except ImportError:
            pass
        print(f'  {k:24s}: {v!r}')
    print()


def aggregate_metrics(results):
    """Replicate nuscenes_vad_dataset.py: sum over fut_valid_flag samples / num_valid."""
    print('=' * 60)
    print('AGGREGATED PLANNING METRICS')
    print('=' * 60)
    sums = defaultdict(float)
    num_valid = 0
    metric_keys = None
    for entry in results.values():
        if not entry.get('fut_valid_flag', False):
            continue
        num_valid += 1
        if metric_keys is None:
            metric_keys = [k for k in entry
                           if k not in ('scene_token', 'ego_fut_traj')]
        for k in metric_keys:
            v = entry[k]
            sums[k] += float(v) if not isinstance(v, bool) else float(v)

    if num_valid == 0:
        print('No samples with fut_valid_flag=True; cannot aggregate.')
        return

    print(f'num valid samples: {num_valid} / {len(results)}\n')
    means = {k: sums[k] / num_valid for k in metric_keys}

    def show_block(suffix, title):
        keys = [f'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
                'plan_obj_col_1s', 'plan_obj_col_2s', 'plan_obj_col_3s',
                'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s']
        keys = [k + suffix for k in keys]
        if not all(k in means for k in keys[:3]):
            return False
        print(f'-- {title} --')
        l2 = [means[f'plan_L2_{s}s{suffix}'] for s in (1, 2, 3)]
        oc = [means[f'plan_obj_col_{s}s{suffix}'] for s in (1, 2, 3)]
        bc = [means[f'plan_obj_box_col_{s}s{suffix}'] for s in (1, 2, 3)]
        print(f'  L2        (m)  1s={l2[0]:.4f}  2s={l2[1]:.4f}  '
              f'3s={l2[2]:.4f}  avg={sum(l2)/3:.4f}')
        print(f'  obj col   (%)  1s={oc[0]*100:.4f}  2s={oc[1]*100:.4f}  '
              f'3s={oc[2]*100:.4f}  avg={sum(oc)/3*100:.4f}')
        print(f'  box col   (%)  1s={bc[0]*100:.4f}  2s={bc[1]*100:.4f}  '
              f'3s={bc[2]*100:.4f}  avg={sum(bc)/3*100:.4f}')
        print()
        return True

    show_block('', 'IL / decoupled actor (default eval output)')
    show_block('_rl_actor', 'RL actor (only present when eval_method="decouple")')


def print_example_trajs(results, n=3):
    print('=' * 60)
    print(f'EXAMPLE PREDICTED TRAJECTORIES (first {n})')
    print('=' * 60)
    for i, (tok, entry) in enumerate(results.items()):
        if i >= n:
            break
        traj = entry.get('ego_fut_traj')
        print(f'sample_token: {tok}  valid={entry.get("fut_valid_flag")}')
        print('  ego_fut_traj (x,y per 0.5s step):')
        print('   ', traj)
    print()


def visualize(results, out_path, num_traj=60, info_path=None, sample_token=None):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 9))

    if sample_token is not None:
        # Single-sample mode: predicted vs (optional) GT.
        if sample_token not in results:
            raise SystemExit(f'sample_token {sample_token} not in results.pkl')
        pred = np.asarray(results[sample_token]['ego_fut_traj'])
        pred = np.vstack([[0, 0], pred])  # prepend ego origin
        ax.plot(pred[:, 0], pred[:, 1], '-o', color='tab:red', label='pred')
        if info_path is not None:
            infos = load_pickle(info_path)
            infos = infos['infos'] if isinstance(infos, dict) else infos
            match = next((it for it in infos if it.get('token') == sample_token), None)
            if match is not None and 'gt_ego_fut_trajs' in match:
                gt = np.asarray(match['gt_ego_fut_trajs']).reshape(-1, 2).cumsum(0)
                gt = np.vstack([[0, 0], gt])
                ax.plot(gt[:, 0], gt[:, 1], '-s', color='tab:green', label='GT')
            else:
                print('WARNING: sample_token not found in info_path or lacks '
                      'gt_ego_fut_trajs; drawing prediction only.')
        ax.set_title(f'sample {sample_token[:12]}...')
        ax.legend()
    else:
        # Overview mode: overlay many predicted trajectories (ego frame).
        items = list(results.items())[:num_traj]
        for tok, entry in items:
            traj = np.asarray(entry['ego_fut_traj'])
            traj = np.vstack([[0, 0], traj])
            ax.plot(traj[:, 0], traj[:, 1], '-', alpha=0.35, color='tab:blue')
        ax.plot(0, 0, 'k*', markersize=14, label='ego (t=0)')
        ax.set_title(f'{len(items)} predicted ego trajectories (BEV, ego frame)')
        ax.legend()

    ax.set_xlabel('x (m, forward)')
    ax.set_ylabel('y (m, left)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f'saved figure -> {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', help='path to results.pkl')
    ap.add_argument('--num-examples', type=int, default=3,
                    help='how many example trajectories to print')
    ap.add_argument('--plot', metavar='PNG', default=None,
                    help='also render a BEV figure to this path')
    ap.add_argument('--num-traj', type=int, default=60,
                    help='how many trajectories to overlay in overview plot')
    ap.add_argument('--info-path', default=None,
                    help='vad_nuscenes_infos_temporal_val.pkl, only needed to '
                         'overlay GT in single-sample plot (no images required)')
    ap.add_argument('--sample-token', default=None,
                    help='draw a single sample (pred vs GT) instead of overview')
    args = ap.parse_args()

    results = load_pickle(args.results)
    print_structure(results)
    if isinstance(results, dict):
        aggregate_metrics(results)
        print_example_trajs(results, args.num_examples)
        if args.plot:
            visualize(results, args.plot, num_traj=args.num_traj,
                      info_path=args.info_path, sample_token=args.sample_token)


if __name__ == '__main__':
    main()
