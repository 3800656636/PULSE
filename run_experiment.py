"""Run: python run_experiment.py --output results
Smoke: python run_experiment.py --horizon 20000 --seeds 7 --output smoke
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
from simulator import METHODS, make_schedule, validate, simulate_adaptive, simulate_baseline

PAPER = {
    "PULSE": [140231.1, 4169.5, 96494.0, 3862.7, .00994, .00039, 120014.6, 4669.6, 0.0, 0.0],
    "ACE": [242567.8, 10598.3, 198830.6, 9827.7, .01021, .00006, 123273.4, 680.7, 1.0, .3],
    "D-MC": [3413914.6, 2713.0, 3090709.6, 2720.3, .14594, .00005, 1762934.6, 630.6, None, None],
    "UCB(2.0)": [609991.1, 87298.7, 591385.7, 87555.3, .00127, .00003, 15363.6, 397.7, None, None],
    "RD-UCB": [270089.2, 118747.5, 251568.9, 118379.6, .00126, .00006, 15210.4, 714.2, None, None],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--horizon', type=int, default=2_000_000)
    p.add_argument('--seeds', type=int, nargs='+', default=[7, 13, 29, 43, 71])
    p.add_argument('--methods', nargs='+', choices=list(METHODS), default=['PULSE', 'ACE', 'D-MC', 'UCB(2.0)', 'RD-UCB'])
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--profile', choices=['practical', 'theoretical'], default='practical')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    T, K, m, M = args.horizon, 20, 10, 10
    mu = .1 + .05 * np.arange(K - 1, -1, -1)
    starts, ends, departure = make_schedule(T)
    validate(T, mu, starts, ends, m)
    step = max(1, T // 1000)
    checkpoints = np.array(sorted(set([0, 1, T, int(starts[4]), departure, departure + 1] + list(range(step, T + 1, step)))), dtype=np.int64)
    profile = {'confidence': .02, 'lp': 100, 'lq': 500, 'epsilon': .01}
    config = dict(horizon=T, arms=K, population_cap=m, potential_players=M,
                  means=mu.tolist(), starts=starts.tolist(), ends=ends.tolist(),
                  departure=departure, seeds=args.seeds, methods=args.methods,
                  profile=args.profile, reward='gaussian', sigma=.5,
                  practical=profile, schedule_inclusive=True,
                  crn='one independent Gaussian innovation per (seed, global round, player), shared across methods',
                  rng='NumPy PCG64 rewards; independent SeedSequence/xorshift64* private player actions',
                  implementation='independent-paper-implementation-v1', python=platform.python_version(),
                  numpy=np.__version__)
    config['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ['simulator.py', 'run_experiment.py']}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    all_rows = []
    for seed in args.seeds:
        print(f'Seed {seed}: preparing common rewards', flush=True)
        noise = np.random.default_rng(np.random.SeedSequence([seed, 20261003])).normal(0, .5, size=(T, M))
        private = np.random.SeedSequence([seed, 991]).generate_state(M, dtype=np.uint64)
        for name in args.methods:
            tic = time.perf_counter()
            method = METHODS[name]
            if method < 2:
                if args.profile == 'theoretical':
                    confidence = 6.
                    lp, lq = math.ceil(866 * math.log(T)), math.ceil(570 * math.log(T))
                    epsilon = min(m ** 1.5 / math.sqrt(T), 1 / K, .1, 1 / (4 * m ** 3)) if method == 0 else min(math.sqrt(1141 * m ** 3 * math.log(T) / (2 * T)), 1 / K, .1)
                else:
                    confidence, lp, lq, epsilon = .02, 100, 500, .01
                metrics, curves, events, home, counts = simulate_adaptive(method, T, mu, starts, ends, m, noise, private.copy(), epsilon, confidence, lp, lq, checkpoints, departure)
                np.savez_compressed(out / f'{name}_{seed}_diagnostics.npz', events=events, final_home=home, sample_counts=counts)
            else:
                metrics, curves = simulate_baseline(method, T, mu, starts, ends, noise, private.copy(), checkpoints, departure)
            elapsed = time.perf_counter() - tic
            np.savez_compressed(out / f'{name}_{seed}_curve.npz', curve=curves)
            row = dict(method=name, seed=seed, total_regret=metrics[0], post_regret=metrics[1],
                       collided_player_rounds=int(metrics[2]), active_player_rounds=int(metrics[3]),
                       collision_rate=metrics[2] / metrics[3], false_releases=metrics[4],
                       probes=int(metrics[5]), verifications=int(metrics[6]), seats=int(metrics[7]), seconds=elapsed)
            all_rows.append(row)
            print(f'  {name:9s} regret={metrics[0]:,.1f} post={metrics[1]:,.1f} collision={row["collision_rate"]:.6f} time={elapsed:.1f}s', flush=True)
            with (out / 'per_seed.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(all_rows)
    summary = []
    for name in args.methods:
        rows = [r for r in all_rows if r['method'] == name]
        r = {'method': name, 'runs': len(rows)}
        for key in ['total_regret', 'post_regret', 'collision_rate', 'collided_player_rounds', 'false_releases']:
            vals = np.array([x[key] for x in rows])
            r[key + '_mean'] = float(np.mean(vals))
            r[key + '_se'] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
        paper = PAPER[name]
        r['paper_total_mean'] = paper[0]
        r['paper_post_mean'] = paper[2]
        r['total_difference_percent'] = 100 * (r['total_regret_mean'] / paper[0] - 1) if T == 2_000_000 else None
        summary.append(r)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    (out / 'paper_reference.json').write_text(json.dumps(PAPER, indent=2), encoding='utf-8')
    from plot_results import make_plots
    make_plots(out)
    print(f'Completed. Results: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
