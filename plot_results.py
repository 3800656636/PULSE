"""Generate publication-style static comparisons from saved independent runs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COLORS = {'PULSE': '#147d92', 'ACE': '#dc7445', 'D-MC': '#8c849f', 'UCB(2.0)': '#5668ad', 'RD-UCB': '#699657'}


def make_plots(out):
    out = Path(out)
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    with (out / 'summary.csv').open(encoding='utf-8') as f:
        rows = {r['method']: r for r in csv.DictReader(f)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.dpi': 170})
    names = [x for x in ['PULSE', 'ACE', 'RD-UCB', 'UCB(2.0)'] if x in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    x = np.arange(len(names))
    for offset, key, color, label in [(-.18, 'total_regret', '#147d92', 'Total'), (.18, 'post_regret', '#dc7445', 'After departures')]:
        ax.bar(x + offset, [float(rows[n][key + '_mean']) for n in names], .34,
               yerr=[float(rows[n][key + '_se'] or 0) for n in names], color=color,
               label=label, capsize=4, error_kw={'linewidth': 1})
    ax.set_xticks(x, names)
    ax.set_ylabel('Cumulative pseudo regret (thousands)')
    ax.set_title('PULSE benchmark | independent implementation', loc='left', pad=28)
    ax.text(0, 1.02, f'T = {config["horizon"]:,}   K = 20   M = 10   {len(config["seeds"])} seeds; mean +/- SE', transform=ax.transAxes, fontsize=9, color='#58616a')
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True); ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000:,.0f}'))
    fig.savefig(out / 'regret_comparison.png'); fig.savefig(out / 'regret_comparison.svg'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for name in config['methods']:
        arr = np.stack([np.load(out / f'{name}_{s}_curve.npz')['curve'] for s in config['seeds']])
        t = arr[0, :, 0]
        mean = arr[:, :, 1].mean(0)
        se = arr[:, :, 1].std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros(len(t))
        for ax in axes:
            if ax is axes[1] and name == 'D-MC':
                continue
            ax.plot(t, mean, label=name, color=COLORS[name], linewidth=1.8)
            ax.fill_between(t, mean - se, mean + se, color=COLORS[name], alpha=.12)
    for ax, title in zip(axes, ['All methods', 'Adaptation methods and UCB baselines']):
        ax.axvline(config['starts'][4], color='#a1a8ae', linestyle=':', linewidth=1)
        ax.axvline(config['departure'], color='#737d86', linestyle='--', linewidth=1)
        ax.set(xlabel='Global round (evaluation only)', ylabel='Cumulative pseudo regret', title=title)
        ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=8)
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    fig.suptitle(f'{len(config["seeds"])}-seed trajectories | shaded region: +/- one standard error', fontsize=12, fontweight='bold')
    fig.savefig(out / 'regret_curves.png'); fig.savefig(out / 'regret_curves.svg'); plt.close(fig)


if __name__ == '__main__':
    import sys
    make_plots(Path(sys.argv[1] if len(sys.argv) > 1 else 'results'))
