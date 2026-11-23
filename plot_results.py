"""Plot decision effects and the advantage of genuine audiovisual pairing."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


METHODS = [('fusion', 'Supervised fusion'), ('shrinkage', 'Genuine-pair shrinkage'),
           ('mean', 'Mean matching'), ('moment', 'Corrected moment'),
           ('random_moment', 'Corrected moment (random)')]
DATASETS = [('contest_moment_summary.json', 'MOSEI subset', '#2766A6', 'o'),
            ('sims_moment_summary.json', 'CH-SIMS', '#16736D', 's'),
            ('mosi_summary.json', 'CMU-MOSI', '#B64E25', '^')]


def interval(axis, mean, bounds, y, color, marker):
    axis.hlines(y, *bounds, color=color, linewidth=1, zorder=2)
    axis.plot(mean, y, marker=marker, markersize=4.0, color=color, linestyle='none', zorder=3)


def plot(root, output, alignment_script, layout_report):
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.linewidth': .75,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': 'white',
    })
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.55), sharey=True)
    figure.subplots_adjust(left=.265, right=.98, top=.855, bottom=.275, wspace=.20)
    rows = np.arange(len(METHODS))[::-1]
    keys = []
    for dataset, (filename, label, color, marker) in enumerate(DATASETS):
        values = json.loads((root / filename).read_text(encoding='utf-8'))['arms']
        keys.append(Line2D([], [], color=color, marker=marker, markersize=4, linestyle='none', label=label))
        for (name, _), row in zip(METHODS, rows):
            position = row + (.22 - dataset * .22)
            metrics = values[name]
            if name == 'fusion':
                axes[0].plot(0, position, color=color, marker=marker, markersize=4, clip_on=True)
            else:
                interval(axes[0], 100 * metrics['f1_minus_fusion'],
                         100 * np.asarray(metrics['f1_minus_fusion_ci']), position, color, marker)
            interval(axes[1], metrics['genuine_vs_single_nll'], metrics['genuine_vs_single_ci'],
                     position, color, marker)

    for letter, axis, title, unit in zip(
        ['a', 'b'], axes, ['Classification change', 'Value of genuine pairing'],
        ['Macro-F1 change (percentage points)', 'Log-loss advantage (nats)'],
    ):
        axis.axvline(0, color='black', linewidth=.7, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.tick_params(axis='x', length=3, width=.7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_title(title, loc='left', fontweight='bold', pad=7)
        axis.text(-.07, 1.09, letter, fontweight='bold', fontsize=9, transform=axis.transAxes)
        axis.set_xlabel(unit, labelpad=4)
        axis.set_ylim(-.5, len(METHODS) - .5)
        axis.margins(x=.1)
    axes[0].set_yticks(rows, [method[1] for method in METHODS])
    axes[1].tick_params(axis='y', labelleft=False)
    figure.legend(handles=keys, loc='lower center', bbox_to_anchor=(.61, .007),
                  ncol=3, frameon=False, columnspacing=1.1, handletextpad=.3)
    figure.canvas.draw()
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    for suffix in ('.pdf', '.svg', '.png'):
        figure.savefig(output.with_suffix(suffix), dpi=360, facecolor='white')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Common output stem for PDF, SVG, and PNG.')
    parser.add_argument('--alignment-script', type=Path)
    parser.add_argument('--layout-report', type=Path)
    args = parser.parse_args()
    plot(args.results, args.output, args.alignment_script, args.layout_report)
