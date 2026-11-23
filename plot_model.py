"""Illustrate the compact architecture and its two readouts on a held-out input."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as DrawingPath
import numpy as np

from plot_ablation import configure_figure, export


SIGNED, SCORE, NEUTRAL = '#1B5E9C', '#237974', '#8A486D'


def box(axis, x, y, width, height, label, fill):
    axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                  boxstyle='round,pad=0.006,rounding_size=0.015',
                                  facecolor=fill, edgecolor='black', linewidth=.72))
    axis.text(x, y, label, ha='center', va='center', linespacing=1.34)


def arrow(axis, points, color='black'):
    path = DrawingPath(points, [DrawingPath.MOVETO, *[DrawingPath.LINETO] * (len(points) - 1)])
    axis.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', linewidth=.8,
                                   mutation_scale=8, color=color))


def architecture(axis, mask):
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    axis.set_title('a  Observations and prediction', loc='left', fontweight='bold', pad=7)
    box(axis, .146, .75, .272, .235, 'Transcript\nand masks', '#EDF4F8')
    box(axis, .49, .75, .271, .235, 'Text anchor\n' + r'$\ell,\ s_0$', '#EDF4F8')
    box(axis, .85, .75, .270, .235, 'Class forecast\n' + r'$\ell+g\;\mapsto\;p$', '#EEF3F9')
    box(axis, .49, .355, .271, .235, 'Masked fusion\n' + r'$g\;\mapsto\;\delta,\nu$', '#ECF5F2')
    box(axis, .85, .355, .270, .235, 'Intensity forecast\n' + r'$\hat{s}(\delta; s_0)$', '#ECF5F2')

    axis.text(.146, .427, 'Binned observations', ha='center', va='center')
    for i, visible in enumerate(mask):
        tile = .266 / len(mask)
        axis.add_patch(Rectangle((.013 + i * tile, .325), tile - .0011, .050,
                                 facecolor='#76AFA5' if visible else '#F5DECF',
                                 edgecolor='none' if visible else '#AE6545', linewidth=.32))
    axis.text(.146, .265, '35% visual gap', ha='center', va='center', fontsize=7.)

    arrow(axis, [(.290, .75), (.343, .75)])
    arrow(axis, [(.285, .351), (.343, .351)])
    arrow(axis, [(.49, .625), (.49, .479)])
    axis.text(.506, .55, 'context', ha='left', va='center', fontsize=7.)
    arrow(axis, [(.633, .773), (.707, .773)])
    arrow(axis, [(.633, .403), (.671, .403), (.671, .694), (.707, .694)], SIGNED)
    arrow(axis, [(.633, .329), (.707, .329)], SCORE)
    axis.text(.67, .298, r'$\delta$', ha='center', va='top', color=SCORE)
    axis.text(.012, .040, r'With no observed audiovisual bins, $g=0$, $p=a$ and $\hat{s}=s_0$.',
              ha='left', va='bottom', fontsize=7.35)


def readouts(anchor, neutral, score, signed):
    signed = np.asarray(signed)
    log_weights = np.log(anchor) + np.stack((-signed, np.full(signed.shape, neutral), signed), axis=-1)
    weights = np.exp(log_weights - log_weights.max(axis=-1, keepdims=True))
    posterior = weights / weights.sum(axis=-1, keepdims=True)
    intensity = np.tanh(np.arctanh(np.clip(score, -.9999, .9999)) + signed)
    return posterior, intensity


def plot(evidence, output, scale):
    configure_figure()
    with np.load(evidence, allow_pickle=False) as packet:
        seed = packet['seeds'].tolist().index(97)
        native, gap = packet['masks'][:2, 1]
        eligible = native.all(-1) & gap[:, 0] & gap[:, -1] & ~gap.all(-1)
        index = int(np.flatnonzero(eligible)[0])
        anchor = packet['anchor'][seed, 0, index].astype(np.float64)
        fused = packet['shared'][seed, 0, index].astype(np.float64)
        score = float(packet['anchor_score'][seed, 0, index] / scale)
        mask, clip = gap[index], packet['ids'][index]
    correction = np.log(fused) - np.log(anchor)
    polar_mean = (correction[0] + correction[2]) / 2.
    neutral = correction[1] - polar_mean
    actual = (correction[2] - correction[0]) / 2.
    limit = max(1.7, abs(actual) + .45)
    x = np.linspace(-limit, limit, 301)
    posterior, intensity = readouts(anchor, neutral, score, x)
    fitted, fitted_score = readouts(anchor, neutral, score, actual)
    np.testing.assert_allclose(fused, fitted, rtol=2e-6, atol=2e-7)
    print(f'CH-SIMS v2 seed 97, clip {clip}, delta {actual:+.5f}, nu {neutral:+.5f}', flush=True)

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.7, 2.67), gridspec_kw={'width_ratios': (1.53, 1.)})
    fig.subplots_adjust(left=.024, right=.983, bottom=.171, top=.868, wspace=.234)
    architecture(left, mask)
    right.set_title('b  Effect of the shared shift', loc='left', fontweight='bold', pad=7)
    curves = ((posterior[:, 2] - posterior[:, 0], fitted[2] - fitted[0], SIGNED, 'Signed probability difference'),
              (intensity, fitted_score, SCORE, 'Signed intensity'),
              (posterior[:, 1], fitted[1], NEUTRAL, 'Neutral probability'))
    right.axhline(0, color='black', linewidth=.45, zorder=0)
    right.axvline(actual, ymax=.72, color='black', linewidth=.65, linestyle=(0, (3, 3)), zorder=1)
    for values, point, color, label in curves:
        right.plot(x, values, color=color, linewidth=1.35, label=label, zorder=2)
        right.plot(actual, point, marker='o', markersize=3.2, color=color, zorder=3)
    right.set(xlim=(-limit, limit), ylim=(-1.02, 1.02), ylabel='Normalized output',
              xlabel=r'Signed logit correction $\delta$')
    right.set_xticks([-1, 0, 1])
    right.set_yticks([-1, -.5, 0, .5, 1])
    right.legend(loc='upper left', frameon=False, fontsize=7.25, handlelength=1.45,
                 borderaxespad=.26, labelspacing=.28)
    export(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--intensity-scale', type=float, default=1.)
    args = parser.parse_args()
    if args.intensity_scale <= 0:
        parser.error('The fitting intensity scale must be positive.')
    plot(args.evidence, args.output, args.intensity_scale)


if __name__ == '__main__':
    main()
