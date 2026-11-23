"""Plot paired component effects, missingness sensitivity and cross-corpus comparisons."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from ablate import ranked_probability_score
from figure_plates import PlatePart


TEAL = '#007F83'
INDIGO = '#303D78'
MAGENTA = '#AD3567'
INK = '#222833'
SLATE = '#647082'
GRID = '#E4E9EE'
EFFECTS = LinearSegmentedColormap.from_list('paired_effects', (MAGENTA, '#FAFAFA', TEAL))


def interval(axis, mean, bounds, row, color, marker):
    low, high = bounds
    axis.hlines(row, low, high, color=color, alpha=.22, linewidth=3.7, zorder=2)
    axis.hlines(row, low, high, color=color, linewidth=.70, zorder=2)
    axis.plot(mean, row, marker=marker, markersize=3.8, color=color, linestyle='none', zorder=3)


def signed_bar(axis, gain, interval_bounds, row, color, *, height=.42):
    axis.barh(row, gain, height=height, color=color, linewidth=0, alpha=.73, zorder=2)
    axis.hlines(row, *interval_bounds, color=INK, linewidth=.85, zorder=3)


def gain_tiles(axis, estimates, *, gain_key='reduction', interval_key='ci95_familywise', color_limit=25):
    rows = np.arange(len(estimates))
    gains = 1000 * np.array([entry[gain_key] for entry in estimates])
    image = axis.imshow(gains[:, None], cmap=EFFECTS, vmin=-color_limit, vmax=color_limit, aspect='auto',
                        extent=(0, 1, len(estimates) - .5, -.5), interpolation='nearest')
    for row, gain, entry in zip(rows, gains, estimates, strict=True):
        low, high = 1000 * np.asarray(entry[interval_key])
        foreground = 'white' if abs(gain) > .68 * color_limit else INK
        axis.text(.5, row - .22, f'{gain:+.1f}'.replace('-', '−'), color=foreground,
                  fontsize=8, fontweight='bold', va='center', ha='center')
        axis.text(.5, row + .23, f'[{low:.1f}, {high:.1f}]'.replace('-', '−'), color=foreground,
                  fontsize=7.1, va='center', ha='center')
    axis.hlines(rows[:-1] + .5, 0, 1, colors='white', linewidth=.9)
    axis.set_xticks([])
    axis.spines[['left', 'bottom']].set_visible(False)
    axis.tick_params(axis='y', length=0, pad=6, labelsize=8)
    return image


def replicate_ticks(axis, values, rows, color):
    axis.scatter(values, rows, marker='|', s=12, color=color, alpha=.75, linewidths=.84, zorder=2)


def seed_comparison(labels, seeds, gains, effects, limits, familywise=None):
    """Place each resampled interval beside the effects from its original fitted seeds."""
    figure, (means, by_seed) = plt.subplots(1, 2, sharey=True, figsize=(6.60, 2.22 if len(labels) < 4 else 2.62))
    figure.subplots_adjust(left=.217, right=.894, top=.845, bottom=.244, wspace=.27)
    positions = np.arange(len(labels))[::-1]
    for row, mean, bounds in zip(positions, effects, limits, strict=True):
        interval(means, mean, bounds, row, TEAL, 'D')
    if familywise is not None:
        for row, bounds in zip(positions, familywise, strict=True):
            means.hlines(row - .175, *bounds, linewidth=.78, color=SLATE, zorder=2)
    means.set_yticks(positions, labels)
    means.set_ylim(-.48, len(labels) - .48)
    means.axvline(0, color=SLATE, linestyle=(0, (3, 3)), linewidth=.68, zorder=1)
    means.xaxis.set_major_locator(MaxNLocator(5))
    means.margins(x=.12)
    means.set_xlabel('Mean RPS reduction × 1,000')
    means.set_title('a', loc='left', pad=6)

    color_limit = max(5., 5. * np.ceil(np.max(np.abs(gains)) / 5.))
    image = by_seed.imshow(gains, cmap=EFFECTS, norm=TwoSlopeNorm(vcenter=0, vmin=-color_limit, vmax=color_limit),
                           aspect='auto', interpolation='none',
                           extent=(-.5, len(seeds) - .5, -.5, len(labels) - .5))
    by_seed.set_xticks(np.arange(len(seeds)), seeds, fontsize=6.65)
    by_seed.set_xlabel('Training seed')
    by_seed.tick_params(axis='y', labelleft=False)
    by_seed.tick_params(axis='x', length=0, pad=4)
    by_seed.set_title('b', loc='left', pad=6)
    for edge in np.arange(len(seeds) - 1) + .5:
        by_seed.axvline(edge, color='white', linewidth=.57)
    by_seed.hlines(np.arange(len(labels) - 1) + .5, -.5, len(seeds) - .5, color='white', linewidth=.65)
    for axis in (means, by_seed):
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0, pad=4)
    by_seed.spines['bottom'].set_visible(False)
    color_axis = by_seed.inset_axes([1.031, .17, .045, .67])
    scale = figure.colorbar(image, cax=color_axis, ticks=[-color_limit, 0, color_limit])
    scale.ax.tick_params(labelsize=7, pad=2, length=1.7)
    scale.outline.set_visible(False)
    return figure


def summaries(root):
    primary = json.loads((root / 'sims_v2_shared_score_ablations.json').read_text(encoding='utf-8'))
    streams = json.loads((root / 'sims_v2_stream_ablations.json').read_text(encoding='utf-8'))
    return [('Independent intensity', primary['contrasts']['independent_intensity_rps']),
            ('Class-only fusion', streams['contrasts']['fusion']),
            ('Text + audio', streams['contrasts']['intensity_audio']),
            ('Text + vision', streams['contrasts']['intensity_vision'])]


def condition_scores(root, arm, run_prefix):
    cached = root / 'module_gap_scores.npz'
    if cached.is_file():
        with np.load(cached, allow_pickle=False) as packet:
            if arm in packet:
                np.testing.assert_array_equal(packet['seeds'], [97, 17, 521])
                np.testing.assert_array_equal(packet['gap_percent'], [35, 65])
                return packet[arm]
    seeds = []
    for seed in (97, 17, 521):
        with np.load(root / f'{run_prefix}{seed}' / 'predictions.npz', allow_pickle=False) as packet:
            seeds.append(ranked_probability_score(packet[arm][1:3], packet['y']).mean(-1))
    return np.stack(seeds)


def configure_figure():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.linewidth': .70,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': .70, 'ytick.major.width': .70,
        'xtick.major.size': 3., 'ytick.major.size': 3., 'legend.labelcolor': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': 'white',
    })


def panel_rows(fig, labels):
    """Read letter anchors in inches and group letters on the same visual row."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    positions = []
    for letter, artist in labels:
        bounds = artist.get_window_extent(renderer)
        positions.append((letter, bounds.x0 / fig.dpi, bounds.y0 / fig.dpi, artist))
    rows = []
    for item in sorted(positions, key=lambda value: -value[2]):
        if not rows or rows[-1][0][2] - item[2] > 8 / 72:
            rows.append([])
        rows[-1].append(item)
    return [sorted(row, key=lambda value: value[1]) for row in rows]


def export(fig, output, *, align_panels=False):
    part = output if isinstance(output, PlatePart) else None
    names = dict(zip('abcdef', part.letters)) if part else {}
    align = align_panels or part is not None
    labels = []
    contexts = {'CH-SIMS v2', 'OMG-Emotion', 'RAVDESS', 'CREMA-D', 'MOUD', 'MELD', 'MSP-Face',
                'MELD, frozen', 'MOUD, frozen'}
    for axis in fig.axes:
        for location in ('left', 'center', 'right'):
            title = axis.get_title(loc=location).strip()
            if title:
                first, *rest = title.split(maxsplit=1)
                letter = first if len(first) == 1 and first in 'abcdefg' else ''
                label = names.get(letter, letter)
                original = axis.set_title(label, loc=location, fontsize=9, fontweight='bold')
                if letter:
                    labels.append((label, original))
                    if (align and rest and rest[0] in contexts
                            and not any(item.get_text().strip() == rest[0] for item in axis.texts)):
                        axis.text(.5, 1.029, rest[0], transform=axis.transAxes, ha='center',
                                  va='bottom', fontsize=7.8)
        if align:
            for artist in list(axis.texts):
                letter = artist.get_text().strip()
                if len(letter) == 1 and letter in 'abcdefg' and artist.get_fontweight() in ('bold', 700):
                    labels.append((names.get(letter, letter), artist))
    if part:
        if not labels and len(part.letters) == 1:
            artist = fig.axes[0].set_title(part.letters, loc='left', fontsize=9, fontweight='bold', pad=6)
            labels.append((part.letters, artist))
        if sorted(letter for letter, _ in labels) != sorted(part.letters):
            raise ValueError(f'Unexpected panel labels for {part.path.stem}: {labels}')
        rows = panel_rows(fig, labels)
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.038)
        placements = [{'letters': ''.join(letter for letter, *_ in row),
                       'baseline_from_top': 72 * (bounds.y1 - row[0][2]) - 1.}
                      for row in rows]
        part.path.with_suffix('.panels.json').write_text(json.dumps(placements), encoding='utf-8')
        for _, artist in labels:
            artist.set_visible(False)
        for suffix in ('.pdf', '.svg'):
            fig.savefig(part.path.with_suffix(suffix), dpi=360, facecolor='white',
                        bbox_inches=bounds)
        plt.close(fig)
        return
    if align and labels:
        rows = panel_rows(fig, labels)
        columns = [x for _, x, _, _ in max(rows, key=len)]
        width, height = fig.get_size_inches()
        for row in rows:
            baseline = max(location for _, _, location, _ in row) + 1 / 72
            for index, (letter, _, _, artist) in enumerate(row):
                artist.set_visible(False)
                fig.text(columns[index] / width, baseline / height, letter, va='baseline',
                         ha='left', color='black', fontsize=9, fontweight='bold')
    bounds = None
    if align_panels:
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(5 / 72)
    for suffix in ('.pdf', '.svg', '.png'):
        fig.savefig(output.with_suffix(suffix), dpi=360, facecolor='white', bbox_inches=bounds)
    plt.close(fig)


def plot(root, output, alignment_script, layout_report):
    configure_figure()
    fig, (effects, severity) = plt.subplots(1, 2, figsize=(6.5, 2.75))
    fig.subplots_adjust(left=.22, right=.97, bottom=.26, top=.86, wspace=.36)
    comparisons = summaries(root)
    positions = np.arange(len(comparisons))[::-1]
    for y, (_, contrast) in zip(positions, comparisons):
        interval(effects, contrast['rps_reduction'] * 1000, np.asarray(contrast['ci95']) * 1000,
                 y, '#1B5E9C', 'o')
    effects.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    effects.set_yticks(positions, [label for label, _ in comparisons])
    effects.set_ylim(-.55, len(comparisons) - .45)
    effects.xaxis.set_major_locator(MaxNLocator(5))
    effects.spines['left'].set_visible(False)
    effects.tick_params(axis='y', length=0)
    effects.set_xlabel(r'RPS reduction ($10^{-3}$)')

    rates = np.asarray([35, 65])
    variants = [
        ('intensity', 'sims_v2_shared_score_seed', 'Shared audio + vision', '#1B5E9C', 'o'),
        ('independent_intensity', 'sims_v2_shared_score_seed', 'Independent intensity', '#A9502C', 's'),
        ('intensity_vision', 'sims_v2_streams_seed', 'Shared vision', '#237974', '^'),
    ]
    for arm, prefix, label, color, marker in variants:
        values = condition_scores(root, arm, prefix)
        for paired in values:
            severity.plot(rates, paired, color=color, linewidth=.65, alpha=.22, zorder=1)
        severity.plot(rates, values.mean(0), color=color, marker=marker,
                      markersize=4, linewidth=1.1, label=label)
    severity.set_xticks(rates)
    severity.set_xlim(30, 70)
    severity.yaxis.set_major_locator(MaxNLocator(5))
    severity.set_xlabel('Contiguous gap width (%)')
    severity.set_ylabel('Ranked probability score')
    for axis, letter, title in [
        (effects, 'a', 'Effect of the shared correction'),
        (severity, 'b', 'Sensitivity to gap width'),
    ]:
        axis.set_title(title, loc='left', fontweight='bold', pad=7)
        axis.text(-.075, 1.07, letter, fontweight='bold', fontsize=9, transform=axis.transAxes)
        axis.tick_params(axis='x', length=3, width=.75)
    fig.legend(*severity.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.59, .008),
               ncol=3, frameon=False, columnspacing=1.0, handletextpad=.4)
    fig.canvas.draw()
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=layout_report)
    export(fig, output)


def plot_readouts(root, output):
    configure_figure()
    fig, axis = plt.subplots(figsize=(4.7, 1.8))
    fig.subplots_adjust(left=.30, right=.97, top=.87, bottom=.35)
    for variant, (label, filename, color, marker) in enumerate([
        ('Maximum pooling', 'sims_v2_minimal_visual_ablations.json', '#1B5E9C', 'o'),
        ('Text-query attention', 'sims_v2_attention_visual_ablations.json', '#237974', '^'),
    ]):
        contrasts = json.loads((root / filename).read_text(encoding='utf-8'))['contrasts']
        offset = (.5 - variant) * .28
        for row, baseline in enumerate(['independent_intensity_rps', 'fusion_rps']):
            contrast = contrasts[baseline]
            interval(axis, 1000 * contrast['rps_reduction'], 1000 * np.asarray(contrast['ci95']),
                     1 - row + offset, color, marker)
        axis.plot([], [], linestyle='none', marker=marker, color=color, markersize=4, label=label)
    axis.set_yticks([0, 1], ['Class-only fusion', 'Independent intensity'])
    axis.set_ylim(-.5, 1.5)
    axis.set_xlim(0, 16)
    axis.set_xticks([0, 4, 8, 12, 16])
    axis.set_xlabel(r'RPS reduction with sharing ($10^{-3}$)')
    axis.axvline(0, color='black', linewidth=.75)
    axis.spines['left'].set_visible(False)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', length=3, width=.75)
    fig.legend(loc='lower center', bbox_to_anchor=(.63, .005), ncol=2, frameon=False,
               columnspacing=.9, handletextpad=.25)
    export(fig, output)


def plot_seed_sensitivity(root, output):
    configure_figure()
    source = root / 'sims_v2_attention_visual_signed_diagnostics.json'
    results = json.loads(source.read_text(encoding='utf-8'))['training_seed_sensitivity']
    rows = [('independent_intensity', 'Independent intensity'),
            ('fusion', 'Class-only fusion'),
            ('intensity_shuffled', 'Shuffled intensity')]
    records = [results['contrasts'][arm] for arm, _ in rows]
    figure = seed_comparison([label for _, label in rows], results['seeds'],
                             1000 * np.array([record['rps_reduction_by_seed'] for record in records]),
                             1000 * np.array([record['rps_reduction'] for record in records]),
                             1000 * np.array([record['ci95_seeds_and_videos'] for record in records]))
    export(figure, output)


def plot_architecture(root, output):
    configure_figure()
    source = root / 'sims_v2_architecture_comparison.json'
    contrasts = json.loads(source.read_text(encoding='utf-8'))['contrasts']
    rows = [('intensity_direct_logits_rps', 'Anchor logit skip'),
            ('intensity_signed_only_rps', 'Neutral contrast'),
            ('intensity_stream_query_rps', 'Text query'),
            ('intensity_pointwise_rps', 'Temporal encoder'),
            ('intensity_unscaled_rps', 'Density normalization')]
    with np.load(root / 'primary_architecture_losses.npz', allow_pickle=False) as packets:
        seeds = packets['seeds']
    records = [contrasts[arm] for arm, _ in rows]
    figure = seed_comparison([label for _, label in rows], seeds,
                             1000 * np.array([record['reduction_by_seed'] for record in records]),
                             1000 * np.array([record['rps_reduction'] for record in records]),
                             1000 * np.array([record['ci95_seeds_and_videos'] for record in records]),
                             familywise=1000 * np.array([record['ci95_familywise'] for record in records]))
    export(figure, output)


def plot_modality_ratings(root, output):
    configure_figure()
    source = root / 'sims_v2_modality_ratings.json'
    effects = json.loads(source.read_text(encoding='utf-8'))['rps_by_agreement']
    groups = [('agree', 'Text and vision agree'),
              ('vision_agrees_with_outcome', 'Vision matches overall polarity'),
              ('vision_disagrees_with_outcome', 'Vision differs from overall polarity')]
    arms = [('intensity', 'Shared intensity', '#1B5E9C', 'D'),
            ('intensity_shuffled', 'Shuffled intensity', '#237974', 'o'),
            ('independent_intensity', 'Independent head', '#A9502C', 's')]
    fig, axis = plt.subplots(figsize=(6.2, 2.25))
    fig.subplots_adjust(left=.34, right=.98, top=.98, bottom=.31)
    positions = np.arange(len(groups))[::-1]
    for row, (group, _) in zip(positions, groups):
        for idx, (arm, _, color, marker) in enumerate(arms):
            contrast = effects[group]['relative_to_anchor'][arm]
            interval(axis, 1000 * contrast['rps_reduction'], 1000 * np.asarray(contrast['ci95']),
                     row + (1 - idx) * .21, color, marker)
    for _, label, color, marker in arms:
        axis.plot([], [], linestyle='none', marker=marker, color=color, markersize=4, label=label)
    axis.set_yticks(positions, [f'{label}\n{effects[group]["n_clips"]:,} clips' for group, label in groups])
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.set_ylim(-.45, len(groups) - .55)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.set_xlabel(r'RPS reduction relative to the text anchor ($10^{-3}$)')
    axis.spines['left'].set_visible(False)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', length=3, width=.75)
    fig.legend(loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.62, .01),
               columnspacing=1.1, handletextpad=.28)
    export(fig, output)


def plot_transfer(root, output):
    configure_figure()
    sources = [('CH-SIMS v2', 'sims_v2_simplified_ablations.json', '#1B5E9C'),
               ('CMU-MOSEI', 'mosei_simplified_ablations.json', '#1B5E9C'),
               ('OMG-Emotion', 'omg_external_valence.json', '#237974')]
    fig, axis = plt.subplots(figsize=(5.9, 1.85))
    fig.subplots_adjust(left=.28, right=.97, bottom=.31, top=.97)
    positions = np.arange(len(sources))[::-1]
    labels = []
    for y, (name, filename, color) in zip(positions, sources):
        report = json.loads((root / filename).read_text(encoding='utf-8'))
        contrast = report['contrasts']['independent_intensity_rps']
        interval(axis, 1000 * contrast['rps_reduction'], 1000 * np.asarray(contrast['ci95_seeds_and_videos']),
                 y, color, 'D')
        labels.append(f'{name}\n{report["n_videos"]} videos, {report["runs"]} seeds')
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.set_yticks(positions, labels)
    axis.set_ylim(-.53, len(sources) - .47)
    axis.margins(x=.07)
    axis.xaxis.set_major_locator(MaxNLocator(7))
    axis.spines['left'].set_visible(False)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', length=3, width=.75)
    axis.set_xlabel(r'RPS reduction with shared intensity ($10^{-3}$)')
    export(fig, output)


def plot_score_gain(root, output):
    configure_figure()
    ten_seed = json.loads((root / 'score_gain_effects.json').read_text(encoding='utf-8'))['cohorts']
    three_seed = json.loads((root / 'mosei_score_gain_effects.json').read_text(encoding='utf-8'))['cohorts']
    reports = [('CH-SIMS v2', ten_seed['ch_sims_v2']),
               ('CMU-MOSEI', three_seed['cmu_mosei']),
               ('OMG-Emotion', ten_seed['omg_emotion'])]
    fig, (gains, effects) = plt.subplots(1, 2, sharey=True, figsize=(6.5, 2.16),
                                       gridspec_kw={'width_ratios': (1, 1.45)})
    fig.subplots_adjust(left=.195, right=.971, top=.875, bottom=.32, wspace=.2)
    positions = np.arange(len(reports))[::-1]
    blue, bronze = '#216392', '#945721'
    for y, (corpus, report) in zip(positions, reports):
        for head, offset, color in [('shared', .14, blue), ('independent', -.14, bronze)]:
            values = np.sort(report['methods'][head]['gain_by_seed'])
            rows = y + offset + np.resize([0., -.097, .097, -.047, .047], len(values))
            replicate_ticks(gains, values, rows, color)
        for name, offset, color, marker in [('shared_old', .14, blue, 'D'),
                                             ('independent_calibrated', -.14, bronze, 's')]:
            if isinstance(output, PlatePart) and corpus != 'CMU-MOSEI' and name == 'shared_old':
                continue
            effect = report['reductions'][name]
            interval(effects, 1000 * effect['mae_reduction'], 1000 * np.asarray(effect['ci95_familywise']),
                     y + offset, color, marker)
    gains.set_yticks(positions, [row[0] for row in reports])
    gains.set_ylim(-.53, 2.53)
    gains.set_xlim(-.03, 1.05)
    gains.set_xticks(np.linspace(0., 1., 5), ['0', '0.25', '0.50', '0.75', '1'])
    gains.set_xlabel(r'Intensity shift multiplier $\alpha$')
    effects.axvline(0, color='black', linestyle=(0, (3, 3)), linewidth=.75, zorder=1)
    effects.xaxis.set_major_locator(MaxNLocator(6))
    effects.tick_params(labelleft=False)
    effects.set_xlabel(r'MAE reduction after scaling ($10^{-3}$)')
    for axis, title in [(gains, 'a   Tuning estimates'), (effects, 'b   Held-out intensity error')]:
        axis.set_title(title, loc='left', fontsize=9, weight='bold', pad=5)
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.tick_params(axis='x', length=3, width=.75)
    gains.legend(handles=[
        Line2D([], [], marker='|', color=blue, linestyle='none', markersize=4, label='Shared'),
        Line2D([], [], marker='|', color=bronze, linestyle='none', markersize=4, label='Independent'),
    ], loc='upper center', bbox_to_anchor=(.5, -.35), frameon=False, ncol=2, fontsize=7,
                 handletextpad=.25, columnspacing=.85)
    effects.legend(handles=[
        Line2D([], [], marker='D', color=blue, linestyle='none', markersize=4, label='vs. unscaled shared'),
        Line2D([], [], marker='s', color=bronze, linestyle='none', markersize=4, label='vs. scaled independent'),
    ], loc='upper center', bbox_to_anchor=(.5, -.35), frameon=False, ncol=2, fontsize=7,
                   handletextpad=.25, columnspacing=.85)
    export(fig, output)


def plot_fixed_transcripts(root, output):
    configure_figure()
    report = json.loads((root / 'ravdess_effects.json').read_text(encoding='utf-8'))
    anchor = report['secondary']['versus_anchor_balanced_rps']
    independent = report['primary_shared_versus_independent_balanced_rps']
    shuffled = report['secondary']['versus_shuffle_balanced_rps']
    fig, (visual, objective) = plt.subplots(1, 2, figsize=(6.5, 1.94),
                                            gridspec_kw={'width_ratios': (1., 1.25)})
    fig.subplots_adjust(left=.10, right=.972, top=.82, bottom=.28, wspace=.59)
    for axis, y, effect, color in ((visual, .41, anchor, '#237974'),
                                    (objective, .64, independent, '#1B5E9C'),
                                    (objective, .09, shuffled, '#945721')):
        actors = 1000 * np.asarray(effect['by_actor'])
        offsets = np.linspace(-.096, .096, len(actors))
        replicate_ticks(axis, actors, y + offsets, color)
        interval(axis, 1000 * effect['reduction'], 1000 * np.asarray(effect['ci']),
                 y - .176, color, 'D')
    for axis, title in [(visual, 'a   Contribution of facial cues'),
                        (objective, 'b   Effect of score supervision')]:
        axis.set_title(title, loc='left', fontweight='bold', pad=6)
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.set_ylim(-.21, .89)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_xlabel(r'Balanced RPS reduction ($10^{-3}$)')
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0)
    visual.set_yticks([])
    objective.set_yticks([.64, .09], ['Independent\nscore', 'Shuffled\nintensity'])
    export(fig, output)


def plot_moud_transfer(root, output):
    configure_figure()
    report = json.loads((root / 'moud_frozen_transfer_full.json').read_text(encoding='utf-8'))
    comparisons = [
        ('Shared vs. independent', report['primary_shared_versus_independent']),
        ('Shared vs. text anchor', report['secondary']['versus_anchor']),
        ('Capped vs. shared', report['secondary']['clipping']),
    ]
    labels, effects = zip(*comparisons, strict=True)
    seeds = report['seeds']
    gains = 1000 * np.array([effect['by_seed'] for effect in effects])
    fig, (axis, summary) = plt.subplots(1, 2, figsize=(6.45, 2.06),
                                       gridspec_kw={'width_ratios': (3.6, 1)}, sharey=True)
    fig.subplots_adjust(left=.223, right=.978, bottom=.225, top=.80, wspace=.075)
    bound = 5 * np.ceil(np.max(np.abs(gains)) / 5)
    axis.imshow(gains, cmap=EFFECTS, vmin=-bound, vmax=bound, aspect='auto', interpolation='nearest')
    for (row, column), gain in np.ndenumerate(gains):
        axis.text(column, row, f'{gain:.1f}'.replace('-', '−'),
                  color='white' if abs(gain) > .73 * bound else INK,
                  fontsize=6.55, va='center', ha='center')
    axis.vlines(np.arange(.5, len(seeds) - 1), -.5, len(labels) - .5, colors='white', linewidth=.55)
    axis.hlines(np.arange(.5, len(labels) - 1), -.5, len(seeds) - .5, colors='white', linewidth=.75)
    axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds], fontsize=6.65)
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(length=0, pad=4)
    axis.spines[['left', 'bottom']].set_visible(False)
    axis.set_xlabel('Paired source-fit seed · RPS gains × 1,000', labelpad=3)
    axis.set_title('a', loc='left', fontweight='bold', pad=5)
    gain_tiles(summary, effects, gain_key='rps_reduction', interval_key='ci95', color_limit=bound)
    summary.tick_params(labelleft=False)
    summary.text(.5, 1.028, 'Mean [95% CI]', fontsize=7.2, ha='center', va='bottom',
                 transform=summary.transAxes)
    export(fig, output)


def plot_ordinal_controls(root, output):
    configure_figure()
    source = json.loads((root / 'ordinal_controls.json').read_text(encoding='utf-8'))['cohorts']
    transfer = json.loads((root / 'moud_ordinal_controls.json').read_text(encoding='utf-8'))
    comparisons = [('rps_only', 'RPS only'), ('ce_plus_rps', 'Cross-entropy + RPS'),
                   ('class_mean_intensity', 'Class mean intensity')]
    cohorts = [('CH-SIMS v2', source['ch_sims_v2']['shared_rps_reductions'], 'reduction', 'ci95_familywise'),
               ('OMG-Emotion', source['omg_emotion']['shared_rps_reductions'], 'reduction', 'ci95_familywise'),
               ('MOUD frozen transfer', transfer['ordinal_controls'], 'rps_reduction', 'ci95')]
    fig, axis = plt.subplots(figsize=(6.45, 3.45))
    fig.subplots_adjust(left=.29, right=.975, bottom=.165, top=.965)
    label_coordinates = axis.get_yaxis_transform()
    for index, (name, estimates, effect_key, interval_key) in enumerate(cohorts):
        group_y = 3 + 4.35 * (len(cohorts) - index - 1)
        axis.text(-.40, group_y, name, transform=label_coordinates, weight='bold', va='center')
        for offset, (key, label) in enumerate(comparisons, start=1):
            y = group_y - offset
            estimate = estimates[key]
            axis.text(-.37, y, label, transform=label_coordinates, va='center')
            interval(axis, 1000 * estimate[effect_key], 1000 * np.asarray(estimate[interval_key]),
                     y, '#1B5E9C', 'D')
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.spines['left'].set_visible(False)
    axis.set_yticks([])
    axis.set_ylim(-.55, 4.35 * (len(cohorts) - 1) + 3.5)
    axis.margins(x=.09)
    axis.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    axis.set_xlabel(r'RPS difference (reference $-$ per-clip shared, $10^{-3}$)')
    export(fig, output)


def plot_frozen_dialogue(root, output):
    configure_figure()
    review = json.loads((root / 'moud_ordinal_controls.json').read_text(encoding='utf-8'))
    dialogue = json.loads((root / 'meld_frozen_transfer.json').read_text(encoding='utf-8'))
    cohorts = [
        ('MOUD reviews', review, review['class_mean_factorial']['sharing_with_class_means'], '#1B5E9C', 'D'),
        ('MELD dialogue', dialogue, dialogue['primary'], '#96502A', 'o'),
    ]
    fig, axis = plt.subplots(figsize=(6.45, 2.38))
    fig.subplots_adjust(left=.19, right=.975, bottom=.215, top=.80)
    positions = (3, 2, 1, 0)
    for offset, (_, report, primary, color, marker) in zip((.125, -.125), cohorts):
        interval(axis, 1000 * primary['rps_reduction'], 1000 * np.asarray(primary['ci95']),
                 positions[0] + offset, color, marker)
        independent = report['methods']['independent_class_mean']['gapped']['rps_by_class']
        shared = report['methods']['intensity_class_mean']['gapped']['rps_by_class']
        classes = ['negative', 'neutral', 'positive']
        gains = np.array([independent[name] - shared[name] for name in classes])
        axis.barh(np.asarray(positions[1:]) + offset, 1000 * gains, height=.18, color=color, alpha=.78, zorder=3)
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.set_yticks(positions, ['Overall', 'Negative', 'Neutral', 'Positive'])
    axis.set_ylim(-.46, 3.46)
    axis.spines['left'].set_visible(False)
    axis.tick_params(axis='y', length=0, pad=6)
    axis.margins(x=.12)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.set_xlabel(r'RPS reduction with class mean sharing ($10^{-3}$)')
    handles = [Line2D([], [], color=color, marker=marker, linestyle='none', markersize=4, label=name)
               for name, _, _, color, marker in cohorts]
    axis.legend(handles=handles, loc='lower right', bbox_to_anchor=(1., 1.045), ncol=2,
                columnspacing=1.25, handletextpad=.38, frameon=False, fontsize=8)
    export(fig, output)


def plot_probability_pool(root, output):
    configure_figure()
    report = json.loads((root / 'polarity_pool.json').read_text(encoding='utf-8'))
    calibration = {**report['local']['primary'], **report['local']['secondary']}
    local_rows = [
        ('One weight', calibration['versus_shared_one']),
        ('Recentered text anchor', calibration['versus_anchor_full']),
        ('Independent head, two weights', calibration['versus_independent_two']),
        ('Shared head, full correction', calibration['versus_shared_full']),
    ]
    frozen = report['frozen']
    groups = [('MELD, local calibration', local_rows, '#1B5E9C', 'D')]
    for cohort, name in (('meld', 'MELD'), ('moud', 'MOUD')):
        estimates = frozen[cohort]['contrasts']
        rows = [('One weight', estimates['versus_shared_one']), ('Text anchor', estimates['versus_anchor'])]
        groups.append((f'{name}, source calibration', rows, '#96502A', 'o'))

    figure, axis = plt.subplots(figsize=(6.45, 3.35))
    figure.subplots_adjust(left=.32, right=.975, bottom=.15, top=.97)
    label_coordinates = axis.get_yaxis_transform()
    y = 10.6
    for group, rows, color, marker in groups:
        axis.text(-.46, y, group, transform=label_coordinates, va='center', weight='bold')
        for label, row in rows:
            y -= .95
            axis.text(-.43, y, label, transform=label_coordinates, va='center')
            interval(axis, 1000 * row['reduction'], 1000 * np.asarray(row['ci95_familywise']),
                     y, color, marker)
        y -= 1.30
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.set_ylim(y + .55, 11.1)
    axis.spines['left'].set_visible(False)
    axis.set_yticks([])
    axis.margins(x=.10)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.set_xlabel(r'RPS reduction with two weights ($10^{-3}$)')
    export(figure, output)


def plot_published_seeds(root, output):
    configure_figure()
    report = json.loads((root / 'sims_v2_published_seeds.json').read_text(encoding='utf-8'))
    comparators = [('tfr_joint', 'TFR-Net joint'), ('lnln_joint', 'LNLN joint'), ('rmnet_core', 'RM-Net core')]
    styles = [
        ('Fixed pooling', '#1B5E9C', 'D', False),
        ('Fixed pooling after scaling', '#1B5E9C', 'D', True),
        ('Attention', '#96502A', 'o', False),
        ('Attention after scaling', '#96502A', 'o', True),
    ]
    figure, axis = plt.subplots(figsize=(6.45, 2.75))
    figure.subplots_adjust(left=.19, right=.975, bottom=.185, top=.785)
    rows = (3.5, 1.75, 0)
    for (name, _), row in zip(comparators, rows):
        sensitivity = report['sensitivity_rps'][name]
        estimates = [report['primary_rps'][name], sensitivity['fixed_pooling_scaled'],
                     sensitivity['attention_raw'], sensitivity['attention_scaled']]
        for offset, estimate, (_, color, marker, scaled) in zip((.42, .14, -.14, -.42), estimates, styles):
            low, high = 1000 * np.array(estimate['ci95_familywise'])
            y = row + offset
            axis.hlines(y, low, high, color=color, linewidth=1.05)
            axis.plot(1000 * estimate['reduction'], y, marker=marker, color=color, linestyle='none',
                      markersize=4, markerfacecolor='white' if scaled else color, zorder=3)
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.set_yticks(rows, [label for _, label in comparators])
    axis.set_ylim(-.68, 4.18)
    axis.tick_params(axis='y', length=0, pad=5)
    axis.spines['left'].set_visible(False)
    axis.margins(x=.09)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.set_xlabel(r'RPS difference (comparator $-$ shared model, $10^{-3}$)')
    handles = [Line2D([], [], label=label, color=color, marker=marker, markersize=4,
                      markerfacecolor='white' if scaled else color) for label, color, marker, scaled in styles]
    axis.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.07), ncol=2,
                columnspacing=1.4, handletextpad=.55, frameon=False)
    export(figure, output)


def plot_published_cohorts(root, output, *, include_raw=True):
    configure_figure()
    sims = json.loads((root / 'sims_v2_published_seeds.json').read_text(encoding='utf-8'))
    omg = json.loads((root / 'omg_published_seeds.json').read_text(encoding='utf-8'))
    baselines = [('tfr_joint', 'TFR-Net'), ('lnln_joint', 'LNLN'), ('rmnet_core', 'RM-Net')]
    figure, axes = plt.subplots(1, 2, figsize=(6.55, 2.38), sharex=True, sharey=True)
    figure.subplots_adjust(left=.126, right=.973, bottom=.24, top=.729 if include_raw else .81, wspace=.198)
    for axis, (letter, corpus, report) in zip(axes, (('a', 'CH-SIMS v2', sims), ('b', 'OMG-Emotion', omg))):
        for row, (name, _) in zip((2, 1, 0), baselines):
            scaled = report['sensitivity_rps'][name]
            if corpus == 'CH-SIMS v2':
                scaled = scaled['fixed_pooling_scaled']
            estimates = ((.17, report['primary_rps'][name], False), (-.17, scaled, True)) if include_raw else (
                (0., scaled, True),)
            for offset, value, unfilled in estimates:
                y = row + offset
                low, high = 1000 * np.asarray(value['ci95_familywise'])
                axis.hlines(y, low, high, linewidth=1.1, color='#1B5E9C')
                axis.plot(1000 * value['reduction'], y, linestyle='none', color='#1B5E9C',
                          marker='D', markersize=4, markerfacecolor='white' if unfilled else '#1B5E9C', zorder=3)
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.set_title(f'{letter}  {corpus}', loc='left', fontweight='bold', pad=7)
        if not include_raw:
            axis.text(.15, 1.04, corpus, transform=axis.transAxes, va='bottom', fontsize=8)
        axis.set_ylim(-.52, 2.52)
        axis.tick_params(axis='y', length=0, pad=5)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.margins(x=.082)
    axes[0].set_yticks((2, 1, 0), [label for _, label in baselines])
    figure.supxlabel('RPS reduction with compact model × 1,000', y=.038, fontsize=8)
    handles = [Line2D([], [], color='#1B5E9C', marker='D', markersize=4,
                      markerfacecolor=face, label=label)
               for face, label in (('#1B5E9C', 'Original probabilities'), ('white', 'After temperature scaling'))]
    if include_raw:
        figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, .987), ncol=2,
                      frameon=False, handlelength=1.4, handletextpad=.48, columnspacing=1.5)
    export(figure, output)


def plot_frozen_lnln_heads(root, output):
    configure_figure()
    report = json.loads((root / 'frozen_lnln_heads.json').read_text(encoding='utf-8'))['cases']
    cohorts = [('sims_v2', 'CH-SIMS v2'), ('omg', 'OMG-Emotion')]
    comparators = [('frozen', 'Original LNLN', '#1B5E9C', 'D'),
                   ('independent', 'Independent correction', '#96502A', 'o')]
    figure, axes = plt.subplots(1, 2, figsize=(6.55, 2.16), sharey=True)
    figure.subplots_adjust(left=.16, right=.979, bottom=.245, top=.708, wspace=.24)
    for axis, (title, key, metric) in zip(axes, (('a  Probability error', 'rps_reductions', 'RPS'),
                                                ('b  Intensity error', 'mae_reductions', 'MAE'))):
        for row, (name, _) in zip((1.18, 0), cohorts):
            for offset, (reference, _, color, marker) in zip((.175, -.175), comparators):
                result = report[name][key][reference]
                interval(axis, 1000 * result['reduction'], 1000 * np.asarray(result['ci95_familywise']),
                         row + offset, color, marker)
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.set_title(title, loc='left', fontweight='bold', pad=7)
        axis.set_ylim(-.45, 1.63)
        axis.tick_params(axis='y', length=0, pad=5)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.margins(x=.095)
        axis.set_xlabel(f'{metric} reduction × 1,000')
    axes[0].set_yticks((1.18, 0), [label for _, label in cohorts])
    handles = [Line2D([], [], label=label, color=color, marker=marker, markersize=4)
               for _, label, color, marker in comparators]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.59, .987), ncol=2, frameon=False,
                  handlelength=1.5, handletextpad=.5, columnspacing=1.5)
    export(figure, output)


def plot_conditional_intensity(root, output, alignment_script, layout_report):
    configure_figure()
    report = json.loads((root / 'conditional_intensity_effects.json').read_text(encoding='utf-8'))['cohorts']
    cohorts = [('ch_sims_v2', 'CH-SIMS v2', '#1B5E9C', 'o'),
               ('omg_emotion', 'OMG-Emotion', '#96502A', 'D')]
    panels = [
        ('a  Probability forecasts', 'rps_reductions', 'RPS reduction',
         [('intensity', 'Huber supervision'), ('conditional_independent', 'Independent tilt'),
          ('conditional_shuffled', 'Permuted targets')]),
        ('b  Within-class intensity', 'within_class_mae_reductions', 'MAE reduction',
         [('conditional_shared', 'Shared tilt'), ('conditional_independent', 'Independent tilt')]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(6.55, 2.38))
    figure.subplots_adjust(left=.17, right=.975, bottom=.232, top=.754, wspace=.45)
    for axis, (title, family, metric, comparisons) in zip(axes, panels):
        rows = np.linspace(2.0, 0.0, len(comparisons))
        for row, (comparison, _) in zip(rows, comparisons):
            for dy, (cohort, _, color, marker) in zip((.13, -.13), cohorts):
                result = report[cohort][family][comparison]
                interval(axis, 1000 * result['reduction'], 1000 * np.asarray(result['ci95_familywise']),
                         row + dy, color, marker)
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.set_yticks(rows, [label for _, label in comparisons])
        axis.tick_params(axis='y', length=0, pad=5)
        axis.set_title(title, loc='left', fontweight='bold', pad=7)
        axis.set_ylim(-.45, 2.43)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.margins(x=.1)
        axis.set_xlabel(f'{metric} × 1,000')
    handles = [Line2D([], [], label=label, color=color, marker=marker, markersize=4)
               for _, label, color, marker in cohorts]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .949), ncol=2,
                  frameon=False, handlelength=1.4, handletextpad=.5, columnspacing=1.6)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_reference_distillation(root, output, alignment_script, layout_report):
    configure_figure()
    report = json.loads((root / 'distillation_effects.json').read_text(encoding='utf-8'))['cohorts']
    references = [('original', 'Original shared head'), ('native', 'Absolute native'),
                  ('native_prior', 'Prior-corrected native'), ('anchored_matched', 'Re-anchored same view'),
                  ('independent', 'Independent score head'), ('teacher', 'RM-Net teacher')]
    cohorts = [('sims_v2', 'a  CH-SIMS v2'), ('omg', 'b  OMG-Emotion')]
    figure, axes = plt.subplots(1, 2, figsize=(6.55, 3.05), sharey=True)
    figure.subplots_adjust(left=.264, right=.975, bottom=.25, top=.89, wspace=.05)
    for axis, (cohort, title) in zip(axes, cohorts, strict=True):
        estimates = report[cohort]['primary_rps'] | report[cohort]['secondary_rps']
        image = gain_tiles(axis, [estimates[key] for key, _ in references])
        axis.hlines(3.5, 0, 1, colors='white', linewidth=2.)
        axis.set_title(title, loc='left', fontweight='bold', pad=6)
    axes[0].set_yticks(range(len(references)), [label for _, label in references])
    scale_axis = figure.add_axes([.445, .135, .39, .025])
    scale = figure.colorbar(image, cax=scale_axis, orientation='horizontal', ticks=[-25, 0, 25])
    scale.outline.set_visible(False)
    scale.ax.tick_params(length=2, pad=1, labelsize=7.1)
    scale.set_label('Reference minus student RPS × 1,000', size=7.5, labelpad=1.5)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_cremad(root, output, alignment_script, layout_report):
    configure_figure()
    report = json.loads((root / 'cremad_effects.json').read_text(encoding='utf-8'))
    comparisons = [('independent_intensity', 'Independent head'), ('fusion', 'Class only'),
                   ('intensity_shuffled', 'Shuffled intensity')]
    rows = np.arange(3)[::-1]
    figure = plt.figure(figsize=(6.55, 3.97))
    grid = figure.add_gridspec(2, 2, left=.178, right=.91, bottom=.136, top=.873,
                              height_ratios=(1.30, 1.), hspace=.64, wspace=.40)
    clips = figure.add_subplot(grid[0, :])
    effects, per_class = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])
    with np.load(root / 'cremad_evidence.npz', allow_pickle=False) as packet:
        shared, independent, labels = (packet[key] for key in ('intensity', 'independent_intensity', 'y'))
    rps_gain = (ranked_probability_score(independent, labels)
                - ranked_probability_score(shared, labels)).mean((0, 1))
    order = np.lexsort((rps_gain, labels))
    counts = np.bincount(labels, minlength=3)
    edges = np.r_[0, counts.cumsum()]
    names = ('Negative', 'Neutral', 'Positive')
    changes = (shared - independent).mean((0, 1)).T
    image = clips.imshow(changes[:, order], extent=(0, len(labels), 2.5, -.5), cmap=EFFECTS,
                         vmin=-.15, vmax=.15, aspect='auto', interpolation='none')
    clips.vlines(edges[1:-1], -.5, 2.5, color='white', linewidth=1.05)
    clips.set_xticks(edges[:-1] + counts / 2, [f'{name}\n{count:,} clips' for name, count in zip(names, counts)])
    clips.set_yticks((0, 1, 2), names)
    clips.tick_params(length=0, pad=4)
    clips.set_xlabel('Clips grouped by rated polarity')
    clips.spines[['left', 'bottom']].set_visible(False)
    clips.set_title('a', loc='left', fontweight='bold', pad=6)
    key = clips.inset_axes([1.025, .12, .015, .76])
    colorbar = figure.colorbar(image, cax=key, ticks=[-.15, 0, .15], extend='both')
    colorbar.set_label('Probability change', size=7.5, labelpad=3)
    colorbar.ax.tick_params(length=2, pad=1.8, labelsize=7.2)
    colorbar.outline.set_visible(False)
    for row, (key, _) in zip(rows, comparisons):
        trial = report['balanced_rps_reductions'][key]
        interval(effects, trial['reduction'] * 1000, np.asarray(trial['ci95_familywise']) * 1000,
                 row, TEAL, 'D' if key == 'independent_intensity' else 'o')
    effects.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    effects.set_yticks(rows, [label for _, label in comparisons])
    effects.set_xlabel('Balanced RPS reduction × 1,000')
    effects.set_title('b', loc='left', fontweight='bold', pad=6)
    polarities = report['balanced_rps_reductions']['independent_intensity']['by_polarity_exploratory']
    for row, polarity in zip(rows, ('negative', 'neutral', 'positive')):
        trial = polarities[polarity]
        interval(per_class, trial['reduction'] * 1000, np.asarray(trial['ci95_familywise']) * 1000,
                 row, TEAL, 'D')
    per_class.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    per_class.set_yticks(rows, ['Negative', 'Neutral', 'Positive'])
    per_class.set_xlabel('Within-polarity RPS reduction × 1,000')
    per_class.set_title('c', loc='left', fontweight='bold', pad=6)
    for axis in (effects, per_class):
        axis.spines['left'].set_visible(False)
        axis.set_ylim(-.45, 2.45)
        axis.tick_params(axis='y', length=0, pad=4)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.margins(x=.12)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_view_stability(root, output, alignment_script, layout_report):
    configure_figure()
    cohorts = (('cremad', 'CREMA-D'), ('ravdess', 'RAVDESS'), ('sims_v2', 'CH-SIMS v2'), ('omg', 'OMG-Emotion'))
    rows = np.arange(len(cohorts))[::-1]
    effects = (('shared', 'Original head', '#1B5E9C', 'D'),
               ('support_threshold', 'Support-matched threshold', '#96502A', 'o'),
               ('signed_average', 'Five-view signed mean', '#237974', 's'))
    fig, (paired, ceiling) = plt.subplots(1, 2, figsize=(6.7, 2.9), sharey=True)
    fig.subplots_adjust(left=.165, right=.979, bottom=.185, top=.784, wspace=.20)

    for row, (key, _) in zip(rows, cohorts, strict=True):
        report = json.loads((root / f'view_stability_{key}.json').read_text(encoding='utf-8'))
        for offset, (test, _, color, marker) in zip((.22, 0, -.22), effects, strict=True):
            result = report['balanced_rps_reductions'][test]
            interval(paired, result['reduction'] * 1000, np.asarray(result['ci95_familywise']) * 1000,
                     row + offset, color, marker)
        bound = report['contraction_ceiling']
        ceiling.barh(row, bound['balanced_rps'] * 1000, height=.37, color='#315879', zorder=2)
        ceiling.plot(bound['by_class'][1] * 1000, row, marker='D', markerfacecolor='white',
                     markeredgecolor='#8A486D', markersize=4.8, linestyle='none', zorder=3)

    paired.set_yticks(rows, [label for _, label in cohorts])
    paired.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    paired.set_xlim(-3.95, 2.25)
    paired.set_xticks([-3, -2, -1, 0, 1, 2])
    paired.set_xlabel('RPS reduction × 1,000')
    paired.set_title('a  Subset-based shrinkage', loc='left', fontweight='bold', pad=8)
    ceiling.set_xlim(0, 42)
    ceiling.set_xticks([0, 10, 20, 30, 40])
    ceiling.set_xlabel('Attainable RPS reduction × 1,000')
    ceiling.set_title('b  Ceiling with known labels', loc='left', fontweight='bold', pad=8)
    for axis in (paired, ceiling):
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0, pad=4)
        axis.set_ylim(-.53, 3.53)

    handles = [Line2D([], [], marker=marker, color=color, linestyle='none', markersize=4.5, label=label)
               for _, label, color, marker in effects]
    handles += [Line2D([], [], linewidth=4, color='#315879', label='Balanced ceiling'),
                Line2D([], [], marker='D', markerfacecolor='white', markeredgecolor='#8A486D',
                       linestyle='none', markersize=4.5, label='Neutral ceiling')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.536, .985), ncols=3,
               frameon=False, handletextpad=.35, columnspacing=1.4, borderaxespad=0.)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=layout_report)
    export(fig, output)


def plot_class_weighting(root, output, alignment_script, layout_report):
    configure_figure()
    cohorts = (('sims_v2', 'CH-SIMS v2', '#1B5E9C', 'D'),
               ('omg', 'OMG-Emotion', '#237974', 'o'))
    references = (('shared', 'Original shared head'), ('global_matched', 'Global matched loss'),
                  ('text_independent', 'Independent score head'), ('text_weight_only', 'Text weights only'),
                  ('text_offset_only', 'Text offset only'), ('inverse_frequency', 'Inverse-frequency weights'))
    contrasts = (('compact', 'Compact head'), ('rmnet', 'RM-Net core'), ('interaction', 'Difference of effects'))
    fig, (conditional, factorial) = plt.subplots(1, 2, figsize=(6.8, 3.2))
    fig.subplots_adjust(left=.205, right=.983, bottom=.165, top=.827, wspace=.72)
    rows, factor_rows = np.arange(5, -1, -1), np.arange(4.45, 0., -1.85)
    for offset, (cohort, _, color, marker) in zip((.14, -.14), cohorts, strict=True):
        weights = json.loads((root / f'text_weighted_loss_{cohort}.json').read_text(encoding='utf-8'))
        controls = json.loads((root / f'rmnet_weighting_effects_{cohort}.json').read_text(encoding='utf-8'))
        for row, (name, _) in zip(rows, references, strict=True):
            result = weights['rps_reductions'][name]
            interval(conditional, result['reduction'] * 1000, np.asarray(result['ci95_familywise']) * 1000,
                     row + offset, color, marker)
        for row, (name, _) in zip(factor_rows, contrasts, strict=True):
            ordinary = controls['weighting_effects'][name]
            balanced = controls['balanced_rps_effects_exploratory'][name]
            interval(factorial, ordinary['reduction'] * 1000, np.asarray(ordinary['ci95_familywise']) * 1000,
                     row + offset + .40, color, marker)
            y = row + offset - .18
            factorial.hlines(y, *(np.asarray(balanced['ci95_familywise']) * 1000), color=color, linewidth=1.)
            factorial.plot(balanced['reduction'] * 1000, y, marker=marker, markersize=4.,
                           markeredgecolor=color, markerfacecolor='white', linestyle='none', zorder=3)
    for axis, points, keys, title in (
            (conditional, rows, references, 'a  Text-conditioned loss'),
            (factorial, factor_rows, contrasts, 'b  Inverse-frequency effects')):
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.set_yticks(points, [label for _, label in keys])
        axis.set_ylim(-.5, 5.5)
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0, pad=4)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.margins(x=.09)
        axis.set_xlabel('RPS reduction × 1,000')
        axis.set_title(title, loc='left', fontweight='bold', pad=7)
    legend = [Line2D([], [], marker=marker, color=color, linestyle='none', markersize=4.5, label=label)
              for _, label, color, marker in cohorts]
    legend += [Line2D([], [], marker='o', color='black', linestyle='none', markersize=4.5, label='Ordinary RPS'),
               Line2D([], [], marker='o', color='black', markerfacecolor='white', linestyle='none',
                      markersize=4.5, label='Class-balanced RPS')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.553, .984), ncols=4,
               frameon=False, columnspacing=1.6, borderaxespad=0.)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=layout_report)
    export(fig, output)


def plot_benchmark_components(root, output, alignment_script, layout_report):
    configure_figure()
    cohorts = (('sims_v2', 'CH-SIMS v2'), ('omg', 'OMG-Emotion'))
    references = (('rmnet_uniform', 'RM-Net'), ('lnln_joint', 'LNLN'))
    groups = (('no_vision', 'without vision', '#2766A6'),
              ('visual_changes', 'change with vision', '#16736D'))
    penalties = (('compact', 'SentiCouple', '#2766A6'), ('rmnet', 'RM-Net', '#16736D'),
                 ('interaction', 'RM-Net − SentiCouple', '#824369'))
    figure, axes = plt.subplots(2, 2, figsize=(6.65, 3.55), sharex='row', sharey='row')
    figure.subplots_adjust(left=.235, right=.988, top=.91, bottom=.16, hspace=.68, wspace=.16)
    comparison_rows = (3.3, 2.3, .95, -.05)
    penalty_rows = (2., 1., 0.)
    for column, (cohort, title) in enumerate(cohorts):
        upper, lower = axes[:, column]
        components = json.loads((root / f'benchmark_components_{cohort}.json').read_text(encoding='utf-8'))
        order = json.loads((root / f'temporal_order_{cohort}.json').read_text(encoding='utf-8'))
        for start, (name, _) in enumerate(references):
            for offset, (part, _, color) in enumerate(groups):
                result = components['comparisons'][name]['grouped'][part]
                signed_bar(upper, result['reduction'] * 1000, np.asarray(result['ci95_familywise']) * 1000,
                           comparison_rows[2 * start + offset], color)
        for row, (model, _, color) in zip(penalty_rows, penalties, strict=True):
            result = order['random_penalties'][model]
            signed_bar(lower, result['reduction'] * 1000, np.asarray(result['ci95_familywise']) * 1000,
                       row, color)
        # Each row shares limits across corpora so the intervals have a common scale.
        upper.set(xlim=(-22, 57), xticks=(-20, 0, 20, 40), ylim=(-.62, 3.87))
        lower.set(xlim=(-2.61, 2.69), xticks=(-2, -1, 0, 1, 2), ylim=(-.5, 2.5))
        upper.set_title(f'{"ab"[column]}  {title}', loc='left', fontweight='bold', fontsize=8.5, pad=6)
        lower.set_title('cd'[column], loc='left', fontweight='bold', fontsize=8.5, pad=6)
        for axis, xlabel in zip((upper, lower), ('Contribution to RPS gap × 1,000',
                                                'RPS change after shuffling × 1,000'), strict=True):
            axis.axvline(0, color='black', linewidth=.72, linestyle=(0, (3, 3)), zorder=1)
            axis.spines['left'].set_visible(False)
            axis.tick_params(axis='y', length=0, pad=4)
            axis.tick_params(axis='x', labelsize=8, length=3, pad=3)
            axis.set_xlabel(xlabel, labelpad=4)
    axes[0, 0].set_yticks(comparison_rows, [f'{reference}, {label}' for _, reference in references
                                         for _, label, _ in groups], fontsize=7.5)
    axes[1, 0].set_yticks(penalty_rows, [label for _, label, _ in penalties], fontsize=7.5)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output, align_panels=True)


def plot_dispersion_pooling(root, output, alignment_script, layout_report):
    configure_figure()
    cohorts = (('sims_v2', 'CH-SIMS v2', '#2766A6', 's', .15),
               ('omg', 'OMG-Emotion', '#16736D', 'o', -.15))
    changes = (('pooling', 'original', 'Max to SD, 48'),
               ('pooling', 'wide', 'Max to SD, 96'),
               ('widening', 'max', '48 to 96, max'),
               ('widening', 'std', '48 to 96, SD'))
    sharing = (('sharing', 'original_max', '48, mean + max'),
               ('sharing', 'original_std', '48, mean + SD'),
               ('sharing', 'wide_max', '96, mean + max'),
               ('sharing', 'wide_std', '96, mean + SD'))
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.52), sharex=True)
    figure.subplots_adjust(left=.158, right=.982, bottom=.207, top=.772, wspace=.65)
    y = (3.28, 2.32, .99, .03)
    limits = [0.]
    for cohort, _, color, marker, offset in cohorts:
        report = json.loads((root / f'dispersion_pooling_{cohort}.json').read_text(encoding='utf-8'))
        for axis, comparisons in zip(axes, (changes, sharing), strict=True):
            for row, (family, key, _) in zip(y, comparisons, strict=True):
                effect = report[family][key]
                bounds = np.asarray(effect['ci95_familywise']) * 1000
                limits.extend(bounds)
                interval(axis, effect['reduction'] * 1000, bounds, row + offset, color, marker)
    margin = np.ptp(limits) * .075
    for axis, rows, title in zip(axes, (changes, sharing),
                                 ('a  Pooling and width', 'b  Shared versus independent'), strict=True):
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.set_yticks(y, [label for _, _, label in rows], fontsize=7.8)
        axis.tick_params(axis='y', length=0, pad=4)
        axis.set_ylim(-.49, 3.81)
        axis.set_xlim(min(limits) - margin, max(limits) + margin)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_xlabel('RPS reduction × 1,000')
        axis.set_title(title, loc='left', fontweight='bold', fontsize=8.5, pad=7)
    handles = [Line2D([], [], color=color, marker=marker, markersize=4.5, linestyle='none', label=label)
               for _, label, color, marker, _ in cohorts]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .945),
                  ncols=2, frameon=False, borderaxespad=0., columnspacing=2.)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_seed_ensemble(root, output, alignment_script, layout_report):
    configure_figure()
    references = (('tfr_joint', 'TFR-Net', '#A84527', '^'), ('lnln_joint', 'LNLN', '#2766A6', 's'),
                  ('rmnet_uniform', 'RM-Net', '#16736D', 'o'))
    cohorts = (('sims_v2', 'CH-SIMS v2', .13, 'D'), ('omg', 'OMG-Emotion', -.13, 'o'))
    figure, axes = plt.subplots(1, 3, figsize=(6.64, 2.5))
    figure.subplots_adjust(left=.092, right=.981, top=.775, bottom=.224, wspace=.69)
    axes[1].sharey(axes[0])
    effect_axis = axes[-1]
    effect_rows = (2., 1., 0.)
    for column, (cohort, title, offset, marker) in enumerate(cohorts):
        report = json.loads((root / f'seed_ensemble_{cohort}.json').read_text(encoding='utf-8'))
        sizes = report['ensemble_sizes']
        compact = report['methods']['original_max_shared']['curve']
        axis = axes[column]
        for row, (name, _, color, model_marker) in zip(effect_rows, references, strict=True):
            reference = report['methods'][name]['curve']
            gap = [(compact[str(k)]['rps'] - reference[str(k)]['rps']) * 1000 for k in sizes]
            axis.plot(sizes, gap, color=color, linewidth=1, marker=model_marker, markersize=3.7)
            estimate = report['comparisons'][name]['gap_reduction']
            interval(effect_axis, estimate['estimate'] * 1000, np.asarray(estimate['ci95_familywise']) * 1000,
                     row + offset, color, marker)
        axis.axhline(0, color='black', linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        axis.set_xticks((1, 3, 5, 10))
        axis.set_xlim(.55, 10.45)
        axis.set_ylim(-5.3, 23.)
        axis.set_xlabel('Fits averaged')
        axis.set_title(f'{"ab"[column]}  {title}', loc='left', fontweight='bold', fontsize=8.5, pad=6)
    axes[0].set_ylabel('RPS gap × 1,000')
    axes[1].tick_params(labelleft=False)
    effect_axis.set_title('c  Gap reduction', loc='left', fontweight='bold', fontsize=8.5, pad=6)
    effect_axis.set_xlabel('RPS reduction × 1,000')
    effect_axis.set_yticks(effect_rows, [label for _, label, _, _ in references])
    effect_axis.set_ylim(-.5, 2.5)
    effect_axis.spines['left'].set_visible(False)
    effect_axis.tick_params(axis='y', length=0, pad=3, labelsize=7.5)
    effect_axis.axvline(0, color='black', linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
    effect_axis.xaxis.set_major_locator(MaxNLocator(4))
    effect_axis.margins(x=.1)
    handles = [Line2D([], [], color=color, marker=marker, markersize=3.7, linewidth=1, label=label)
               for _, label, color, marker in references]
    names = [Line2D([], [], color='black', marker=marker, markersize=3.9, linestyle='none', label=title)
             for _, title, _, marker in cohorts]
    figure.legend(handles=handles + names, loc='upper center', bbox_to_anchor=(.507, .948),
                  ncols=5, frameon=False, borderaxespad=0., columnspacing=1.4, handlelength=1.5)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_gap_location(root, output, alignment_script, layout_report):
    configure_figure()
    cohorts = (('sims_v2', 'CH-SIMS v2', '#2766A6', 's', .15),
               ('omg', 'OMG-Emotion', '#16736D', 'o', -.15))
    protocols = (('resampling_fixed_start', 'Original location'),
                 ('resampling', 'All locations'),
                 ('worst_start', 'Worst location'))
    stages = (('anchor', 'Anchor'), ('fusion', 'Fusion'), ('interaction', 'Interaction'))
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.24), sharex=True)
    figure.subplots_adjust(left=.182, right=.983, bottom=.245, top=.758, wspace=.48)
    effects, components = axes
    rows = (2., 1., 0.)
    limits = [0.]
    for cohort, _, color, marker, offset in cohorts:
        report = json.loads((root / f'gap_location_{cohort}.json').read_text(encoding='utf-8'))
        for axis, estimates, keys in ((effects, report, protocols), (components, report['components'], stages)):
            for y, (key, _) in zip(rows, keys, strict=True):
                value = estimates[key]
                bounds = np.asarray(value['ci95_familywise']) * 1000
                limits.extend(bounds)
                interval(axis, value['reduction'] * 1000, bounds, y + offset, color, marker)
    margin = np.ptp(limits) * .085
    for axis, choices, title in zip(axes, (protocols, stages), ('a  Evaluation locations', 'b  Training stages'),
                                    strict=True):
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.set_yticks(rows, [label for _, label in choices])
        axis.tick_params(axis='y', length=0, pad=4)
        axis.set_xlim(min(limits) - margin, max(limits) + margin)
        axis.set_ylim(-.55, 2.55)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_xlabel('RPS reduction × 1,000')
        axis.set_title(title, loc='left', fontweight='bold', fontsize=8.5, pad=7)
    handles = [Line2D([], [], color=color, marker=marker, markersize=4.2, linestyle='none', label=label)
               for _, label, color, marker, _ in cohorts]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .945),
                  ncols=2, frameon=False, borderaxespad=0., columnspacing=2.)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_face_crops(root, output, alignment_script, layout_report):
    report = json.loads((root / 'face_crop_omg.json').read_text(encoding='utf-8'))
    contrasts = (
        ('a  Compact model representations',
         (('Face blend vs whole frame', 'primary', 'whole'),
          ('Face blend vs center blend', 'primary', 'center_scene'),
          ('Face crop vs whole frame', 'localized_only', 'whole'),
          ('Face crop vs center crop', 'localized_only', 'center'))),
        ('b  Published fusion references',
         (('Face blend for LNLN', 'benchmark_representation', 'lnln'),
          ('Face blend for RM-Net', 'benchmark_representation', 'rmnet'),
          ('Compact vs LNLN', 'benchmark_matched', 'lnln'),
          ('Compact vs RM-Net', 'benchmark_matched', 'rmnet'))),
    )
    plot_input_contrasts(report, contrasts, output, alignment_script, layout_report)


def plot_audio_channel(root, output, alignment_script, layout_report):
    report = json.loads((root / 'audio_channel_omg.json').read_text(encoding='utf-8'))
    contrasts = (
        ('a  Compact model comparisons',
         (('Adding audio', 'primary', 'adding_audio'),
          ('Shared vs independent', 'primary', 'sharing'),
          ('Shared vs class only', 'primary', 'class_only'),
          ('Change in sharing gain', 'primary', 'sharing_change'))),
        ('b  Published fusion references',
         (('Audio added to LNLN', 'benchmark_audio', 'lnln'),
          ('Audio added to RM-Net', 'benchmark_audio', 'rmnet'),
          ('Compact vs LNLN', 'benchmark_matched', 'lnln'),
          ('Compact vs RM-Net', 'benchmark_matched', 'rmnet'))),
    )
    plot_input_contrasts(report, contrasts, output, alignment_script, layout_report)


def plot_marginal_tilt(root, output, alignment_script, layout_report):
    configure_figure()
    report = json.loads((root / 'marginal_tilt.json').read_text(encoding='utf-8'))['cohorts']
    cohorts = (('sims_v2', 'CH-SIMS v2', '#27639B', '-'),
               ('omg', 'OMG-Emotion', '#864366', '--'),
               ('ravdess', 'RAVDESS', '#176B74', '-.'))
    contrasts = (('sharing', 'Against independent', '#176B74', 'o', .14),
                 ('marginal', 'Against linear', '#864366', 'D', -.14))
    figure, (response, effects) = plt.subplots(1, 2, figsize=(6.5, 2.60))
    figure.subplots_adjust(left=.112, right=.985, bottom=.31, top=.87, wspace=.60)
    y_levels = (2., 1., 0.)
    limits = [0.]
    for y, (key, label, color, style) in zip(y_levels, cohorts, strict=True):
        law = report[key]['response']
        delta = np.asarray(law['delta'])
        correction = (np.asarray(law['exact']) - delta).mean(axis=0)
        response.plot(delta[0], correction, color=color, linestyle=style, linewidth=1.35, label=label)
        for name, _, mark_color, marker, offset in contrasts:
            value = report[key]['primary'][name]
            bounds = np.asarray(value['ci95_familywise']) * 1000
            limits.extend(bounds)
            interval(effects, value['reduction'] * 1000, bounds, y + offset, mark_color, marker)
    response.axhline(0, color='black', linewidth=.65, linestyle=(0, (2, 3)), zorder=0)
    response.set_xlabel(r'Signed correction $\delta$')
    response.set_ylabel(r'$D(\kappa\delta)-\delta$')
    response.set_xticks((-2, -1, 0, 1, 2))
    response.yaxis.set_major_locator(MaxNLocator(4))
    response.set_title('a  Exact marginal response', loc='left', fontweight='bold', fontsize=8.5, pad=7)
    response.legend(loc='upper center', bbox_to_anchor=(.5, -.32), frameon=False,
                    ncol=2, columnspacing=.8, handlelength=1.7, borderaxespad=0.)

    effects.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    effects.spines['left'].set_visible(False)
    effects.set_yticks(y_levels, [label for _, label, _, _ in cohorts])
    effects.tick_params(axis='y', length=0, pad=4)
    effects.set_ylim(-.45, 2.45)
    margin = max(np.ptp(limits) * .06, .5)
    effects.set_xlim(min(limits) - margin, max(limits) + margin)
    effects.xaxis.set_major_locator(MaxNLocator(5))
    effects.set_xlabel('RPS reduction × 1,000')
    effects.set_title('b  Exact shared head', loc='left', fontweight='bold', fontsize=8.5, pad=7)
    handles = [Line2D([], [], marker=marker, linestyle='none', color=color, markersize=4, label=label)
               for _, label, color, marker, _ in contrasts]
    effects.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.32),
                   frameon=False, borderaxespad=0., handletextpad=.45)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_input_contrasts(report, contrasts, output, alignment_script, layout_report):
    configure_figure()
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 2.22))
    figure.subplots_adjust(left=.248, right=.978, bottom=.168, top=.845, wspace=.74)
    for axis, (title, items) in zip(axes, contrasts, strict=True):
        gain_tiles(axis, [report[family][method] for _, family, method in items])
        axis.set_yticks(range(len(items)), [label for label, _, _ in items])
        axis.set_xlabel('RPS reduction × 1,000', labelpad=4)
        axis.set_title(title, loc='left', fontweight='bold', fontsize=8.5, pad=7)
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_msp_face_transfer(root, output):
    configure_figure()
    report = json.loads((root / 'msp_face_transfer.json').read_text(encoding='utf-8'))
    rows = [('Independent intensity', report['primary'], '#176B74', 'D'),
            ('Class-only fusion', report['secondary']['fusion'], '#28405E', 'o'),
            ('Text anchor', report['secondary']['anchor'], '#28405E', 'o'),
            ('LNLN', report['secondary']['lnln'], '#744771', 's'),
            ('RM-Net', report['secondary']['rmnet'], '#744771', 's')]
    figure, axis = plt.subplots(figsize=(6.5, 2.04))
    figure.subplots_adjust(left=.236, right=.98, bottom=.24, top=.965)
    positions = np.arange(len(rows))[::-1]
    bounds = [0.]
    for y, (_, effect, color, marker) in zip(positions, rows, strict=True):
        limits = np.asarray(effect['ci95_familywise']) * 1000
        bounds.extend(limits)
        interval(axis, effect['reduction'] * 1000, limits, y, color, marker)
    margin = max(np.ptp(bounds) * .06, .5)
    axis.set_xlim(min(bounds) - margin, max(bounds) + margin)
    axis.set_ylim(-.48, len(rows) - .52)
    axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    axis.spines['left'].set_visible(False)
    axis.set_yticks(positions, [name for name, *_ in rows])
    axis.tick_params(axis='y', length=0, pad=6)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.set_xlabel('RPS reduction × 1,000')
    export(figure, output)


def plot_msp_face_refit(root, output, alignment_script, layout_report):
    configure_figure()
    report = json.loads((root / 'msp_face_refit_paired.json').read_text(encoding='utf-8'))
    secondary, adaptation = report['secondary'], report['adaptation']
    refit_rows = (
        ('Independent intensity', report['primary'], '#176B74'),
        ('Text anchor', secondary['anchor'], '#28405E'),
        ('Class-only fusion', secondary['fusion'], '#28405E'),
        ('Shuffled intensity', secondary['intensity_shuffled'], '#28405E'),
        ('Visual projection', secondary['intensity_projection'], '#28405E'),
        ('LNLN', secondary['lnln'], '#744771'),
        ('RM-Net', secondary['rmnet'], '#744771'),
    )
    if isinstance(output, PlatePart):
        refit_rows = refit_rows[2:]
    panels = (
        ('a', refit_rows),
        ('b', (
            ('Over independent', adaptation['sharing_gain_change'], '#176B74'),
            ('Over text anchor', adaptation['visual_gain_change'], '#28405E'),
        )),
    )
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(6.5, 3.46),
                               gridspec_kw={'height_ratios': (len(refit_rows), 2)})
    figure.subplots_adjust(left=.24, right=.98, bottom=.15, top=.91, hspace=.34)
    bounds = [0.]
    for axis, (title, rows) in zip(axes, panels, strict=True):
        positions = np.arange(len(rows))[::-1]
        for y, (_, effect, color) in zip(positions, rows, strict=True):
            limits = np.asarray(effect['ci95_familywise']) * 1000
            bounds.extend(limits)
            signed_bar(axis, effect['reduction'] * 1000, limits, y, color)
        axis.set_ylim(-.5, len(rows) - .5)
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.spines['left'].set_visible(False)
        axis.set_yticks(positions, [name for name, *_ in rows])
        axis.tick_params(axis='y', length=0, pad=6)
        axis.set_title(title, loc='left', fontweight='bold', fontsize=8.5, pad=6)
    margin = max(np.ptp(bounds) * .06, .5)
    axes[0].set_xlim(min(bounds) - margin, max(bounds) + margin)
    axes[0].tick_params(axis='x', length=0)
    axes[1].xaxis.set_major_locator(MaxNLocator(6))
    axes[1].set_xlabel('Paired RPS difference × 1,000')
    if alignment_script:
        sys.path.insert(0, str(alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(figure, json_out=layout_report)
    export(figure, output)


def plot_neutral_readout(root, output):
    configure_figure()
    report = json.loads((root / 'neutral_readout_effects.json').read_text(encoding='utf-8'))
    cohorts = [
        ('sims_v2_individual', 'CH-SIMS v2, individual'),
        ('sims_v2_class_mean', 'CH-SIMS v2, class mean'),
        ('omg_individual', 'OMG-Emotion, individual'),
        ('omg_class_mean', 'OMG-Emotion, class mean'),
        ('moud', 'MOUD, frozen'),
        ('meld', 'MELD, frozen'),
        ('ravdess', 'RAVDESS, balanced'),
    ]
    contrasts = [('factorization', 'Factorized', '#1B5E9C', 'D'),
                 ('gradient_only', 'Gradient only', '#96502A', 'o')]
    errors = [('signed_error', 'Signed', '#245C63'),
              ('neutral_probability_error', 'Neutral', '#8A486D')]
    rows = np.array([7.4, 6.4, 4.95, 3.95, 2.5, 1.5, .05])
    figure, (effect, component) = plt.subplots(1, 2, figsize=(6.6, 3.22), sharey=True,
                                               gridspec_kw={'width_ratios': (1.23, 1)})
    figure.subplots_adjust(left=.232, right=.98, bottom=.145, top=.763, wspace=.24)
    for (case, _), row in zip(cohorts, rows):
        summary = report['cases'][case]
        for offset, (key, _, color, marker) in zip((.18, -.18), contrasts):
            result = summary['contrasts'][key]
            low, high = 1000 * np.asarray(result['ci95_familywise'])
            effect.hlines(row + offset, low, high, color=color, linewidth=1.1)
            effect.plot(1000 * result['reduction'], row + offset, color=color, marker=marker,
                        linestyle='none', markerfacecolor=color if key == 'factorization' else 'white',
                        markersize=4, zorder=3)
        softmax, factorized = (summary['methods'][name] for name in ('softmax_shared', 'factorized_shared'))
        for offset, (key, _, color) in zip((.18, -.18), errors):
            component.barh(row + offset, 1000 * (softmax[key] - factorized[key]), height=.25, color=color)
    effect.set_yticks(rows, [label for _, label in cohorts])
    component.tick_params(labelleft=False)
    effect.legend(handles=[Line2D([], [], color=color, marker=marker, markersize=4,
                                  markerfacecolor=color if key == 'factorization' else 'white',
                                  label=label) for key, label, color, marker in contrasts],
                  loc='lower left', bbox_to_anchor=(0, 1.05), ncol=2, frameon=False,
                  borderaxespad=0, columnspacing=.9, handletextpad=.4, handlelength=1.4)
    component.legend(handles=[Line2D([], [], color=color, linewidth=3, label=label)
                              for _, label, color in errors],
                     loc='lower left', bbox_to_anchor=(0, 1.05), ncol=2, frameon=False,
                     borderaxespad=0, columnspacing=.9, handletextpad=.4, handlelength=1.1)
    for axis, title in (
        (effect, 'a  Readout comparisons'),
        (component, 'b  Factorization components'),
    ):
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.set_ylim(-.52, 7.95)
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0, pad=5)
        axis.margins(x=.11)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.set_title(title, loc='left', y=1.17, fontweight='bold')
        axis.set_xlabel('RPS reduction × 1,000')
    export(figure, output)


def plot_visual_components(root, output):
    configure_figure()
    report = json.loads((root / 'visual_component_effects.json').read_text(encoding='utf-8'))['cases']
    cohorts = [('sims_v2', 'CH-SIMS v2'), ('omg', 'OMG-Emotion'), ('ravdess', 'RAVDESS, balanced'),
               ('moud', 'MOUD, frozen'), ('meld', 'MELD, frozen')]
    pooling_rows = [('sims_v2', 'deep', 'CH-SIMS v2, deep'),
                    ('sims_v2', 'projection', 'Projection'),
                    ('omg', 'deep', 'OMG-Emotion, deep'),
                    ('omg', 'projection', 'Projection')]
    left_styles = [('depth', 'Remove blocks', '#1B5E9C', 'D'),
                   ('half_gain', 'Midpoint contrast', '#8A486D', 'o')]
    right_styles = [('add_mean', 'Mean', '#245C63', 'D'),
                    ('add_maximum', 'Maximum', '#96502A', 'o')]
    figure, (depth, pooling) = plt.subplots(1, 2, figsize=(6.7, 2.8))
    figure.subplots_adjust(left=.185, right=.979, bottom=.166, top=.74, wspace=.74)
    left_rows = [4.25, 3.25, 2.25, .85, -.15]
    right_rows = [4.05, 3.05, 1.12, .12]
    for (name, _), row in zip(cohorts, left_rows):
        contrasts = report[name]['contrasts']
        for offset, (key, _, color, marker) in zip((.17, -.17), left_styles):
            if key in contrasts:
                estimate = contrasts[key]
                interval(depth, 1000 * estimate['reduction'], 1000 * np.asarray(estimate['ci95_familywise']),
                         row + offset if 'half_gain' in contrasts else row, color, marker)
    for (name, stage, _), row in zip(pooling_rows, right_rows):
        for offset, (key, _, color, marker) in zip((.17, -.17), right_styles):
            estimate = report[name]['contrasts'][f'pooling__{stage}_{key}']
            interval(pooling, 1000 * estimate['reduction'], 1000 * np.asarray(estimate['ci95_familywise']),
                     row + offset, color, marker)
    depth.set_yticks(left_rows, [label for _, label in cohorts])
    pooling.set_yticks(right_rows, [label for _, _, label in pooling_rows])
    for axis, title, styles in (
        (depth, 'a  Visual residual blocks', left_styles),
        (pooling, 'b  Pooling statistics', right_styles),
    ):
        axis.axvline(0, color='black', linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
        axis.set_ylim(-.65, 4.73)
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0, pad=5)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.margins(x=.09)
        axis.set_title(title, loc='left', y=1.17, fontweight='bold')
        axis.set_xlabel('RPS difference × 1,000')
        handles = [Line2D([], [], label=label, color=color, marker=marker, markersize=3.6)
                   for _, label, color, marker in styles]
        axis.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.055), ncol=2,
                    frameon=False, borderaxespad=0, columnspacing=.8, handletextpad=.35, handlelength=1.2)
    export(figure, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alignment-script', type=Path)
    parser.add_argument('--layout-report', type=Path)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--readouts', action='store_true')
    variant.add_argument('--seed-sensitivity', action='store_true')
    variant.add_argument('--architecture', action='store_true')
    variant.add_argument('--modality-ratings', action='store_true')
    variant.add_argument('--transfer', action='store_true')
    variant.add_argument('--score-gain', action='store_true')
    variant.add_argument('--fixed-transcripts', action='store_true')
    variant.add_argument('--moud-transfer', action='store_true')
    variant.add_argument('--ordinal-controls', action='store_true')
    variant.add_argument('--frozen-dialogue', action='store_true')
    variant.add_argument('--probability-pool', action='store_true')
    variant.add_argument('--published-seeds', action='store_true')
    variant.add_argument('--published-cohorts', action='store_true')
    variant.add_argument('--frozen-lnln-heads', action='store_true')
    variant.add_argument('--conditional-intensity', action='store_true')
    variant.add_argument('--reference-distillation', action='store_true')
    variant.add_argument('--cremad', action='store_true')
    variant.add_argument('--view-stability', action='store_true')
    variant.add_argument('--class-weighting', action='store_true')
    variant.add_argument('--benchmark-components', action='store_true')
    variant.add_argument('--dispersion-pooling', action='store_true')
    variant.add_argument('--seed-ensemble', action='store_true')
    variant.add_argument('--gap-location', action='store_true')
    variant.add_argument('--face-crops', action='store_true')
    variant.add_argument('--audio-channel', action='store_true')
    variant.add_argument('--marginal-tilt', action='store_true')
    variant.add_argument('--msp-face-transfer', action='store_true')
    variant.add_argument('--msp-face-refit', action='store_true')
    variant.add_argument('--neutral-readout', action='store_true')
    variant.add_argument('--visual-components', action='store_true')
    args = parser.parse_args()
    if args.readouts:
        plot_readouts(args.results, args.output)
    elif args.architecture:
        plot_architecture(args.results, args.output)
    elif args.seed_sensitivity:
        plot_seed_sensitivity(args.results, args.output)
    elif args.modality_ratings:
        plot_modality_ratings(args.results, args.output)
    elif args.transfer:
        plot_transfer(args.results, args.output)
    elif args.score_gain:
        plot_score_gain(args.results, args.output)
    elif args.fixed_transcripts:
        plot_fixed_transcripts(args.results, args.output)
    elif args.moud_transfer:
        plot_moud_transfer(args.results, args.output)
    elif args.ordinal_controls:
        plot_ordinal_controls(args.results, args.output)
    elif args.frozen_dialogue:
        plot_frozen_dialogue(args.results, args.output)
    elif args.probability_pool:
        plot_probability_pool(args.results, args.output)
    elif args.published_seeds:
        plot_published_seeds(args.results, args.output)
    elif args.published_cohorts:
        plot_published_cohorts(args.results, args.output)
    elif args.frozen_lnln_heads:
        plot_frozen_lnln_heads(args.results, args.output)
    elif args.conditional_intensity:
        plot_conditional_intensity(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.reference_distillation:
        plot_reference_distillation(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.cremad:
        plot_cremad(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.view_stability:
        plot_view_stability(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.class_weighting:
        plot_class_weighting(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.benchmark_components:
        plot_benchmark_components(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.dispersion_pooling:
        plot_dispersion_pooling(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.seed_ensemble:
        plot_seed_ensemble(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.gap_location:
        plot_gap_location(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.face_crops:
        plot_face_crops(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.audio_channel:
        plot_audio_channel(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.marginal_tilt:
        plot_marginal_tilt(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.msp_face_transfer:
        plot_msp_face_transfer(args.results, args.output)
    elif args.msp_face_refit:
        plot_msp_face_refit(args.results, args.output, args.alignment_script, args.layout_report)
    elif args.neutral_readout:
        plot_neutral_readout(args.results, args.output)
    elif args.visual_components:
        plot_visual_components(args.results, args.output)
    else:
        plot(args.results, args.output, args.alignment_script, args.layout_report)
