"""Draw the manuscript figures from the released prediction packets."""

import argparse
import json
from pathlib import Path
import sys

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as ArtistPath
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
from tempfile import TemporaryDirectory

import plot_ablation as supporting
from figure_plates import SUPPLEMENT_PLATES, assemble_plate, plate_parts
from plot_ablation import configure_figure, export
from plot_model import arrow, box


TEAL = '#007F83'
MAGENTA = '#AD3567'
INDIGO = '#303D78'
INK = '#222833'
SLATE = '#647082'
GRID = '#E4E9EE'
PALE_BLUE = '#EDF1FA'
PALE_TEAL = '#E8F3F0'
PALE_PINK = '#F7ECF0'
PALE_AMBER = '#FCF2E5'
POLARITY = ('Negative', 'Neutral', 'Positive')
CLASS_COLORS = (MAGENTA, INDIGO, TEAL)
VIEWS = ('Full', '35% gap', '65% gap', 'No vision')
CHANNELS = LinearSegmentedColormap.from_list('clip_channels', (INDIGO, 'white', TEAL))
CHANGE = LinearSegmentedColormap.from_list('paired_change', (MAGENTA, '#F9F9F9', TEAL))


def report(runs, name):
    return json.loads((runs / f'{name}.json').read_text(encoding='utf-8'))


def panel(axis, letter, x=0, y=1.045):
    axis.text(x, y, letter, transform=axis.transAxes, ha='left', va='bottom',
              fontweight='bold', fontsize=10, clip_on=False, color='black')


def left_of_y_labels(axis):
    figure = axis.figure
    bounds = axis.yaxis.get_tightbbox(figure.canvas.get_renderer())
    return bounds.transformed(figure.transFigure.inverted()).x0


def forest_axis(axis, labels, xlabel, limits=None):
    axis.set_axisbelow(True)
    axis.grid(axis='y', color=GRID, linewidth=.45)
    axis.axvline(0, color=SLATE, linewidth=.72, linestyle=(0, (3, 3)), zorder=1)
    axis.spines['left'].set_visible(False)
    axis.set_yticks(np.arange(len(labels))[::-1], labels)
    axis.tick_params(axis='y', length=0, pad=5)
    axis.set_ylim(-.62, len(labels) - .38)
    axis.set_xlabel(xlabel, labelpad=3)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    if limits:
        axis.set_xlim(*limits)
    else:
        axis.margins(x=.11)


def architecture_flow(axis, visual, visible, audio, audible):
    axis.set(xlim=(0, 1.02), ylim=(0, 1))
    axis.set_axis_off()
    panel(axis, 'a', y=1.03)
    box(axis, .103, .824, .174, .235, 'Text + masks', PALE_BLUE)
    for row, label, fill, values, observed in (
        (.500, 'Observed audio bins', PALE_AMBER, audio, audible),
        (.176, 'Observed visual bins', PALE_TEAL, visual, visible),
    ):
        box(axis, .103, row, .174, .235, '', fill)
        axis.text(.103, row + .052, label, ha='center', va='center', fontsize=7.4)
        features = normalized_channels(values)
        sample = np.ma.array(features, mask=np.broadcast_to(~observed, features.shape))
        bottom = row - .092
        axis.imshow(sample, extent=(.034, .172, bottom, bottom + .083),
                    cmap=CHANNELS.with_extremes(bad='#F9FAFB'),
                    vmin=-2, vmax=2, aspect='auto', interpolation='none', zorder=3)
        axis.add_patch(Rectangle((.034, bottom), .138, .083, facecolor='none',
                                 edgecolor='#8D9CA8', linewidth=.45, zorder=4))
    box(axis, .337, .824, .211, .235, 'Text anchor\n' + r'$\ell,\ s_0$', PALE_BLUE)
    box(axis, .337, .500, .211, .235, 'Audio encoder\nMean + max', PALE_AMBER)
    box(axis, .337, .176, .211, .235, 'Visual encoder\nMean + max', PALE_TEAL)
    box(axis, .597, .500, .162, .384, '', PALE_PINK)
    axis.text(.597, .620, 'Fusion', ha='center', va='center', fontsize=8.2)
    axis.text(.594, .505, r'Neutral $\nu$', color=INDIGO, ha='center', va='center', fontsize=7.6)
    axis.text(.594, .386, r'Shared $\delta$', color=MAGENTA, ha='center', va='center',
              fontsize=7.6, weight='bold')
    axis.add_patch(Rectangle((.763, .057), .253, .862, facecolor='white', edgecolor='black', linewidth=.75))
    axis.plot([.763, 1.016], [.489, .489], color='black', linewidth=.6)
    axis.text(.889, .75, 'Polarity', ha='center', va='center', fontsize=8.35, weight='bold')
    axis.text(.889, .608, r'$\mathrm{softmax}(\ell+[-\delta,\nu,\delta])$',
              ha='center', va='center', fontsize=8.0)
    axis.text(.889, .352, 'Intensity', ha='center', va='center', fontsize=8.35, weight='bold')
    axis.text(.889, .192, r'$c\,\tanh(\operatorname{arctanh}(s_0)+\alpha\delta)$',
              ha='center', va='center', fontsize=7.85)
    for row in (.824, .500, .176):
        arrow(axis, [(.193, row), (.229, row)])
    arrow(axis, [(.444, .790), (.482, .790), (.482, .605), (.513, .605)])
    arrow(axis, [(.444, .500), (.513, .500)])
    arrow(axis, [(.444, .176), (.482, .176), (.482, .395), (.513, .395)])
    arrow(axis, [(.448, .875), (.764, .875)])
    axis.text(.600, .905, r'$\ell,\ s_0$', va='bottom', ha='center', fontsize=7.4, color=SLATE)
    arrow(axis, [(.679, .505), (.701, .505), (.701, .706), (.763, .706)], color=INDIGO)
    axis.plot([.679, .725], [.386, .386], color=MAGENTA, linewidth=.95)
    arrow(axis, [(.725, .386), (.725, .575), (.763, .575)], color=MAGENTA)
    arrow(axis, [(.725, .386), (.725, .305), (.763, .305)], color=MAGENTA)
    axis.scatter([.725], [.386], s=7, color=MAGENTA, zorder=4)


def filmstrip(axis, frames, slots):
    axis.set(xlim=(0, 24), ylim=(0, 1))
    axis.set_axis_off()
    width_inches, height_inches = axis.figure.get_size_inches()
    box_size = axis.get_position()
    points_per_bin = width_inches * box_size.width / 24
    image_height, image_width = frames.shape[1:3]
    height_per_bin = points_per_bin * image_height / image_width / (height_inches * box_size.height)
    edge_limit = 2 * min(slots[0] + .5, 23.5 - slots[-1]) - .08
    width = min(round(24 / len(slots) * .907, 2), .84 / height_per_bin, edge_limit)
    height = width * height_per_bin
    lower = .08
    for bin_index, frame in zip(slots, frames, strict=True):
        center = bin_index + .5
        left, right = center - width / 2, center + width / 2
        axis.imshow(frame, extent=(left, right, lower, lower + height), interpolation='antialiased',
                    aspect='auto', zorder=1)
        axis.add_patch(Rectangle((left, lower), width, height, facecolor='none', edgecolor=INK,
                                 linewidth=.5, zorder=2))
        axis.plot([center, center], [0, lower], color='black', linewidth=.5, zorder=3)
    return lower + height


def normalized_channels(values):
    return (values - values.mean(1, keepdims=True)) / values.std(1, keepdims=True).clip(1e-6)


def feature_strip(axis, features, visible, center, height=.59):
    bottom = center - height / 2
    observed = np.ma.array(features, mask=np.broadcast_to(~visible, features.shape))
    image = axis.imshow(observed, extent=(0, len(visible), bottom, bottom + height), cmap=CHANNELS,
                        vmin=-2, vmax=2, aspect='auto', interpolation='none', zorder=1)
    edges = np.diff(np.r_[False, ~visible, False].astype(int))
    for start, end in zip(np.flatnonzero(edges > 0), np.flatnonzero(edges < 0), strict=True):
        axis.add_patch(Rectangle((start, bottom), end - start, height, facecolor='#F1F3F5',
                                 edgecolor='#9DA7B3', linewidth=.48, zorder=2))
        lines = []
        for origin in np.arange(start - 1.8, end, 1.40):
            left, right = max(start, origin), min(end, origin + 2.)
            lines.append([(left, bottom + height * (left - origin) / 2),
                          (right, bottom + height * (right - origin) / 2)])
        axis.add_collection(LineCollection(lines, colors='#A8B3BE', linewidths=.48, zorder=3))
    return image


def probability_stack(axis, row, prediction, alpha):
    axis.barh(row, prediction, left=np.r_[0., np.cumsum(prediction[:-1])], height=.262,
              color=CLASS_COLORS, edgecolor='none', alpha=alpha, zorder=2)


def model_trace(runs):
    with np.load(runs / 'paper_example.npz', allow_pickle=False) as packet:
        data = dict(packet)
    with np.load(runs / 'audiovisual_example.npz', allow_pickle=False) as packet:
        audio, visual = packet['audio'], packet['vision']
        audible, visible = packet['masks'][1]
    figure = plt.figure(figsize=(6.75, 5.70))
    grid = figure.add_gridspec(3, 19, left=.126, right=.972, top=.916, bottom=.181,
                              height_ratios=(1.13, .74, 1.63), hspace=.42, wspace=0)
    architecture = figure.add_subplot(grid[0, :])
    frames = figure.add_subplot(grid[1, :])
    observations, shifts, probabilities, intensity = (figure.add_subplot(grid[2, span]) for span in
                                                       (slice(0, 5), slice(6, 9), slice(10, 14), slice(15, 19)))
    rows = np.arange(len(VIEWS))[::-1]
    channels = normalized_channels(data['visual'])
    for i, row in enumerate(rows):
        image = feature_strip(observations, channels, data['masks'][i], row)
        shifts.barh(row + .145, data['signed'][i], height=.255, color=MAGENTA, zorder=2)
        shifts.barh(row - .145, data['neutral'][i], height=.255, color=INDIGO, zorder=2)
        probability_stack(probabilities, row + .143, data['anchor'][i], .39)
        probability_stack(probabilities, row - .143, data['fused'][i], 1.0)
        before, after = data['anchor_score'][i], data['fused_score'][i]
        intensity.hlines(row, before, after, color=TEAL, linewidth=1.05, zorder=2)
        intensity.vlines(before, row - .18, row + .18, color=SLATE, linewidth=1.2, zorder=3)
        intensity.vlines(after, row - .113, row + .113, color=TEAL, linewidth=2.0, zorder=4)
        if i == 1:
            for axis in (observations, shifts, probabilities, intensity):
                axis.axhspan(row - .465, row + .465, facecolor=PALE_PINK, zorder=0)
            for offset, update in ((.145, data['signed'][i]), (-.145, data['neutral'][i])):
                shifts.text(update / 2, row + offset, f'{update:.2f}'.replace('-', '−'),
                            ha='center', va='center', color='white', fontsize=6.3)
            rated = int(data['polarity'])
            for y, forecast, color in ((row + .143, data['anchor'][i], INK),
                                       (row - .143, data['fused'][i], 'white')):
                value = forecast[rated]
                probabilities.text(forecast[:rated].sum() + value / 2, y, f'{value:.1%}',
                                   color=color, ha='center', va='center', fontsize=6.35)
            intensity.text(before, row + .22, f'{before:.3f}'.replace('-', '−'),
                           color=SLATE, va='bottom', ha='right', fontsize=6.6)
            intensity.text(after, row - .18, f'{after:.3f}'.replace('-', '−'),
                           color=TEAL, va='top', ha='left', fontsize=6.6)
    shifts.axvline(0, color=SLATE, linewidth=.65, zorder=1)
    intensity.axvline(float(data['target']), color=SLATE, linewidth=.75, linestyle=(0, (3, 3)), zorder=1)
    observations.set_xticks([.5, 11.5, 23.5], ['1', '12', '24'])
    observations.set_xlim(0, 24)
    shifts.set_xlim(min(data['signed'].min(), data['neutral'].min()) - .08, .12)
    shifts.set_xticks([-1, -.5, 0])
    probabilities.set_xlim(0, 1)
    probabilities.set_xticks([0, .5, 1])
    intensity.set_xlim(min(data['fused_score'].min(), data['anchor_score'].min()) - .05, .02)
    intensity.set_xticks([-.6, -.3, 0])
    traces = (observations, shifts, probabilities, intensity)
    for axis, label in zip(traces, ('CLIP time bin', 'Logit update', 'Class probability', 'Signed intensity'),
                           strict=True):
        axis.set_ylim(-.50, len(rows) - .50)
        axis.set_yticks(rows, VIEWS if axis is observations else [''] * len(rows))
        axis.tick_params(axis='y', length=0, pad=4)
        axis.tick_params(axis='x', length=2.6, pad=2.5)
        axis.spines['left'].set_visible(False)
        axis.set_xlabel(label, labelpad=3.5)
        for boundary in [.5, 1.5, 2.5]:
            axis.axhline(boundary, color=GRID, linewidth=.55, zorder=0)

    key = observations.inset_axes([.09, -.34, .80, .034])
    scale = figure.colorbar(image, cax=key, orientation='horizontal', ticks=[-2, 0, 2])
    scale.outline.set_visible(False)
    scale.ax.tick_params(labelsize=7, length=2, pad=1)
    scale.set_label('Feature z score', labelpad=1, size=7.3)
    legend = [Line2D([], [], color=color, lw=3.0, label=label)
              for color, label in zip(CLASS_COLORS, POLARITY, strict=True)]
    probabilities.legend(handles=legend, loc='upper center', bbox_to_anchor=(1.10, -.34), frameon=False,
                         ncol=3, columnspacing=.70, handletextpad=.33, handlelength=1.05,
                         borderpad=0, fontsize=7.1)
    shifts.legend(handles=[Patch(facecolor=MAGENTA, label=r'$\delta$'), Patch(facecolor=INDIGO, label=r'$\nu$')],
                  loc='upper center', bbox_to_anchor=(.48, -.34), ncol=2, frameon=False,
                  handletextpad=.36, handlelength=1.0, columnspacing=.8, fontsize=7.5, borderpad=0)

    figure.canvas.draw()
    label_edge = left_of_y_labels(observations)
    for axis, lift in ((frames, .025), (architecture, 0)):
        position = axis.get_position()
        axis.set_position(Bbox.from_extents(label_edge, position.y0 + lift, position.x1, position.y1 + lift))
    for axis, letter in zip(traces, 'cdef', strict=True):
        position = axis.get_position()
        label_x = (label_edge - position.x0) / position.width if axis is observations else 0
        panel(axis, letter, x=label_x)

    top = filmstrip(frames, data['frames'], data['frame_slots'])
    frames.text(.5, top + .12, f'“{data["transcript"].item()}”', transform=frames.transAxes,
                ha='center', va='bottom', fontsize=8.6, color=INK)
    for number in data['frame_slots'] + 1:
        frames.text(number - .5, -.032, str(number), ha='center', va='top', fontsize=7.1,
                    color=SLATE, clip_on=False)
    hidden = np.flatnonzero(~data['masks'][1])
    frames.add_patch(Rectangle((hidden[0], -.247), len(hidden), .093, facecolor='#F0D5DF',
                               edgecolor=MAGENTA, linewidth=.6, clip_on=False))
    panel(frames, 'b', y=top + .105)
    architecture_flow(architecture, visual, visible, audio, audible)
    return figure


def statement_predictions(axis, runs):
    with np.load(runs / 'statement_examples.npz', allow_pickle=False) as packet:
        frames = packet['frames'][:, 1]
        labels = packet['labels']
        anchor = packet['anchor'].mean(0)
        fused = packet['fused'].mean(0)
        statement = packet['statement'].item()
    axis.set(xlim=(0, 3), ylim=(0, 1))
    axis.set_axis_off()
    bounds = axis.get_position()
    panel_height = bounds.height * axis.figure.get_figheight()
    column_width = bounds.width * axis.figure.get_figwidth() / 3
    frame_width = min(.88, .78 * panel_height / column_width * frames.shape[2] / frames.shape[1])
    frame_height = frame_width * column_width * frames.shape[1] / frames.shape[2] / panel_height
    frame_bottom = 1. - frame_height
    axis.text(1.5, 1.05, f'“{statement}”', va='bottom', ha='center', fontsize=8.8)
    for column, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        center = column + .5
        left = center - frame_width / 2
        axis.imshow(frame, extent=(left, left + frame_width, frame_bottom, 1.),
                    aspect='auto', interpolation='antialiased')
        for y, prediction, alpha in ((frame_bottom - .034, anchor[column], .38),
                                     (frame_bottom - .102, fused[column], 1.)):
            for width, start, color in zip(prediction, np.r_[0., prediction.cumsum()[:-1]], CLASS_COLORS, strict=True):
                axis.add_patch(Rectangle((left + frame_width * start, y - .047), frame_width * width, .047,
                                         facecolor=color, edgecolor='white', linewidth=.35, alpha=alpha))
        before, after = anchor[column, label], fused[column, label]
        axis.text(center, frame_bottom - .169, f'{before:.1%} → {after:.1%}',
                  fontsize=7.75, color=CLASS_COLORS[label], fontweight='bold', ha='center', va='top')
    axis.text(.025, frame_bottom - .060, 'Anchor', fontsize=7., ha='right', va='center')
    axis.text(.025, frame_bottom - .128, 'Fusion', fontsize=7., ha='right', va='center')


def coupling_evidence(runs):
    c = report(runs, 'sims_v2_compact_visual')['supervision']['versus_independent_rps']
    r = report(runs, 'ravdess_effects')['primary_shared_versus_independent_balanced_rps']
    crema = report(runs, 'cremad_effects')
    figure = plt.figure(figsize=(6.75, 3.87))
    grid = figure.add_gridspec(2, 2, left=.139, right=.940, bottom=.137, top=.912,
                              hspace=.11, wspace=.50, width_ratios=(1.12, 1.),
                              height_ratios=(2.19, 1.))
    examples = figure.add_subplot(grid[0, :])
    fits, components = [figure.add_subplot(grid[1, column]) for column in (0, 1)]
    effects = [c, r, crema['balanced_rps_reductions']['independent_intensity']]
    per_seed = 1000 * np.asarray([effect['by_seed'] for effect in effects])
    np.testing.assert_allclose(per_seed.mean(1), 1000 * np.asarray([effect['reduction'] for effect in effects]))
    confidence = fits.imshow(per_seed, extent=(.5, per_seed.shape[1] + .5, -.5, 2.5), cmap=CHANGE,
                             norm=TwoSlopeNorm(vcenter=0, vmin=-35, vmax=35), interpolation='none', aspect='auto')
    fits.vlines(np.arange(1.5, per_seed.shape[1] + .5), -.5, 2.5, color='white', linewidth=.50)
    fits.hlines([.5, 1.5], .5, per_seed.shape[1] + .5, color='white', linewidth=.65)
    fits.set_xticks([1, 5, per_seed.shape[1]])
    fits.set_yticks((2, 1, 0), ('CH-SIMS v2', 'RAVDESS', 'CREMA-D'))
    fits.tick_params(axis='y', length=0, pad=4)
    fits.set_xlabel('Paired fit', labelpad=3)
    fits.spines[['left', 'bottom']].set_visible(False)
    key = fits.inset_axes([1.026, .14, .026, .72])
    colorbar = figure.colorbar(confidence, cax=key, ticks=[-35, 0, 35])
    colorbar.ax.tick_params(length=2, pad=1.6, labelsize=7.25)
    colorbar.outline.set_visible(False)
    class_names = ('negative', 'neutral', 'positive')
    values = 1000 * np.array([crema['contributions_by_polarity'][p] for p in class_names])
    for j, (color, offset) in enumerate(((MAGENTA, .20), (INDIGO, -.20))):
        for row, value in zip((2, 1, 0), values[:, j], strict=True):
            components.barh(row + offset, value, height=.205, color=color, zorder=2)
            position = max(value, 1.7) + .8 if value >= 0 else value - .8
            components.text(position, row + offset, f'{value:.1f}'.replace('-', '−'),
                            ha='left' if value >= 0 else 'right', va='center', fontsize=6.8)
    forest_axis(components, POLARITY, r'RPS component gain ($10^{-3}$)', limits=(-18, 53))
    components.grid(False, axis='y')
    components.legend(handles=[Patch(color=MAGENTA, label='Signed'), Patch(color=INDIGO, label='Neutral')],
                      loc='lower right', bbox_to_anchor=(1., 1.080), ncol=2, frameon=False, fontsize=7.35,
                      handlelength=1, handletextpad=.38, columnspacing=.68, borderpad=0)
    figure.canvas.draw()
    label_edge = left_of_y_labels(fits)
    example_bounds = examples.get_position()
    data_center = (label_edge + components.get_position().x1) / 2
    examples.set_position(example_bounds.translated(data_center - (example_bounds.x0 + example_bounds.x1) / 2, 0))
    statement_predictions(examples, runs)
    for axis, letter in ((examples, 'a'), (fits, 'b')):
        position = axis.get_position()
        panel(axis, letter, x=(label_edge - position.x0) / position.width)
    panel(components, 'c')
    return figure


def within_polarity_examples(runs):
    archived = runs / 'within_polarity_examples.npz'
    if archived.is_file():
        with np.load(archived, allow_pickle=False) as observed:
            return observed['clip_numbers'], observed['ratings'], observed['updates']
    with np.load(runs / 'conditional_intensity_evidence.npz', allow_pickle=False) as data:
        ids, videos, labels, scores = (data[f'ch_sims_v2__{key}']
                                      for key in ('ids', 'groups', 'y', 'sentiment'))
        genuine, permuted = (data[f'ch_sims_v2__conditional_{arm}_signed_shift']
                             for arm in ('shared', 'shuffled'))

    mean_updates = 2 * np.stack((genuine, permuted)).mean(axis=(1, 2))
    samples = []
    for polarity in (0, 2):
        candidates = []
        for video in np.unique(videos):
            subset = np.flatnonzero((videos == video) & (labels == polarity))
            if len(subset) == 8 and len(np.unique(scores[subset])) >= 3:
                candidates.append((round(float(np.ptp(scores[subset])), 4), video, subset))
        candidates.sort(key=lambda candidate: candidate[:2])
        _, _, sample = candidates[len(candidates) // 2]
        samples.append(sample[np.lexsort((ids[sample], scores[sample]))])
    clips = np.array([[int(ids[index].rsplit('$_$', 1)[-1]) for index in subset] for subset in samples])
    return clips, scores[np.asarray(samples)], np.stack([mean_updates[:, subset] for subset in samples])


def within_polarity_evidence(runs):
    clips, ratings, updates = within_polarity_examples(runs)
    figure = plt.figure(figsize=(6.70, 2.08))
    grid = figure.add_gridspec(1, 3, left=.027, right=.973, bottom=.37, top=.835,
                              wspace=.065, width_ratios=(1., .31, 1.))
    cases = [figure.add_subplot(grid[0, column]) for column in (0, 2)]
    bounds = cases[0].get_position()
    row_labels = ('Human rating', 'Fit on original\nratings', 'Fit on shuffled\nratings')
    for row, label in enumerate(row_labels):
        height = bounds.y1 - (row + .5) * bounds.height / len(row_labels)
        figure.text(.5, height, label, ha='center', va='center', fontsize=7.35)
    figure.text(.5, .255, 'Clip number within video', ha='center', va='center', fontsize=8)
    colorbar_axis = figure.add_axes([.335, .144, .33, .050])
    position = np.arange(8)
    scale = np.ceil(2 * np.max(np.abs(updates))) / 2
    for column, axis in enumerate(cases):
        values = np.vstack((np.full(8, np.nan), updates[column]))
        image = axis.imshow(values, cmap=CHANGE.with_extremes(bad=GRID), vmin=-scale, vmax=scale,
                            interpolation='nearest', aspect='auto')
        panel(axis, 'ab'[column], y=1.12)
        axis.set_xticks(position, [str(clip) for clip in clips[column]])
        axis.set_yticks([])
        axis.tick_params(axis='both', length=0)
        axis.vlines(np.arange(.5, 7.5), -.5, 2.5, color='white', linewidth=.85)
        axis.hlines([.5, 1.5], -.5, 7.5, color='white', linewidth=.95)
        axis.spines[['left', 'bottom']].set_visible(False)
        if column == 1:
            example, = np.flatnonzero(clips[column] == 14)
            axis.add_patch(Rectangle((example - .48, .52), .96, 1.96, facecolor='none',
                                     edgecolor=INDIGO, linewidth=1.0, zorder=3))
        for slot, value in enumerate(ratings[column]):
            axis.text(slot, 0, f'{value:.1f}'.replace('-', '−'), ha='center', va='center',
                      color=INK, fontsize=7.0)
        for row, series in enumerate(updates[column], start=1):
            for slot, update in enumerate(series):
                rgb = image.cmap(image.norm(update))[:3]
                brightness = np.dot(rgb, (.2126, .7152, .0722))
                value = f'{round(float(update), 2) + 0.0:.2f}'.replace('-', '−')
                axis.text(slot, row, value, ha='center', va='center',
                          color='white' if brightness < .49 else INK, fontsize=6.6)
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation='horizontal', ticks=[-scale, 0, scale])
    colorbar.set_label('Change in positive-to-negative log odds', size=7.6, labelpad=2)
    colorbar.ax.tick_params(length=2, pad=1, labelsize=7)
    colorbar.outline.set_visible(False)

    return figure


def plot_gap_locations(figure, grid, runs):
    with np.load(runs / 'gap_examples.npz', allow_pickle=False) as packet:
        data = dict(packet)
    outer = grid.subgridspec(3, 2, height_ratios=(1.43, 1., 1.), hspace=.32, wspace=.16)
    inputs = outer[0, :].subgridspec(2, 2, height_ratios=(1.28, 1.), hspace=.055, wspace=.21)
    scored = 100 * (data['intensity'] - data['anchor'])
    probability_axes, intensity_axes = [], []
    probability_limits = [0.0]
    intensity_limits = list(data['sentiment'])
    for case in range(2):
        frames = figure.add_subplot(inputs[0, case])
        top = filmstrip(frames, data['frames'][case], data['frame_slots'])
        panel(frames, 'ab'[case], y=top + .109)
        feature = figure.add_subplot(inputs[1, case])
        values = normalized_channels(data['visual'][case].T)
        feature_strip(feature, values, data['masks'][0, case], 2., height=.75)
        for row, view in ((1, 1), (0, 2)):
            feature_strip(feature, values, data['masks'][view, case], row, height=.74)
        feature.set(xlim=(0, 24), ylim=(-.4, 2.41), xticks=[])
        feature.set_yticks([2, 1, 0], ['Full', '35%', '65%'] if case == 0 else ['', '', ''])
        feature.spines[['left', 'bottom']].set_visible(False)
        feature.tick_params(axis='y', length=0, pad=4, labelsize=7.25)
        probability = figure.add_subplot(outer[1, case])
        intensity = figure.add_subplot(outer[2, case])
        probability_axes.append(probability)
        intensity_axes.append(intensity)
        probability.axhline(0, color='black', linewidth=.66, zorder=0)
        intensity.axhline(float(data['sentiment'][case]), color='black', linewidth=.72,
                         linestyle=(0, (3, 3)), zorder=1)
        for start, count, width, color, style in zip(data['offsets'], data['counts'], data['widths'],
                                                     (TEAL, INDIGO), ('-', '--'), strict=True):
            gap_centers = np.arange(count) + width / 2
            window = slice(start, start + count)
            for axis, samples, bounds in (
                (probability, scored[:, window, case, data['y'][case]], probability_limits),
                (intensity, data['intensity_score'][:, window, case], intensity_limits),
            ):
                low, high = np.quantile(samples, [.25, .75], axis=0)
                axis.fill_between(gap_centers, low, high, color=color, alpha=.16, linewidth=0, zorder=1)
                axis.plot(gap_centers, samples.mean(0), color=color, linestyle=style, linewidth=1.21, zorder=3)
                bounds.extend((low.min(), high.max(), samples.mean(0).min(), samples.mean(0).max()))
        for axis, letter in ((probability, 'cd'[case]), (intensity, 'ef'[case])):
            panel(axis, letter, y=1.035)
            axis.set_xlim(0, 24)
            axis.set_xticks([.5, 5.5, 11.5, 17.5, 23.5], ['1', '6', '12', '18', '24'])
            axis.yaxis.set_major_locator(MaxNLocator(3))
            if case == 0:
                axis.set_ylabel('Rated-class probability\nchange (pp)' if axis is probability else 'Signed intensity',
                                labelpad=3)
            else:
                axis.tick_params(axis='y', labelleft=False)
        probability.tick_params(axis='x', labelbottom=False, length=2)
        intensity.set_xlabel('Gap center (time bin)', labelpad=3.5)
    for axes, points in ((probability_axes, probability_limits), (intensity_axes, intensity_limits)):
        low, high = min(points), max(points)
        padding = (high - low) * .09
        for axis in axes:
            axis.set_ylim(low - padding, high + padding)
    return intensity_axes[0].get_position().y0


def robustness_evidence(runs):
    figure = plt.figure(figsize=(6.75, 4.13))
    grid = figure.add_gridspec(1, 1, left=.145, right=.976, top=.930, bottom=.156)
    traces_bottom = plot_gap_locations(figure, grid[0, :], runs)
    figure.legend(handles=[Line2D([], [], color=TEAL, label='35% gap'),
                           Line2D([], [], color=INDIGO, ls='--', label='65% gap'),
                           Line2D([], [], color='black', ls=(0, (3, 3)), label='Audience rating')],
                  loc='lower center', bbox_to_anchor=(.550, traces_bottom - .151), ncol=3,
                  frameon=False, fontsize=7.65, handletextpad=.48)
    return figure


def plot_gain_transitions(axis, counts):
    margins = (counts.sum(1), counts.sum(0))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    scale = .69 / counts.sum()
    colors = (TEAL, MAGENTA)
    labels = ('Lower RPS', 'No RPS gain')
    left, right = .140, .860
    left_control = left + .38 * (right - left)
    right_control = right - .38 * (right - left)
    starts = [[.80, .80 - sizes[0] * scale - .055] for sizes in margins]
    for source in range(len(labels)):
        for target in range(len(labels)):
            left_top = starts[0][source] - scale * counts[source, :target].sum()
            right_top = starts[1][target] - scale * counts[:source, target].sum()
            count = counts[source, target]
            height = count * scale
            coordinates = [(left, left_top), (left_control, left_top), (right_control, right_top), (right, right_top),
                           (right, right_top - height), (right_control, right_top - height),
                           (left_control, left_top - height), (left, left_top - height), (left, left_top)]
            operations = [ArtistPath.MOVETO, ArtistPath.CURVE4, ArtistPath.CURVE4, ArtistPath.CURVE4,
                          ArtistPath.LINETO, ArtistPath.CURVE4, ArtistPath.CURVE4, ArtistPath.CURVE4,
                          ArtistPath.CLOSEPOLY]
            axis.add_patch(PathPatch(ArtistPath(coordinates, operations), facecolor=colors[target],
                                     alpha=.53 if source == target else .25, edgecolor='white', linewidth=.58))
            if count:
                axis.text(left + .078, left_top - height / 2, f'{count:,}', ha='center', va='center',
                          fontsize=7.2, color=INK)
    for x, dx, name, starts_at, sizes, alignment in (
        (left, -.022, 'Frozen', starts[0], margins[0], 'right'),
        (right, .022, 'Refitted', starts[1], margins[1], 'left'),
    ):
        axis.text(x, .935, name, ha='center', va='bottom', fontsize=8.1)
        for start, count, label, color in zip(starts_at, sizes, labels, colors, strict=True):
            height = scale * count
            axis.add_patch(Rectangle((x - .009, start - height), .018, height, facecolor=color, edgecolor='none'))
            axis.text(x + dx, start - height / 2, f'{label}\n{count:,}', ha=alignment, va='center',
                      color=INK, fontsize=7.3, linespacing=1.28)


def transfer_evidence(runs):
    moud = report(runs, 'moud_frozen_transfer_full')
    meld = report(runs, 'meld_frozen_transfer')
    frozen = report(runs, 'msp_face_transfer_matched')
    refit = report(runs, 'msp_face_refit_paired')
    figure = plt.figure(figsize=(6.75, 3.85))
    grid = figure.add_gridspec(2, 2, left=.150, right=.894, bottom=.09, top=.882,
                              hspace=.43, wspace=.49, height_ratios=(1.17, .95))
    overall, target = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    switching = figure.add_subplot(grid[1, :])
    for row, source in zip((2, 1, 0), (moud, meld, frozen), strict=True):
        methods = source['methods']
        prior = methods['anchor'].get('gapped', methods['anchor'])['rps']
        shared = methods['intensity'].get('gapped', methods['intensity'])['rps']
        overall.hlines(row, prior, shared, color=SLATE, linewidth=1.05)
        overall.vlines(prior, row - .105, row + .105, color=SLATE, linewidth=1.15)
        overall.vlines(shared, row - .105, row + .105, color=TEAL, linewidth=1.85)
        overall.text(prior - .0015, row, f'{prior:.3f}', color=SLATE, ha='right', va='center', fontsize=7.)
        overall.text(shared + .0015, row, f'{shared:.3f}', color=TEAL, ha='left', va='center', fontsize=7.)
    overall.set_yticks([2, 1, 0], ['MOUD', 'MELD', 'MSP-Face'])
    overall.set_ylim(-.46, 2.46)
    overall.set_xlim(.198, .321)
    overall.set_xlabel('Frozen RPS', labelpad=4)
    overall.xaxis.set_major_locator(MaxNLocator(5))
    overall.spines['left'].set_visible(False)
    overall.tick_params(axis='y', length=0, pad=5)
    target.axhline(0, color='black', linewidth=.72, linestyle=(0, (3, 3)), zorder=0)
    for compare, offset, color, style in [('primary', -.04, TEAL, '-'), ('anchor', .04, INDIGO, '--')]:
        estimates = [source['primary'] if compare == 'primary' else source['secondary'][compare]
                     for source in (frozen, refit)]
        position = np.arange(2) + offset
        mean = 1000 * np.asarray([effect['reduction'] for effect in estimates])
        low, high = 1000 * np.asarray([effect['ci95_familywise'] for effect in estimates]).T
        target.plot(position, mean, color=color, linestyle=style, linewidth=1.25, zorder=2)
        target.errorbar(position, mean, yerr=[mean - low, high - mean], fmt='none', ecolor=color,
                        capsize=2.7, capthick=.88, elinewidth=.88, alpha=.88, zorder=3)
        offset_x, alignment, offset_y = (-.043, 'right', 1.15) if compare == 'primary' else (.06, 'left', -1.15)
        for x, value in zip(position, mean, strict=True):
            target.text(x + offset_x, value + offset_y, f'{value:.1f}'.replace('-', '−'),
                        color=color, ha=alignment, va='center', fontsize=6.9)
    target.set_xticks([0, 1], ['Frozen', 'Refitted'])
    target.set_xlim(-.25, 1.25)
    target.set_ylabel(r'RPS reduction ($10^{-3}$)', labelpad=2)
    target.yaxis.set_major_locator(MaxNLocator(5))
    panel(overall, 'a', y=1.075)
    panel(target, 'b', y=1.075)
    if 'clip_gain_transitions' in refit:
        counts = np.asarray(refit['clip_gain_transitions'])
        if counts.shape != (2, 2) or np.any(counts < 0) or counts.sum() != refit['n_clips']:
            raise ValueError('Invalid counts for the matched MSP-Face clips.')
    else:
        from msp_face_effects import clip_gain_transitions

        with (np.load(runs / 'msp_face_transfer_matched_evidence.npz', allow_pickle=False) as frozen_cases,
              np.load(runs / 'msp_face_refit_evidence.npz', allow_pickle=False) as refitted_cases):
            for key in ('ids', 'seeds', 'y', 'sentiment', 'masks'):
                np.testing.assert_array_equal(frozen_cases[key], refitted_cases[key])
            y = frozen_cases['y']
            changes = [
                (supporting.ranked_probability_score(table['anchor'][:, 1:3], y)
                 - supporting.ranked_probability_score(table['intensity'][:, 1:3], y)).mean((0, 1))
                for table in (frozen_cases, refitted_cases)
            ]
        np.testing.assert_allclose(np.mean(changes, axis=1),
                                   [x['secondary']['anchor']['reduction'] for x in (frozen, refit)], atol=1e-7)
        counts = clip_gain_transitions(*changes)
    for axis, entries in ((overall, [(SLATE, '-', 'Text anchor'), (TEAL, '-', 'Shared')]),
                          (target, [(TEAL, '-', 'vs. independent'), (INDIGO, '--', 'vs. anchor')])):
        handles = [Line2D([], [], color=color, ls=style, label=label) for color, style, label in entries]
        axis.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.104), ncol=2, frameon=False,
                    fontsize=7.2, handlelength=1.28, handletextpad=.34, columnspacing=.62, borderpad=0)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    upper = Bbox.union([axis.get_tightbbox(renderer) for axis in (overall, target)])
    upper_bounds = upper.transformed(figure.transFigure.inverted())
    position = switching.get_position()
    switching.set_position(Bbox.from_extents(upper_bounds.x0, position.y0, upper_bounds.x1, position.y1))
    plot_gain_transitions(switching, counts)
    flow = Bbox.union([artist.get_window_extent(renderer) for artist in (*switching.texts, *switching.patches)])
    shift = (flow.x0 + flow.x1 - upper.x0 - upper.x1) / (2 * switching.bbox.width)
    switching.set_xlim(shift, 1 + shift)
    panel(switching, 'c', y=1.08)
    return figure


def plot_audiovisual_example(runs, output):
    with np.load(runs / 'audiovisual_example.npz', allow_pickle=False) as packet:
        clip = dict(packet)
    figure = plt.figure(figsize=(6.55, 2.89))
    layout = figure.add_gridspec(2, 2, left=.164, right=.966, top=.78, bottom=.19,
                                 height_ratios=(1., 1.85), hspace=.43, wspace=.21)
    figure.text(.164, .939, f'“{clip["transcript_en"].item()}”', fontsize=8.4, va='top', color=INK)
    rated = int(clip['polarity'])
    inputs = ('audio', 'vision')
    forecasts = ('anchor', 'text_audio', 'text_vision', 'fused')
    labels = ('Text anchor', 'Text + audio', 'Text + vision', 'Text + audio + vision')
    for column, view in enumerate((1, 2)):
        features = figure.add_subplot(layout[0, column])
        predictions = figure.add_subplot(layout[1, column])
        panel(features, 'ab'[column], y=1.13)
        for row, name, mask in zip((1, 0), inputs, clip['masks'][view], strict=True):
            feature_strip(features, normalized_channels(clip[name]), mask, row, height=.73)
        features.set(xlim=(0, 24), ylim=(-.49, 1.49), xticks=(.5, 11.5, 23.5),
                     xticklabels=('1', '12', '24'))
        features.tick_params(axis='x', length=2, pad=2)
        features.set_yticks((1, 0), ('Audio', 'Vision') if column == 0 else ('', ''))
        features.tick_params(axis='y', length=0, pad=4)
        features.spines['left'].set_visible(False)

        score_x = 1.16
        predictions.text(score_x, 3.62, 'Intensity', fontsize=7.1, ha='center', va='bottom')
        for row, name in zip(range(3, -1, -1), forecasts, strict=True):
            estimate = clip[name][view]
            alpha = 1. if name == 'fused' else .46
            left = np.r_[0., estimate.cumsum()[:-1]]
            predictions.barh(row, estimate, left=left, height=.61, color=CLASS_COLORS,
                             alpha=alpha, edgecolor='none')
            value = estimate[rated]
            predictions.text(left[rated] + value / 2, row, f'{value:.1%}',
                             color='white' if name == 'fused' else INK,
                             ha='center', va='center', fontsize=7.3)
            if name != 'anchor':
                score = f'{float(clip[f"{name}_score"][view]):.2f}'.replace('-', '−')
                predictions.text(score_x, row, score, color=INK, ha='center', va='center', fontsize=7.1)
        predictions.set(xlim=(0, 1.29), ylim=(-.46, 3.51), xticks=(0, .5, 1))
        predictions.spines['bottom'].set_bounds(0, 1.)
        predictions.set_yticks(tuple(range(3, -1, -1)), labels if column == 0 else ('',) * 4)
        predictions.tick_params(axis='y', length=0, pad=4, labelsize=7.5)
        predictions.tick_params(axis='x', length=2, pad=2)
        predictions.spines['left'].set_visible(False)
    figure.text(.565, .063, 'Class probability', fontsize=8.1, ha='center', va='bottom')
    export(figure, output)


def supplementary(runs, output, alignment_script, report_directory):
    single_argument = {
        'audiovisual_example': plot_audiovisual_example,
        'fixed_transcripts': supporting.plot_fixed_transcripts,
        'frozen_dialogue': supporting.plot_frozen_dialogue,
        'frozen_lnln_heads': supporting.plot_frozen_lnln_heads,
        'intensity_calibration': supporting.plot_score_gain,
        'modality_agreement': supporting.plot_modality_ratings,
        'moud_transfer': supporting.plot_moud_transfer,
        'msp_face_transfer': supporting.plot_msp_face_transfer,
        'neutral_readout': supporting.plot_neutral_readout,
        'ordinal_controls': supporting.plot_ordinal_controls,
        'probability_pool': supporting.plot_probability_pool,
        'seed_sensitivity': supporting.plot_seed_sensitivity,
        'visual_components': supporting.plot_visual_components,
    }
    with_layout = {
        'audio_channel': supporting.plot_audio_channel,
        'class_weighting': supporting.plot_class_weighting,
        'conditional_intensity': supporting.plot_conditional_intensity,
        'cremad': supporting.plot_cremad,
        'dispersion_pooling': supporting.plot_dispersion_pooling,
        'face_crops': supporting.plot_face_crops,
        'gap_location': supporting.plot_gap_location,
        'marginal_tilt': supporting.plot_marginal_tilt,
        'msp_face_refit': supporting.plot_msp_face_refit,
        'seed_ensemble': supporting.plot_seed_ensemble,
        'view_stability': supporting.plot_view_stability,
    }
    report = report_directory / 'benchmark_components_alignment.json' if report_directory else None
    supporting.plot_benchmark_components(runs, output / 'matched_checks.pdf', alignment_script, report)
    with TemporaryDirectory(prefix='kbs_supporting_panels_') as temporary:
        directory = Path(temporary)
        drawn = set()
        for name, blocks in SUPPLEMENT_PLATES.items():
            parts = list(plate_parts(directory, blocks))
            for part in parts:
                source = part.path.stem
                if source in single_argument:
                    single_argument[source](runs, part)
                else:
                    layout = report_directory / f'{source}_alignment.json' if report_directory else None
                    with_layout[source](runs, part, alignment_script, layout)
                drawn.add(source)
            assemble_plate(output / f'{name}.pdf', parts)
        if drawn != set(single_argument) | set(with_layout):
            raise ValueError('The supplementary plates must account for every diagnostic plot.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alignment-script', type=Path)
    parser.add_argument('--report-directory', type=Path)
    parser.add_argument('--supplement', action='store_true')
    args = parser.parse_args()
    configure_figure()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.supplement:
        supplementary(args.runs, args.output, args.alignment_script, args.report_directory)
        return
    if args.alignment_script:
        sys.path.insert(0, str(args.alignment_script))
        from audit_panel_alignment import require_matplotlib_panel_alignment
    for name, render in [('model_trace', model_trace), ('coupling_evidence', coupling_evidence),
                         ('within_polarity_evidence', within_polarity_evidence),
                         ('robustness_evidence', robustness_evidence), ('transfer_evidence', transfer_evidence)]:
        figure = render(args.runs)
        if args.alignment_script:
            log = args.report_directory / f'{name}_alignment.json' if args.report_directory else None
            mixed_rows = {
                'model_trace': ('c', 'The two full-width rows align to the observation labels.'),
                'coupling_evidence': ('a', 'The frame strip centers over the labeled result panels.'),
                'transfer_evidence': ('c', 'The flow spans both plots and their outer labels.'),
            }
            exemptions = []
            if name in mixed_rows:
                letter, reason = mixed_rows[name]
                exemptions.append(dict(panels=[letter], checks=['column'], reason=reason))
            require_matplotlib_panel_alignment(figure, json_out=log, exemptions=exemptions)
        export(figure, args.output / f'{name}.pdf', align_panels=True)


if __name__ == '__main__':
    main()
