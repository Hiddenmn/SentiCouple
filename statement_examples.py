"""Extract a matched CREMA-D triplet without using the model's predictions for selection."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ablate import read_predictions
from compare_benchmarks import load_matched
from figure_example import known_statements
from prepare_cremad import frames_at_centers
from score_anchor_effects import SEEDS


def matched_triplet(ids, labels, ratings, valid):
    cohorts = defaultdict(list)
    for row in np.flatnonzero(valid):
        actor, line, _, _ = ids[row].split('$_$', 1)[1].split('_')
        cohorts[actor, line].append(row)
    medians = np.array([np.median(ratings[valid & (labels == polarity)]) for polarity in range(3)])
    choices = []
    for (actor, line), rows in cohorts.items():
        by_label = [[row for row in rows if labels[row] == polarity] for polarity in range(3)]
        if not all(by_label):
            continue
        indices = np.array([
            min(group, key=lambda row: (abs(ratings[row] - median), ids[row]))
            for group, median in zip(by_label, medians, strict=True)
        ])
        distance = np.abs(ratings[indices] - medians).sum()
        choices.append((distance, actor, line, indices))
    if not choices:
        raise ValueError('No held-out actor and statement had fully observed clips of all three polarities.')
    _, _, line, indices = min(choices, key=lambda group: group[:3])
    return line, indices


def collect(runs, release):
    paths = [runs / f'cremad_heads_seed{seed}' for seed in SEEDS]
    reference = read_predictions(paths[0])
    packets = load_matched(paths, reference)
    valid = reference['masks'][0, 1].all(axis=1)
    line, indices = matched_triplet(reference['ids'], reference['y'], reference['sentiment'], valid)
    ids = reference['ids'][indices]
    labels = reference['y'][indices]
    slots = np.array([5, 17])
    frames = []
    for clip in ids:
        name = clip.split('$_$', 1)[1]
        native = frames_at_centers(release / 'VideoFlash' / f'{name}.flv')
        if len(native) != 24:
            raise ValueError(f'Expected 24 native frame centers for {name}.')
        frames.append([native[slot] for slot in slots])
    prior = np.stack([packet['anchor'][0, indices] for packet in packets])
    np.testing.assert_allclose(prior, np.repeat(prior[:, :1], 3, axis=1), atol=2e-6, rtol=2e-5)
    return dict(ids=ids, labels=labels, ratings=reference['sentiment'][indices],
                seeds=np.asarray(SEEDS), statement=known_statements(release)[line],
                frames=np.asarray(frames), slots=slots, anchor=prior,
                fused=np.stack([packet['intensity'][0, indices] for packet in packets]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.runs, args.release)
    np.savez_compressed(args.output, **data)
    print(data['statement'])
    print(data['ids'].tolist(), data['ratings'].tolist())


if __name__ == '__main__':
    main()
