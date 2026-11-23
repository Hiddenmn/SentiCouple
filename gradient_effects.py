"""Pair the clipped signed update with the original and uniform auxiliary updates."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions
from compact_visual import stats
from compare_benchmarks import load_matched, load_metadata, summarize_run
from score_anchor_effects import SEEDS


def evaluate(reference_paths, capped_paths, resamples):
    reference = read_predictions(reference_paths[0])
    paths = {'original': reference_paths, 'trial': capped_paths}
    packets = {name: load_matched(runs, reference) for name, runs in paths.items()}
    metadata = {name: load_metadata(runs) for name, runs in paths.items()}
    for seed, left, right, original, trial in zip(SEEDS, metadata['original'], metadata['trial'],
                                                  packets['original'], packets['trial']):
        a, b = left['config'], right['config']
        if a['seed'] != seed or b['seed'] != seed or a['visual_encoding'] != 'pointwise':
            raise ValueError(f'Expected matched pointwise runs for seed {seed}.')
        for field in a.keys() - {'output', 'arms'}:
            if a[field] != b.get(field):
                raise ValueError(f'Mismatched {field} for seed {seed}.')
        np.testing.assert_array_equal(original['anchor'], trial['anchor'])

    label, sentiment, groups = (reference[key] for key in ('y', 'sentiment', 'groups'))
    _, membership = np.unique(groups, return_inverse=True)
    methods, losses = {}, {}
    for name, source, arm in [('original', 'original', 'intensity'),
                              ('capped', 'trial', 'intensity_capped'),
                              ('uniform', 'trial', 'intensity_uniform')]:
        metrics, rps, mae = summarize_run(packets[source], metadata[source], arm, label, sentiment)
        metrics['score_gain'] = [run['selection'][arm]['score_gain'] for run in metadata[source]]
        if name != 'original':
            metrics['clipped_fraction'] = [run['selection'][arm]['clipped_fraction'] for run in metadata[source]]
        methods[name], losses[name] = metrics, (rps, mae)
    pairs = [('versus_original', 'original', 'capped'), ('versus_uniform', 'uniform', 'capped')]
    contrasts = stats(losses, pairs, membership, resamples)
    return dict(seeds=SEEDS, n_clips=len(label), n_videos=int(membership.max()) + 1,
                label_counts=np.bincount(label, minlength=3).tolist(), methods=methods, contrasts=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-prefix', required=True)
    parser.add_argument('--trial-prefix', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resamples', default=10000, type=int)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose at least one resample.')
    report = evaluate([Path(f'{args.reference_prefix}{seed}') for seed in SEEDS],
                      [Path(f'{args.trial_prefix}{seed}') for seed in SEEDS], args.resamples)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, summary in report['methods'].items():
        print(f'{name:10s} RPS={summary["rps"]:.6f}, MAE={summary["mae"]:.6f}')
    for name, effect in report['contrasts'].items():
        low, high = effect['ci95_familywise']
        print(f'{name:24s} {effect["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]')


if __name__ == '__main__':
    main()
