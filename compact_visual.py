"""Compare learned visual pooling and local temporal mixing under identical score calibration."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions, seed_video_interval
from compare_benchmarks import load_matched, load_metadata, summarize_run
from score_anchor_effects import SEEDS


def stats(losses, comparisons, membership, replicates):
    tail = .025 / (2 * len(comparisons))
    result = {}
    for contrast, baseline, candidate in comparisons:
        for index, metric in enumerate(('rps', 'mae')):
            differences = losses[baseline][index] - losses[candidate][index]
            key = f'{contrast}_{metric}'
            result[key] = dict(reduction=float(differences.mean()), by_seed=differences.mean(axis=1).tolist(),
                               ci95_familywise=seed_video_interval(differences, membership, replicates,
                                                                   quantiles=(tail, 1 - tail)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pooled', required=True, type=Path, nargs='+')
    parser.add_argument('--attention', required=True, type=Path, nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resamples', default=10000, type=int)
    args = parser.parse_args()
    if len(args.pooled) != len(SEEDS) or len(args.attention) != len(SEEDS) or args.resamples < 1:
        parser.error('Provide ten paired runs for each architecture in the protocol seed order.')
    reference = read_predictions(args.pooled[0])
    sources = {key: load_matched(paths, reference) for key, paths in
               [('pooled', args.pooled), ('attention', args.attention)]}
    configs = {key: load_metadata(paths) for key, paths in [('pooled', args.pooled), ('attention', args.attention)]}
    for pooled, attention, left, right, seed in zip(configs['pooled'], configs['attention'],
                                                    sources['pooled'], sources['attention'], SEEDS):
        a, b = pooled['config'], attention['config']
        if (a['seed'] != seed or b['seed'] != seed or a['backbone'] != 'temporal' or
                b['backbone'] != 'stream-query' or not a['calibrate_score_gain'] or not b['calibrate_score_gain']):
            raise ValueError(f'Expected matched calibrated pooling architectures for seed {seed}.')
        for key in a.keys() - {'arms', 'output', 'backbone'}:
            if a[key] != b.get(key):
                raise ValueError(f'Seed {seed} has a mismatched {key}.')
        np.testing.assert_array_equal(left['anchor'], right['anchor'])
    labels, scores, groups = (reference[key] for key in ('y', 'sentiment', 'groups'))
    _, membership = np.unique(groups, return_inverse=True)
    methods, losses = {}, {}
    arms = [('attention', 'attention', 'intensity'),
            ('pooled', 'pooled', 'intensity'),
            ('per_bin', 'pooled', 'intensity_pointwise'),
            ('class_only', 'pooled', 'fusion'),
            ('independent', 'pooled', 'independent_intensity')]
    for name, source, arm in arms:
        methods[name], rps, mae = summarize_run(sources[source], configs[source], arm, labels, scores)
        losses[name] = (rps, mae)
        methods[name]['score_gain'] = [row['selection'][arm]['score_gain'] for row in configs[source]]
    main_effects = [('learned_pooling', 'pooled', 'attention'), ('temporal_mixing', 'per_bin', 'pooled')]
    head_effects = [('versus_class_only', 'class_only', 'pooled'), ('versus_independent', 'independent', 'pooled')]
    architecture = stats(losses, main_effects, membership, args.resamples)
    supervision = stats(losses, head_effects, membership, args.resamples)
    result = dict(seeds=list(SEEDS), n_clips=len(labels), n_videos=int(membership.max()) + 1,
                  resamples=args.resamples, methods=methods, architecture=architecture, supervision=supervision)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for key, effect in {**architecture, **supervision}.items():
        lo, hi = effect['ci95_familywise']
        print(f'{key:24s} {effect["reduction"]:+.6f} [{lo:+.6f}, {hi:+.6f}]')


if __name__ == '__main__':
    main()
