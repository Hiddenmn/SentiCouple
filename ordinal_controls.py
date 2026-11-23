"""Compare shared intensity with two ordinal losses and class mean targets."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions, seed_video_interval
from compare_benchmarks import load_matched, load_metadata, seed_clip_accuracy, summarize_run
from score_anchor_effects import SEEDS


COHORTS = {
    'ch_sims_v2': ('sims_v2_compact_pooled_seed', 'sims_v2_ordinal_controls_seed',
                  'sims_v2_independent_class_mean_seed'),
    'omg_emotion': ('omg_pointwise_heads_seed', 'omg_ordinal_controls_seed',
                    'omg_independent_class_mean_seed'),
}
CONTROLS = {'rps_only': 'fusion_rps', 'ce_plus_rps': 'fusion_ce_rps',
            'class_mean_intensity': 'intensity_class_mean'}
REFERENCES = {'class_only': 'fusion', 'independent': 'independent_intensity', 'shared': 'intensity'}


def paired_reduction(values, membership, resamples, family_size):
    tail = .025 / family_size
    return dict(reduction=float(values.mean()), by_seed=values.mean(axis=1).tolist(),
                ci95_familywise=seed_video_interval(values, membership, resamples,
                                                    quantiles=(tail, 1 - tail)))


def verify_pairs(references, trials, reference_meta, trial_meta):
    for seed, left, right, left_meta, right_meta in zip(SEEDS, references, trials, reference_meta, trial_meta):
        left_config, right_config = left_meta['config'], right_meta['config']
        for key in (left_config.keys() | right_config.keys()) - {'arms', 'output'}:
            if left_config.get(key) != right_config.get(key):
                raise ValueError(f'Seed {seed} differs in {key}.')
        if left_config['seed'] != seed or left_meta['splits'] != right_meta['splits']:
            raise ValueError(f'Seed {seed} or its splits do not match.')
        if not left_config['calibrate_score_gain']:
            raise ValueError(f'Seed {seed} lacks tuning-set score calibration.')
        np.testing.assert_array_equal(left['anchor'], right['anchor'])
        if 'intensity' in right:
            np.testing.assert_allclose(left['intensity'], right['intensity'], atol=1e-6, rtol=1e-5)


def factorial_contrasts(individual_shared, individual_independent, mean_shared, mean_independent):
    individual_gain = individual_independent - individual_shared
    mean_gain = mean_independent - mean_shared
    return {'sharing_with_class_means': mean_gain, 'change_in_sharing_gain': mean_gain - individual_gain}


def compare_cohort(runs, reference_prefix, control_prefix, independent_prefix, resamples, factorial=False):
    reference_paths = [runs / f'{reference_prefix}{seed}' for seed in SEEDS]
    template = read_predictions(reference_paths[0])
    references = load_matched(reference_paths, template)
    reference_meta = load_metadata(reference_paths)
    evaluations = [(REFERENCES, references, reference_meta)]
    control_groups = [(CONTROLS, control_prefix)]
    if factorial:
        control_groups.append(({'independent_class_mean': 'independent_class_mean'}, independent_prefix))
    for arms, prefix in control_groups:
        paths = [runs / f'{prefix}{seed}' for seed in SEEDS]
        trials, trial_meta = load_matched(paths, template), load_metadata(paths)
        verify_pairs(references, trials, reference_meta, trial_meta)
        evaluations.append((arms, trials, trial_meta))

    labels, scores, groups = (template[key] for key in ('y', 'sentiment', 'groups'))
    videos, membership = np.unique(groups, return_inverse=True)
    methods, losses = {}, {}
    for arms, packets, metadata in evaluations:
        for name, arm in arms.items():
            summary, rps, mae = summarize_run(packets, metadata, arm, labels, scores)
            summary['accuracy'] = float(seed_clip_accuracy(packets, arm, labels).mean())
            methods[name] = summary
            losses[name] = (rps, mae)

    shared_rps, shared_mae = losses['shared']
    rps_effects = {name: paired_reduction(losses[name][0] - shared_rps, membership, resamples,
                                          len(COHORTS) * len(CONTROLS))
                   for name in CONTROLS}
    mae_effect = paired_reduction(losses['class_mean_intensity'][1] - shared_mae, membership, resamples,
                                  len(COHORTS))
    result = dict(n_clips=len(labels), n_videos=len(videos), methods=methods,
                  shared_rps_reductions=rps_effects, shared_mae_reduction_vs_class_mean=mae_effect)
    if factorial:
        contrasts = factorial_contrasts(shared_rps, losses['independent'][0],
                                        losses['class_mean_intensity'][0], losses['independent_class_mean'][0])
        result['class_mean_factorial'] = {
            name: paired_reduction(values, membership, resamples, len(COHORTS) * len(contrasts))
            for name, values in contrasts.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--factorial', action='store_true', help='Include independent class mean supervision.')
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Use at least one bootstrap resample.')
    cohorts = {name: compare_cohort(args.runs, *prefixes, args.resamples, factorial=args.factorial)
               for name, prefixes in COHORTS.items()}
    summary = dict(seeds=list(SEEDS), resamples=args.resamples, cohorts=cohorts)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, cohort in cohorts.items():
        effects = {**cohort['shared_rps_reductions'], **cohort.get('class_mean_factorial', {}),
                   'class_mean_intensity (MAE)': cohort['shared_mae_reduction_vs_class_mean']}
        for comparison, estimate in effects.items():
            lo, hi = estimate['ci95_familywise']
            print(f'{name} {comparison}: {estimate["reduction"]:+.6f} [{lo:+.6f}, {hi:+.6f}]')


if __name__ == '__main__':
    main()
