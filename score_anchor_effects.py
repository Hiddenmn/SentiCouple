"""Estimate the paired effect of fitting the frozen text anchor to continuous scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions, seed_video_interval
from compare_benchmarks import load_matched, load_metadata, summarize
from coupling import logit_coordinates


SEEDS = (97, 17, 521, 31, 59, 113, 251, 359, 701, 997)
HEADS = {'shared': 'intensity', 'independent': 'independent_intensity'}


def verify_pair(original_meta, continuous_meta, original, continuous, seed):
    old, new = original_meta['config'], continuous_meta['config']
    defaults = {'neutral_margin': 0., 'score_anchor': 'class-mean', 'calibrate_score_gain': False}
    changed = {'output', 'arms', 'score_anchor'}
    for key in (old.keys() | new.keys()) - changed:
        if old.get(key, defaults.get(key)) != new.get(key, defaults.get(key)):
            raise ValueError(f'Seed {seed} has a mismatched {key}.')
    if (old['seed'] != seed or new['seed'] != seed or
            old.get('score_anchor', 'class-mean') != 'class-mean' or new['score_anchor'] != 'continuous'):
        raise ValueError(f'Expected both score anchors for seed {seed}.')
    if original_meta['splits'] != continuous_meta['splits']:
        raise ValueError(f'Seed {seed} must retain the same split counts.')
    np.testing.assert_array_equal(original['anchor'], continuous['anchor'], err_msg=f'anchor, seed {seed}')
    for arm in HEADS.values():
        old_score = original[f'{arm}_score'][-1]
        new_score = continuous[f'{arm}_score'][-1]
        np.testing.assert_allclose(old_score, continuous['class_mean_anchor_score'][-1], atol=2e-6, rtol=3e-6)
        np.testing.assert_allclose(new_score, continuous['anchor_score'][-1], atol=2e-6, rtol=3e-6)


def gap_anchor_error(packets, key, target):
    return np.stack([np.abs(packet[key][1:3] - target).mean(axis=0) for packet in packets])


def report_cohort(original_paths, continuous_paths, seeds, replicates, family_tail):
    reference = read_predictions(original_paths[0])
    original = load_matched(original_paths, reference)
    continuous = load_matched(continuous_paths, reference)
    original_meta, continuous_meta = map(load_metadata, (original_paths, continuous_paths))
    for pair in zip(original_meta, continuous_meta, original, continuous, seeds):
        verify_pair(*pair)
    labels, target, groups = (reference[key] for key in ('y', 'sentiment', 'groups'))
    _, membership = np.unique(groups, return_inverse=True)
    anchor_errors = {kind: gap_anchor_error(continuous, key, target) for kind, key in
                     [('class_mean', 'class_mean_anchor_score'), ('continuous', 'anchor_score')]}
    results = {}
    losses = {}
    for kind, packets in [('class_mean', original), ('continuous', continuous)]:
        results[kind] = {'anchor_mae': float(anchor_errors[kind].mean())}
        losses[kind] = {}
        for name, arm in HEADS.items():
            summary, rps, mae = summarize(packets, arm, f'{arm}_score', labels, target)
            results[kind][name] = summary
            losses[kind][name] = {'rps': rps, 'mae': mae}
        signed = {name: logit_coordinates(np.stack([p[arm][1:3] for p in packets]))[0]
                  for name, arm in HEADS.items()}
        results[kind]['mean_signed_shared_minus_independent'] = float(
            (signed['shared'] - signed['independent']).mean())

    def estimate(effect, adjusted=False):
        interval = seed_video_interval(effect, membership, replicates)
        outcome = dict(difference=float(effect.mean()), ci95_seeds_and_videos=interval,
                       by_seed=effect.mean(axis=-1).tolist())
        if adjusted:
            outcome['ci95_familywise'] = seed_video_interval(
                effect, membership, replicates, quantiles=(family_tail, 1 - family_tail))
        return outcome

    reductions = {'anchor_mae': estimate(anchor_errors['class_mean'] - anchor_errors['continuous'], True)}
    for name in HEADS:
        for metric in ('rps', 'mae'):
            effect = losses['class_mean'][name][metric] - losses['continuous'][name][metric]
            reductions[f'{name}_{metric}'] = estimate(effect, adjusted=name == 'shared')
    interactions = {}
    for metric in ('rps', 'mae'):
        gains = {kind: arms['independent'][metric] - arms['shared'][metric] for kind, arms in losses.items()}
        interactions[metric] = estimate(gains['continuous'] - gains['class_mean'])
    return dict(n_clips=len(labels), n_videos=int(membership.max()) + 1,
                methods=results, reductions=reductions, sharing_interaction=interactions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', action='append', nargs=3, required=True,
                        metavar=('NAME', 'CLASS_MEAN_RUN_PREFIX', 'CONTINUOUS_RUN_PREFIX'))
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 1 or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error('Use positive resampling and at least two distinct paired fits.')
    if len({row[0] for row in args.cohort}) != len(args.cohort):
        parser.error('Cohort names must be distinct.')
    family_size = 3 * len(args.cohort)
    reports = {}
    for name, original, continuous in args.cohort:
        old_runs = [Path(f'{original}{seed}') for seed in args.seeds]
        new_runs = [Path(f'{continuous}{seed}') for seed in args.seeds]
        report = report_cohort(old_runs, new_runs, args.seeds, args.resamples, .025 / family_size)
        reports[name] = report
        for contrast, result in report['reductions'].items():
            interval = result.get('ci95_familywise', result['ci95_seeds_and_videos'])
            print(f'{name:10s} {contrast:18s} {result["difference"]:+.6f} [{interval[0]:+.6f}, {interval[1]:+.6f}]')
    result = dict(cohorts=reports, seeds=list(args.seeds), resamples=args.resamples, family_size=family_size)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
