"""Paired comparisons of class-conditional and direct intensity supervision."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import grouped_seed_video_interval, ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from coupling import logit_coordinates, rps_summary
from ordinal_controls import COHORTS, SEEDS, paired_reduction, verify_pairs
from polarity_pool import coordinates, outcomes


TILTED = ('conditional_shared', 'conditional_independent', 'conditional_shuffled')
COMPARATORS = ('intensity', 'conditional_independent', 'conditional_shuffled')
TRIAL_PREFIXES = {'ch_sims_v2': 'sims_v2', 'omg_emotion': 'omg'}


def collect_evidence(root):
    arrays, metadata = {}, {}
    for cohort, (reference_prefix, control_prefix, _) in COHORTS.items():
        trial_prefix = TRIAL_PREFIXES[cohort]
        groups = ((reference_prefix, ('intensity', 'fusion')),
                  (control_prefix, ('intensity_class_mean',)),
                  (f'{trial_prefix}_conditional_intensity_seed', TILTED))
        reference_paths = [root / f'{reference_prefix}{seed}' for seed in SEEDS]
        example = read_predictions(reference_paths[0])
        references = load_matched(reference_paths, example)
        reference_meta = load_metadata(reference_paths)
        prefix = f'{cohort}__'
        for key in ('ids', 'groups', 'y', 'sentiment'):
            arrays[prefix + key] = example[key]
        signed_anchor = logit_coordinates(np.stack([packet['anchor'][1:3] for packet in references]))[0]
        meta = dict(splits=reference_meta[0]['splits'], models={})
        for run_prefix, arms in groups:
            paths = [root / f'{run_prefix}{seed}' for seed in SEEDS]
            packets, runs = load_matched(paths, example), load_metadata(paths)
            verify_pairs(references, packets, reference_meta, runs)
            if arms == TILTED:
                for run, reference in zip(runs, reference_meta):
                    shared_size = reference['selection']['intensity']['parameters']
                    independent_size = reference['selection']['independent_intensity']['parameters']
                    for arm, expected in zip(arms, (shared_size, independent_size, shared_size)):
                        if run['selection'][arm]['parameters'] != expected:
                            raise ValueError(f'{arm} must match the corresponding original head.')
            for arm in arms:
                selections = [run['selection'][arm] for run in runs]
                meta['models'][arm] = dict(fusion_parameters=[item['parameters'] for item in selections],
                                           selected_epochs=[item['epoch'] for item in selections],
                                           score_gain=[item['score_gain'] for item in selections])
                for key in (arm, f'{arm}_score', f'{arm}_uncalibrated_score', f'{arm}_class_mean_score'):
                    if key in packets[0]:
                        arrays[prefix + key] = np.stack([packet[key][1:3] for packet in packets])
                if arm in ('conditional_shared', 'conditional_shuffled'):
                    signed = logit_coordinates(arrays[prefix + arm])[0]
                    arrays[f'{prefix}{arm}_signed_shift'] = signed - signed_anchor
                for packet, run in zip(packets, runs):
                    np.testing.assert_array_equal(packet[arm][-1], packet['anchor'][-1])
                    if arm in TILTED:
                        anchor = run['selection']['anchor']
                        expected = packet[arm] @ np.asarray(anchor['class_centers']) * anchor['intensity_scale']
                        np.testing.assert_allclose(packet[f'{arm}_class_mean_score'], expected, atol=2e-6)
                        np.testing.assert_allclose(packet[f'{arm}_score'][-1], expected[-1], atol=2e-6)
        published = json.loads((root / f'{trial_prefix}_published_seeds.json').read_text(encoding='utf-8'))
        if (published['seeds'] != list(SEEDS) or published['n_clips'] != len(example['y'])
                or published['n_videos'] != len(np.unique(example['groups']))):
            raise ValueError(f'The published comparison does not match {cohort}.')
        meta['published_rps'] = {key: published['methods'][key]['rps']
                                 for key in ('tfr_joint', 'lnln_joint', 'rmnet_core')}
        metadata[cohort] = meta
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    arrays['seeds'] = np.asarray(SEEDS)
    return arrays


def within_video_pairs(label, intensity, membership):
    left, right = [], []
    for video in np.unique(membership):
        for category in np.unique(label[membership == video]):
            clips = np.flatnonzero((membership == video) & (label == category))
            i, j = np.triu_indices(len(clips), k=1)
            distinct = intensity[clips[i]] != intensity[clips[j]]
            left.extend(clips[i[distinct]])
            right.extend(clips[j[distinct]])
    if not left:
        raise ValueError('The within-video diagnostic needs unequal scores within the same polarity.')
    left, right = np.asarray(left), np.asarray(right)
    return left, right


def pair_concordance(shifts, left, right, order):
    difference = shifts[..., left] - shifts[..., right]
    direction = np.where(np.abs(difference) <= 1e-6, 0, np.sign(difference))
    return .5 + .5 * order * direction


def order_group_totals(shared, shuffled, label, intensity, membership):
    left, right = within_video_pairs(label, intensity, membership)
    order = np.sign(intensity[left] - intensity[right])
    videos, pair_group = np.unique(membership[left], return_inverse=True)
    genuine = pair_concordance(shared, left, right, order).mean(axis=1)
    permutation = pair_concordance(shuffled, left, right, order).mean(axis=1)

    def by_video(values):
        return [np.bincount(pair_group, weights=row, minlength=len(videos)).tolist() for row in values]

    return dict(pair_counts=np.bincount(pair_group).tolist(),
                original_sums=by_video(genuine), shuffled_sums=by_video(permutation))


def summarize_order_groups(record, resamples, *, equal_video_weight=False):
    counts = np.asarray(record['pair_counts'], dtype=np.int64)
    original = np.asarray(record['original_sums'], dtype=np.float64)
    shuffled = np.asarray(record['shuffled_sums'], dtype=np.float64)
    if original.shape != (len(SEEDS), counts.size) or shuffled.shape != original.shape or (counts <= 0).any():
        raise ValueError('Expected matched paired-fit totals for each source video.')
    n_pairs = int(counts.sum())
    weights = counts
    if equal_video_weight:
        original = original / counts
        shuffled = shuffled / counts
        weights = np.ones_like(counts)
    total_weight = int(weights.sum())
    tail = .025 / (len(COHORTS) * 2)

    def estimate(totals):
        per_seed = totals.sum(axis=1) / total_weight
        return dict(reduction=float(per_seed.mean()), by_seed=per_seed.tolist(),
                    ci95_familywise=grouped_seed_video_interval(totals, weights, resamples,
                                                                quantiles=(tail, 1 - tail)))

    return dict(pairs=n_pairs, source_videos=counts.size,
                concordance_shared=float(original.sum(axis=1).mean() / total_weight),
                concordance_shuffled=float(shuffled.sum(axis=1).mean() / total_weight),
                gain_above_half=estimate(original - .5 * weights),
                gain_above_shuffle=estimate(original - shuffled))


def within_video_order(shared, shuffled, label, intensity, membership, resamples):
    groups = order_group_totals(shared, shuffled, label, intensity, membership)
    return summarize_order_groups(groups, resamples)


def summarize_cohort(arrays, cohort, meta, resamples):
    def series(key):
        return arrays[f'{cohort}__{key}']

    y, intensity = series('y'), series('sentiment')
    _, membership = np.unique(series('groups'), return_inverse=True)
    n_cohorts = len(COHORTS)
    methods, rps, mae = {}, {}, {}
    for name, selection in meta['models'].items():
        posterior = series(name)
        rps[name] = ranked_probability_score(posterior, y).mean(axis=1)
        mae[name] = np.abs(series(f'{name}_score') - intensity).mean(axis=1)
        methods[name] = dict(**rps_summary(posterior, y), mae=float(mae[name].mean()), **selection)
        if f'{cohort}__{name}_uncalibrated_score' in arrays:
            uncalibrated = np.abs(series(f'{name}_uncalibrated_score') - intensity)
            methods[name]['raw_mae'] = float(uncalibrated.mean())
    shared = 'conditional_shared'
    primary = {name: paired_reduction(rps[name] - rps[shared], membership, resamples,
                                      n_cohorts * len(COMPARATORS)) for name in COMPARATORS}
    secondary = {name: paired_reduction(mae[name] - mae[shared], membership, resamples,
                                        n_cohorts * len(COMPARATORS)) for name in COMPARATORS}
    within_class = {}
    for name in ('conditional_shared', 'conditional_independent'):
        base_error = np.abs(series(f'{name}_class_mean_score') - intensity).mean(axis=1)
        within_class[name] = paired_reduction(base_error - mae[name], membership, resamples, n_cohorts * 2)
        within_class[name]['class_mean_mae'] = float(base_error.mean())
    encoded_y = outcomes(y)
    decomposition = ((coordinates(series('intensity')) - encoded_y) ** 2
                     - (coordinates(series(shared)) - encoded_y) ** 2).mean(axis=(0, 1, 2)) / 4
    np.testing.assert_allclose(decomposition.sum(), primary['intensity']['reduction'], atol=2e-8)
    order = within_video_order(series('conditional_shared_signed_shift'), series('conditional_shuffled_signed_shift'),
                               y, intensity, membership, resamples)
    return dict(splits=meta['splits'], methods=methods, published_rps=meta['published_rps'],
                rps_reductions=primary, mae_reductions=secondary, within_class_mae_reductions=within_class,
                rps_coordinates=dict(zip(('signed', 'neutral'), decomposition.tolist())), within_video_order=order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', type=Path)
    source.add_argument('--evidence', type=Path)
    source.add_argument('--grouped-order', type=Path, help='Reproduce concordance from per-video aggregates.')
    parser.add_argument('--equal-video-weight', action='store_true',
                        help='Give every video equal weight in the grouped ordering comparison.')
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', default=20_000, type=int)
    args = parser.parse_args()
    if args.resamples < 1 or (args.write_evidence and not args.runs):
        parser.error('Use a positive resample count and write evidence only from training runs.')
    if args.equal_video_weight and args.grouped_order is None:
        parser.error('Equal video weighting requires --grouped-order.')
    if args.grouped_order is not None:
        grouped = json.loads(args.grouped_order.read_text(encoding='utf-8'))
        summary = {cohort: summarize_order_groups(grouped[cohort], args.resamples,
                                                   equal_video_weight=args.equal_video_weight)
                   for cohort in COHORTS}
        args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return
    if args.runs:
        arrays = collect_evidence(args.runs)
    else:
        with np.load(args.evidence, allow_pickle=False) as npz:
            arrays = dict(npz)
    np.testing.assert_array_equal(arrays['seeds'], SEEDS)
    metadata = json.loads(arrays['metadata'].item())
    cohorts = {name: summarize_cohort(arrays, name, metadata[name], args.resamples) for name in COHORTS}
    summary = dict(seeds=list(SEEDS), resamples=args.resamples, cohorts=cohorts)
    if args.write_evidence:
        with args.write_evidence.with_suffix('.npz.part').open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        args.write_evidence.with_suffix('.npz.part').replace(args.write_evidence)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, report in cohorts.items():
        for family in ('rps_reductions', 'mae_reductions', 'within_class_mae_reductions'):
            for comparison, entry in report[family].items():
                low, high = entry['ci95_familywise']
                print(f'{name} {family} {comparison}: {entry["reduction"]:+.5f} [{low:+.5f}, {high:+.5f}]')
        for comparison in ('gain_above_half', 'gain_above_shuffle'):
            entry = report['within_video_order'][comparison]
            low, high = entry['ci95_familywise']
            print(f'{name} within-video order {comparison}: {entry["reduction"]:+.5f} [{low:+.5f}, {high:+.5f}]')


if __name__ == '__main__':
    main()
