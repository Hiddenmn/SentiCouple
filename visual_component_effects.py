"""Estimate the paired contributions of visual depth, mean pooling and maximum pooling."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from ordinal_controls import COHORTS, paired_reduction, verify_pairs
from polarity_pool import coordinates, outcomes
from prepare_ravdess import ACTOR_FOLDS
from published_seed_effects import BASELINES, count_anchor, run_paths, verify_training
from score_anchor_effects import SEEDS
from transfer_moud import check_configuration, matched_input, replay_seed


COHORT_SIZES = {'sims_v2': (2122, 85), 'omg': (834, 65), 'ravdess': (1056, 24),
                'moud': (498, 79), 'meld': (2610, 143)}
ORIGINAL = {'deep': 'intensity', 'deep_independent': 'independent_intensity'}
PROJECTED = {'projection': 'intensity_projection', 'independent': 'independent_projection',
             'class_only': 'fusion_projection'}
POOLING = {'deep_mean_only': 'intensity_mean_pool', 'deep_max_only': 'intensity_max_pool',
           'projection_mean_only': 'intensity_projection_mean', 'projection_max_only': 'intensity_projection_max'}
TRANSFER = {'deep': 'intensity_class_mean', 'deep_independent': 'independent_class_mean',
            'projection': 'intensity_class_mean_projection', 'independent': 'independent_class_mean_projection'}
FAMILY_SIZES = dict(depth=5, pooling=8, independent=5, class_only=3, anchor=5,
                    calibrated_mae=3, half_gain=2, benchmarks=3, benchmark_secondary=6)
METADATA_KEYS = ('ids', 'groups', 'y', 'sentiment')


def run_group(root, prefix):
    return [root / f'{prefix}{seed}' for seed in SEEDS]


def matched_models(root, original_prefix, new_prefix, pooling=False):
    original_paths = run_group(root, original_prefix)
    new_paths = run_group(root, new_prefix)
    reference = read_predictions(original_paths[0])
    original, new = (load_matched(paths, reference) for paths in (original_paths, new_paths))
    original_meta, new_meta = (load_metadata(paths) for paths in (original_paths, new_paths))
    verify_pairs(original, new, original_meta, new_meta)
    for seed, left, right in zip(SEEDS, original_meta, new_meta):
        width = left['config']['width']
        expected = 6 * width * (width + 1)
        full = left['selection']['intensity']['parameters']
        projected = right['selection']['intensity_projection']['parameters']
        if full - projected != expected:
            raise ValueError(f'Seed {seed} must remove exactly the two visual blocks.')
        if pooling:
            for role, arm in POOLING.items():
                count = full if role.startswith('deep') else projected
                if right['selection'][arm]['parameters'] != count:
                    raise ValueError(f'The {arm} pooling control must match its reference parameter count.')
    probabilities = {'anchor': np.stack([packet['anchor'] for packet in original])}
    scores = {'anchor': np.stack([packet['anchor_score'] for packet in original])}
    anchor_params = count_anchor(original_paths[0])
    parameters = {'anchor': dict(fusion=0, total=anchor_params)}
    families = [(ORIGINAL, original, original_meta), (PROJECTED, new, new_meta)]
    if pooling:
        families.append((POOLING, new, new_meta))
    for arms, packets, metadata in families:
        for name, arm in arms.items():
            probabilities[name] = np.stack([packet[arm] for packet in packets])
            scores[name] = np.stack([packet[f'{arm}_score'] for packet in packets])
            np.testing.assert_array_equal(probabilities[name][:, -1], probabilities['anchor'][:, -1])
            count = metadata[0]['selection'][arm]['parameters']
            if any(run['selection'][arm]['parameters'] != count for run in metadata):
                raise ValueError(f'Parameter counts differ across seeds for {arm}.')
            parameters[name] = dict(fusion=count, total=count + anchor_params)
    return reference, probabilities, scores, parameters


def actor_models(root):
    folds = [matched_models(root, f'ravdess_fold{fold}_seed', f'ravdess_visual_components_fold{fold}_seed')
             for fold in range(len(ACTOR_FOLDS))]
    records, probabilities, scores, parameters = zip(*folds)
    for record, actors, counts in zip(records, ACTOR_FOLDS, parameters):
        if set(record['groups']) != {f'actor{actor:02d}' for actor in actors}:
            raise ValueError('RAVDESS actor membership must match the original folds.')
        if counts != parameters[0]:
            raise ValueError('RAVDESS folds must use the same network sizes.')
    record = {key: np.concatenate([fold[key] for fold in records]) for key in METADATA_KEYS}
    if len(record['ids']) != len(np.unique(record['ids'])):
        raise ValueError('Each RAVDESS clip must have one held-out prediction per seed.')
    merged_probabilities = {name: np.concatenate([fold[name] for fold in probabilities], axis=2)
                            for name in probabilities[0]}
    merged_scores = {name: np.concatenate([fold[name] for fold in scores], axis=2) for name in scores[0]}
    return record, merged_probabilities, merged_scores, parameters[0]


def add_published(runs, record, probabilities, scores, parameters):
    paths = run_paths(runs, SEEDS)
    for name in BASELINES:
        metadata = load_metadata(paths[name])
        verify_training(name, metadata, SEEDS)
        packets = load_matched(paths[name], record)
        probabilities[name] = np.stack([packet['posterior'] for packet in packets])
        scores[name] = np.stack([packet['score'] for packet in packets])
        parameters[name] = dict(total=metadata[0]['trainable_parameters'])


def frozen_models(args, target_path):
    prefixes = (*COHORTS['omg_emotion'], 'omg_visual_components_seed')
    groups = [run_group(args.runs, prefix) for prefix in prefixes]
    config = check_configuration(*(load_metadata(runs) for runs in groups))
    source, source_ids, source_index, target, target_index = matched_input(
        args.omg_data or Path(config['data']), target_path, config, torch.device(args.device))
    forecasts = []
    for anchor, shared, independent, projected in zip(*groups):
        comparisons = [(shared, (TRANSFER['deep'],)), (independent, (TRANSFER['deep_independent'],)),
                       (projected, (TRANSFER['projection'], TRANSFER['independent']))]
        forecasts.append(replay_seed(anchor, comparisons, config, source, source_ids, source_index, target_index))
    probabilities = {name: np.stack([item[arm] for item in forecasts])
                     for name, arm in dict(anchor='anchor', **TRANSFER).items()}
    record = dict(ids=target.ids, groups=target.groups, y=target.label)
    return record, probabilities, {}, {}


def describe(name, record, probabilities, scores, parameters):
    label, groups = (record[key] for key in ('y', 'groups'))
    group_names, membership = np.unique(groups, return_inverse=True)
    n = len(label)
    if (n, len(group_names)) != COHORT_SIZES[name]:
        raise ValueError(f'Unexpected sample or source count for {name}.')
    counts = np.bincount(label, minlength=3)
    balanced = name == 'ravdess'
    weights = 1.
    if balanced:
        labels_by_group = np.stack([np.bincount(label[membership == i], minlength=3)
                                    for i in range(len(group_names))])
        np.testing.assert_array_equal(labels_by_group, np.broadcast_to([32, 4, 8], labels_by_group.shape))
        weights = n / (3 * counts[label])
    methods, risk = {}, {}
    absolute_error = {key: np.abs(score[:, 1:3] - record['sentiment']).mean(axis=1)
                      for key, score in scores.items()}
    correct = {}
    for key, values in probabilities.items():
        gap = values[:, 1:3]
        risk[key] = ranked_probability_score(gap, label).mean(axis=1) * weights
        methods[key] = rps_summary(gap, label)
        correct[key] = (gap.argmax(-1) == label).mean(axis=1)
        components = (np.square(coordinates(gap) - outcomes(label)) / 4).mean(axis=1)
        signed, neutral = (components * np.asarray(weights)[..., None]).mean(axis=(0, 1))
        np.testing.assert_allclose(signed + neutral, risk[key].mean(), atol=2e-7)
        methods[key].update(signed_error=float(signed), neutral_probability_error=float(neutral),
                            **parameters.get(key, {}))
        if key in absolute_error:
            methods[key]['mae'] = float(absolute_error[key].mean())
    depth, projected = risk['deep'], risk['projection']
    effects = dict(depth=depth - projected, independent=risk['independent'] - projected,
                   anchor=risk['anchor'] - projected)
    if 'class_only' in risk:
        effects['class_only'] = risk['class_only'] - projected
        effects['calibrated_mae'] = absolute_error['deep'] - absolute_error['projection']
    if name in ('sims_v2', 'ravdess'):
        effects['half_gain'] = (depth + risk['deep_independent']) / 2 - projected
    if name in ('sims_v2', 'omg'):
        for head in ('deep', 'projection'):
            effects[f'pooling__{head}_add_maximum'] = risk[f'{head}_mean_only'] - risk[head]
            effects[f'pooling__{head}_add_mean'] = risk[f'{head}_max_only'] - risk[head]
    if name == 'sims_v2':
        for key in BASELINES:
            effects[f'benchmarks__{key}'] = risk[key] - projected
            effects[f'benchmark_secondary__{key}_accuracy'] = correct['projection'] - correct[key]
            effects[f'benchmark_secondary__{key}_mae'] = absolute_error[key] - absolute_error['projection']
    summary = dict(clips=n, groups=len(group_names), class_counts=counts.tolist(),
                   metric='balanced_rps' if balanced else 'rps', methods=methods, effects=list(effects))
    return summary, dict(membership=membership, **effects)


def collect(args):
    description, arrays = {}, {}

    def add(name, result):
        summary, contrasts = describe(name, *result)
        description[name] = summary
        arrays.update({f'{name}/{key}': value for key, value in contrasts.items()})

    for name, cohort in (('sims_v2', 'ch_sims_v2'), ('omg', 'omg_emotion')):
        models = matched_models(args.runs, COHORTS[cohort][0], f'{name}_visual_components_seed', pooling=True)
        if name == 'sims_v2':
            add_published(args.runs, *models)
        add(name, models)
    add('ravdess', actor_models(args.runs))
    for name in ('moud', 'meld'):
        packet = getattr(args, name)
        add(name, frozen_models(args, packet))
        with packet.open('rb') as stream:
            description[name]['feature_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        print(f'Replayed source forecasts and evaluated frozen transfer to {name.upper()}.', flush=True)
    return dict(description=np.asarray(json.dumps(description)), seeds=np.asarray(SEEDS), **arrays)


def estimate(data, resamples):
    cases = json.loads(data['description'].item())
    if list(cases) != list(COHORT_SIZES) or data['seeds'].tolist() != list(SEEDS):
        raise ValueError('The evidence must include every declared cohort and seed in protocol order.')
    family_counts = Counter(key.split('__')[0] for summary in cases.values() for key in summary['effects'])
    if family_counts != FAMILY_SIZES:
        raise ValueError('The available paired contrasts differ from the declared interval families.')
    for name, summary in cases.items():
        membership = data[f'{name}/membership']
        if (len(membership), len(np.unique(membership))) != COHORT_SIZES[name]:
            raise ValueError(f'Wrong group membership for {name}.')
        summary['contrasts'] = {}
        for key in summary.pop('effects'):
            values = data[f'{name}/{key}']
            if values.shape != (len(SEEDS), summary['clips']):
                raise ValueError(f'Wrong number of paired losses for {name}/{key}.')
            summary['contrasts'][key] = paired_reduction(values, membership, resamples,
                                                        FAMILY_SIZES[key.split('__')[0]])
    return dict(seeds=SEEDS, resamples=resamples, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', type=Path)
    source.add_argument('--evidence', type=Path)
    parser.add_argument('--omg-data', type=Path)
    parser.add_argument('--moud', type=Path)
    parser.add_argument('--meld', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose at least one bootstrap draw.')
    if args.runs:
        if not args.moud or not args.meld:
            parser.error('Provide both frozen target feature packets.')
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        data = collect(args)
    else:
        if any((args.omg_data, args.moud, args.meld, args.write_evidence)):
            parser.error('An existing evidence packet is sufficient to run the analysis.')
        with np.load(args.evidence, allow_pickle=False) as archive:
            data = dict(archive)
    report = estimate(data, args.resamples)
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(args.write_evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, summary in report['cases'].items():
        value = summary['contrasts']['depth']
        print(f'{name:9} {summary["metric"]:12} {value["reduction"]:+.6f} {value["ci95_familywise"]}')


if __name__ == '__main__':
    main()
