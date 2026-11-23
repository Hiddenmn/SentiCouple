"""Compare neutral probability factorization under matched supervision and masks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from ordinal_controls import COHORTS, paired_reduction, verify_pairs
from polarity_pool import coordinates, outcomes
from score_anchor_effects import SEEDS
from transfer_moud import check_configuration, matched_input, replay_seed


ROLES = ('softmax_shared', 'softmax_independent', 'factorized_shared',
         'factorized_independent', 'gradient_only_shared')
IN_DOMAIN = tuple(f'{source}_{target}' for source in ('sims_v2', 'omg')
                  for target in ('individual', 'class_mean'))
TRANSFER = ('moud', 'meld')
RPS_CONTRASTS = ('factorization', 'factorization_independent', 'sharing_interaction',
                 'sharing_factorized', 'versus_gradient_only', 'gradient_only')


def readout_arms(class_mean):
    suffix = '_class_mean' if class_mean else ''
    shared, independent = f'intensity{suffix}', f'independent{suffix or "_intensity"}'
    return dict(zip(ROLES, (shared, independent, f'{shared}_factorized',
                           f'independent{suffix}_factorized', f'{shared}_gradient_decoupled')))


def paths(root, prefix):
    return [root / f'{prefix}{seed}' for seed in SEEDS]


def matched_in_domain(root, baseline_prefix, factor_prefix, class_mean_prefixes=()):
    shared_paths = paths(root, baseline_prefix)
    reference = read_predictions(shared_paths[0])
    baseline = load_matched(shared_paths, reference)
    baseline_meta = load_metadata(shared_paths)
    factor_paths = paths(root, factor_prefix)
    candidate = load_matched(factor_paths, reference)
    candidate_meta = load_metadata(factor_paths)
    verify_pairs(baseline, candidate, baseline_meta, candidate_meta)
    sources = {}
    for prefix in class_mean_prefixes:
        coarse_paths = paths(root, prefix)
        coarse, metadata = load_matched(coarse_paths, reference), load_metadata(coarse_paths)
        verify_pairs(baseline, coarse, baseline_meta, metadata)
        sources[prefix] = coarse, metadata
    regimes = {}
    anchor = np.stack([packet['anchor'] for packet in baseline])
    granularities = ('individual', 'class_mean') if class_mean_prefixes else ('individual',)
    for granularity in granularities:
        class_mean = granularity == 'class_mean'
        arms = readout_arms(class_mean)
        if class_mean:
            shared, independent = [sources[prefix] for prefix in class_mean_prefixes]
        else:
            shared = independent = (baseline, baseline_meta)
        probabilities = {'anchor': anchor}
        scores, parameters = {}, {}
        trial = candidate, candidate_meta
        for (role, arm), (forecasts, metadata) in zip(arms.items(), (shared, independent, trial, trial, trial)):
            probabilities[role] = np.stack([packet[arm] for packet in forecasts])
            scores[role] = np.stack([packet[f'{arm}_score'] for packet in forecasts])
            parameters[role] = [run['selection'][arm]['parameters'] for run in metadata]
            np.testing.assert_array_equal(probabilities[role][:, -1], anchor[:, -1])
        for left, right in (('softmax_shared', 'factorized_shared'),
                            ('softmax_independent', 'factorized_independent'),
                            ('softmax_shared', 'gradient_only_shared')):
            np.testing.assert_array_equal(parameters[left], parameters[right])
        regimes[granularity] = probabilities, scores
    return reference, regimes


def describe_case(record, probabilities, scores, balanced=False):
    labels, groups = record['y'], record['groups']
    names, membership = np.unique(groups, return_inverse=True)
    labels_by_group = np.stack([np.bincount(labels[membership == i], minlength=3)
                                for i in range(len(names))])
    counts = labels_by_group.sum(0)
    if balanced:
        np.testing.assert_array_equal(labels_by_group, np.broadcast_to([32, 4, 8], labels_by_group.shape))
        weights = len(labels) / (3 * counts[labels])
    else:
        weights = 1.

    means, risks = {}, {}
    for name, probability in probabilities.items():
        gap = probability[:, 1:3]
        risks[name] = ranked_probability_score(gap, labels).mean(axis=1) * weights
        components = ((coordinates(gap) - outcomes(labels))**2 / 4).mean(axis=1)
        signed, neutral = (components * np.asarray(weights)[..., None]).mean(axis=(0, 1))
        np.testing.assert_allclose(signed + neutral, risks[name].mean(), atol=3e-7)
        means[name] = dict(**rps_summary(gap, labels),
                           signed_error=float(signed), neutral_probability_error=float(neutral))

    def risk_reduction(baseline, candidate):
        return risks[baseline] - risks[candidate]

    primary = risk_reduction('softmax_shared', 'factorized_shared')
    independent = risk_reduction('softmax_independent', 'factorized_independent')
    gains = dict(factorization=primary, factorization_independent=independent,
                 sharing_interaction=primary - independent,
                 sharing_factorized=risk_reduction('factorized_independent', 'factorized_shared'),
                 versus_gradient_only=risk_reduction('gradient_only_shared', 'factorized_shared'),
                 gradient_only=risk_reduction('softmax_shared', 'gradient_only_shared'))
    if scores:
        errors = {name: np.abs(values[:, 1:3] - record['sentiment']).mean(axis=1) for name, values in scores.items()}
        for name, values in errors.items():
            means[name]['mae'] = float(values.mean())
        gains['mae'] = errors['softmax_shared'] - errors['factorized_shared']
    else:
        gains['versus_text'] = risk_reduction('anchor', 'factorized_shared')
    description = dict(clips=len(labels), groups=len(names), class_counts=counts.tolist(),
                       metric='balanced_rps' if balanced else 'rps', methods=means)
    return description, dict(membership=membership, **gains)


def actor_case(runs):
    from prepare_ravdess import ACTOR_FOLDS

    folds = [matched_in_domain(runs, f'ravdess_fold{fold}_seed', f'ravdess_neutral_readout_fold{fold}_seed')
             for fold in range(len(ACTOR_FOLDS))]
    records, forecasts = zip(*folds)
    for record, actors in zip(records, ACTOR_FOLDS):
        if set(record['groups']) != {f'actor{actor:02d}' for actor in actors}:
            raise ValueError('The RAVDESS actor groups must match the published fold construction.')
    record = {key: np.concatenate([fold[key] for fold in records])
              for key in ('ids', 'groups', 'y', 'sentiment')}
    if len(record['ids']) != 1056 or len(np.unique(record['ids'])) != 1056:
        raise ValueError('Expected one held-out forecast for each RAVDESS clip.')
    probability_blocks, score_blocks = zip(*(fold['individual'] for fold in forecasts))
    probabilities = {name: np.concatenate([fold[name] for fold in probability_blocks], axis=2)
                     for name in probability_blocks[0]}
    scores = {name: np.concatenate([fold[name] for fold in score_blocks], axis=2) for name in score_blocks[0]}
    return describe_case(record, probabilities, scores, balanced=True)


def target_case(args, name, target_path, expected):
    reference, shared, independent = COHORTS['omg_emotion']
    groups = [paths(args.runs, prefix) for prefix in (reference, shared, independent, 'omg_neutral_readout_seed')]
    config = check_configuration(*(load_metadata(path_list) for path_list in groups))
    device = torch.device(args.device)
    source, source_ids, source_index, target, target_index = matched_input(
        args.omg_data or Path(config['data']), target_path, config, device)
    if (len(target.ids), len(np.unique(target.groups))) != expected:
        raise ValueError(f'The {name} packet does not contain the expected held-out cohort.')
    arms = readout_arms(class_mean=True)
    forecasts = []
    for original, softmax_shared, softmax_independent, factorized in zip(*groups):
        comparisons = [(softmax_shared, (arms['softmax_shared'],)),
                       (softmax_independent, (arms['softmax_independent'],)),
                       (factorized, tuple(arms[role] for role in ROLES[2:]))]
        forecasts.append(replay_seed(original, comparisons, config, source, source_ids, source_index, target_index))
    probabilities = {role: np.stack([seed[arm] for seed in forecasts])
                     for role, arm in dict(anchor='anchor', **arms).items()}
    metadata, values = describe_case(dict(y=target.label, groups=target.groups), probabilities, None)
    with target_path.open('rb') as stream:
        metadata['feature_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(f'Replayed all source fits on {name}.', flush=True)
    return metadata, values


def collect(args):
    evidence, description = {}, {}

    def add(case, result):
        metadata, arrays = result
        description[case] = metadata
        evidence.update({f'{case}__{key}': value for key, value in arrays.items()})

    for tag, corpus in (('sims_v2', 'ch_sims_v2'), ('omg', 'omg_emotion')):
        primary, shared_mean, independent_mean = COHORTS[corpus]
        record, regimes = matched_in_domain(args.runs, primary, f'{tag}_neutral_readout_seed',
                                           (shared_mean, independent_mean))
        for granularity, (probabilities, scores) in regimes.items():
            add(f'{tag}_{granularity}', describe_case(record, probabilities, scores))
    add('ravdess', actor_case(args.runs))
    for name, expected in (('moud', (498, 79)), ('meld', (2610, 143))):
        add(name, target_case(args, name, getattr(args, name), expected))
    return dict(description=np.asarray(json.dumps(description)), **evidence)


def estimate(evidence, resamples):
    cases = json.loads(evidence['description'].item())
    for name, case in cases.items():
        membership = evidence[f'{name}__membership']
        contrasts = {}
        families = {key: len(IN_DOMAIN) + len(TRANSFER) for key in RPS_CONTRASTS}
        if name in IN_DOMAIN:
            families['mae'] = len(IN_DOMAIN)
        elif name in TRANSFER:
            families['versus_text'] = len(TRANSFER)
        else:
            families = dict(factorization=1, mae=len(RPS_CONTRASTS),
                            **{key: len(RPS_CONTRASTS) for key in RPS_CONTRASTS[1:]})
        for key, family in families.items():
            values = evidence[f'{name}__{key}']
            if values.shape != (len(SEEDS), case['clips']):
                raise ValueError(f'Wrong number of predictions for {name}/{key}.')
            contrasts[key] = paired_reduction(values, membership, resamples, family)
        case['contrasts'] = contrasts
    return dict(seeds=SEEDS, resamples=resamples, cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--omg-data', type=Path)
    parser.add_argument('--moud', type=Path)
    parser.add_argument('--meld', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose a positive number of bootstrap draws.')
    if args.evidence:
        if any((args.runs, args.moud, args.meld, args.omg_data, args.write_evidence)):
            parser.error('An existing evidence packet is sufficient for the analysis.')
        with np.load(args.evidence, allow_pickle=False) as archive:
            evidence = dict(archive)
    else:
        if not all((args.runs, args.moud, args.meld)):
            parser.error('Provide the fit directories and both frozen-transfer feature packets.')
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        evidence = collect(args)
    report = estimate(evidence, args.resamples)
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **evidence)
        temporary.replace(args.write_evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, case in report['cases'].items():
        effect = case['contrasts']['factorization']
        print(f'{name:20} {case["metric"]:12} {effect["reduction"]:+.6f} {effect["ci95_familywise"]}')


if __name__ == '__main__':
    main()
