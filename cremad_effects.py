"""Paired, class-balanced inference on the held-out CREMA-D actors."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from score_anchor_effects import SEEDS


HEADS = ('anchor', 'fusion', 'intensity', 'independent_intensity', 'intensity_shuffled')
COMPARATORS = ('independent_intensity', 'anchor', 'fusion', 'intensity_shuffled')
POLARITIES = ('negative', 'neutral', 'positive')


def collect_runs(runs, cohort):
    run_paths = [runs / f'cremad_heads_seed{seed}' for seed in SEEDS]
    example = read_predictions(run_paths[0])
    packets, metadata = load_matched(run_paths, example), load_metadata(run_paths)
    expected = cohort['splits']['test']
    np.testing.assert_array_equal(np.unique(example['groups']), ['actor' + actor for actor in expected['actors']])
    np.testing.assert_array_equal(np.bincount(example['y'], minlength=3), expected['class_counts'])
    if len(example['ids']) != expected['clips']:
        raise ValueError('Evaluation clips differ from the prepared actor partition.')
    reference = metadata[0]['config']
    expected_splits = {role: dict(n=summary['clips'], videos=len(summary['actors']),
                                   labels=summary['class_counts'])
                       for role, summary in zip(('fit', 'tune', 'eval'),
                                                (cohort['splits'][part] for part in ('train', 'valid', 'test')),
                                                strict=True)}
    for seed, packet, run in zip(SEEDS, packets, metadata, strict=True):
        config = run['config']
        for field in config.keys() | reference.keys():
            if field not in ('seed', 'output') and config.get(field) != reference.get(field):
                raise ValueError(f'Seed {seed} does not match the reference setting for {field}.')
        if config['seed'] != seed or config['neutral_margin'] != cohort['neutral_margin']:
            raise ValueError('Seed or neutral margin differs from the planned experiment.')
        if run['splits'] != expected_splits:
            raise ValueError(f'Seed {seed} has a different actor partition summary.')
        for head in HEADS[1:]:
            np.testing.assert_array_equal(packet[head][-1], packet['anchor'][-1])
    evidence = {key: example[key] for key in ('ids', 'groups', 'y', 'sentiment')}
    for head in HEADS:
        evidence[head] = np.stack([packet[head][1:3] for packet in packets])
        evidence[f'{head}_score'] = np.stack([packet[f'{head}_score'][1:3] for packet in packets])
    evidence['seeds'] = np.asarray(SEEDS)
    evidence['cohort'] = np.asarray(json.dumps(cohort))
    evidence['fusion_parameters'] = np.asarray(json.dumps({head: metadata[0]['selection'][head]['parameters']
                                                           for head in HEADS[1:]}))
    return evidence


def balanced_reduction(values, label, actor, seed_draws, actor_draws, family_size, *, include_polarity=False):
    n_actors = actor_draws.shape[1]
    block = 3 * actor + label
    counts = np.bincount(block, minlength=n_actors * 3).reshape(n_actors, 3)
    totals = np.stack([np.bincount(block, weights=row, minlength=n_actors * 3) for row in values])
    actor_weighted = (actor_draws @ totals.T.reshape(n_actors, -1)).reshape(-1, 3, len(values))
    sampled_counts = actor_draws @ counts
    if (sampled_counts == 0).any():
        raise ValueError('A sampled set of actors lacks one polarity.')
    by_class = (actor_weighted * seed_draws[:, None, :]).sum(-1) / (sampled_counts * len(values))
    statistics = by_class.mean(-1)
    estimated_by_class = np.stack([values[:, label == category].mean(-1) for category in range(3)])
    per_seed = estimated_by_class.mean(0)
    tail = .025 / family_size
    result = dict(reduction=float(per_seed.mean()), by_seed=per_seed.tolist(),
                  ci95_familywise=np.quantile(statistics, [tail, 1 - tail]).tolist())
    if include_polarity:
        tail = .025 / len(POLARITIES)
        intervals = np.quantile(by_class, [tail, 1 - tail], axis=0).T
        result['by_polarity_exploratory'] = {
            name: dict(reduction=float(difference), ci95_familywise=bounds.tolist())
            for name, difference, bounds in zip(POLARITIES, estimated_by_class.mean(-1), intervals, strict=True)
        }
    return result


def summarize(evidence, count):
    y, groups, score = (evidence[key] for key in ('y', 'groups', 'sentiment'))
    actors, actor_idx = np.unique(groups, return_inverse=True)
    seeds = evidence['seeds']
    np.testing.assert_array_equal(seeds, SEEDS)
    rng = np.random.default_rng(29)
    seed_draws = rng.multinomial(len(seeds), np.repeat(1 / len(seeds), len(seeds)), count)
    actor_draws = rng.multinomial(len(actors), np.repeat(1 / len(actors), len(actors)), count)
    classes = [y == k for k in range(3)]
    results, rps, mae, error_axes = {}, {}, {}, {}
    for head in HEADS:
        posterior = evidence[head]
        loss = ranked_probability_score(posterior, y)
        rps[head] = loss.mean(axis=1)
        mae[head] = np.abs(evidence[f'{head}_score'] - score).mean(axis=1)
        error_axes[head] = ((coordinates(posterior) - outcomes(y)) ** 2).mean(axis=1) / 4
        by_class = [float(rps[head][:, mask].mean()) for mask in classes]
        by_gap_and_class = np.stack([loss[..., mask].mean(axis=(0, 2)) for mask in classes])
        accuracy = (posterior.argmax(axis=-1) == y).mean(axis=1)
        results[head] = dict(rps=float(loss.mean()), balanced_rps=float(np.mean(by_class)),
                             mae=float(mae[head].mean()), rps_by_class=by_class,
                             balanced_rps_by_gap=by_gap_and_class.mean(axis=0).tolist(),
                             balanced_accuracy=float(np.mean([accuracy[:, mask].mean() for mask in classes])))
    paired = {ref: balanced_reduction(rps[ref] - rps['intensity'], y, actor_idx, seed_draws, actor_draws,
                                      1 if ref == 'independent_intensity' else len(COMPARATORS) - 1,
                                      include_polarity=ref == 'independent_intensity')
              for ref in COMPARATORS}
    delta = error_axes['independent_intensity'] - error_axes['intensity']
    axis_by_class = np.stack([delta[:, mask].mean(axis=(0, 1)) for mask in classes])
    axis_reductions = axis_by_class.mean(axis=0)
    np.testing.assert_allclose(axis_reductions.sum(), paired['independent_intensity']['reduction'], atol=2e-8)
    return dict(seeds=seeds.tolist(), resamples=count, n_actors=len(actors), n_clips=len(y),
                cohort=json.loads(evidence['cohort'].item()),
                fusion_parameters=json.loads(evidence['fusion_parameters'].item()),
                methods=results, balanced_rps_reductions=paired,
                calibrated_mae_reduction=paired_reduction(mae['independent_intensity'] - mae['intensity'],
                                                          actor_idx, count, 1),
                contributions_by_polarity={name: axis.tolist() for name, axis in
                                           zip(POLARITIES, axis_by_class, strict=True)},
                signed_reduction=float(axis_reductions[0]), neutral_reduction=float(axis_reductions[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and supply --cohort with --runs.')
    if args.runs:
        evidence = collect_runs(args.runs, json.loads(args.cohort.read_text(encoding='utf-8')))
    else:
        with np.load(args.evidence, allow_pickle=False) as archive:
            evidence = dict(archive)
    summary = summarize(evidence, args.resamples)
    if args.write_evidence:
        np.savez_compressed(args.write_evidence, **evidence)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for ref, value in summary['balanced_rps_reductions'].items():
        a, b = value['ci95_familywise']
        print(f'{ref}: {value["reduction"]:+.6f} [{a:+.6f}, {b:+.6f}]')
    print('MAE comparison:', summary['calibrated_mae_reduction'])


if __name__ == '__main__':
    main()
