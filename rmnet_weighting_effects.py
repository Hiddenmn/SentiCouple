"""Separate inverse-frequency weighting from the matched RM-Net benchmark difference."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import ranked_probability_score
from compare_benchmarks import load_matched, load_metadata
from cremad_effects import balanced_reduction
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_seed_effects import run_paths, verify_training
from score_anchor_effects import SEEDS


COHORTS = ('sims_v2', 'omg')
COMPACT = {'compact_uniform': 'shared', 'compact_frequency': 'inverse_frequency'}
MODELS = (*COMPACT, 'rmnet_uniform', 'rmnet_frequency')


def collect(runs, cohort):
    original_runs = run_paths(runs, SEEDS, cohort)['rmnet_core']
    uniform_runs = [runs / f'{cohort}_rmnet_uniform_seed{seed}' for seed in SEEDS]
    original, uniform = load_metadata(original_runs), load_metadata(uniform_runs)
    verify_training('rmnet_core', original, SEEDS, cohort)
    for before, after, path in zip(original, uniform, uniform_runs, strict=True):
        config = dict(after['config'])
        if config.pop('rmnet_class_weighting', None) != 'uniform':
            raise ValueError(f'{path.name} must use uniform weights in both class losses.')
        expected = dict(before['config'], output=config['output'])
        if (config != expected or after['upstream_revision'] != before['upstream_revision'] or
                after['trainable_parameters'] != before['trainable_parameters']):
            raise ValueError(f'{path.name} has another change besides the weighting intervention.')
        history = after['tune_rps_by_epoch']
        if after['selection']['epoch'] != int(np.argmin(history)) + 1:
            raise ValueError(f'{path.name} must use the best tuning RPS checkpoint.')
    compact_paths = [runs / Path(row['config']['match_run']).name for row in original]
    class_counts = [row['splits']['fit']['labels'] for row in load_metadata(compact_paths)]
    if any(row != class_counts[0] for row in class_counts) or min(class_counts[0]) < 1:
        raise ValueError('The inference-only comparison requires the common fitting counts.')
    with np.load(runs / f'text_weighted_loss_{cohort}_evidence.npz', allow_pickle=False) as ref:
        np.testing.assert_array_equal(ref['seeds'], SEEDS)
        evidence = {key: ref[key] for key in ('ids', 'groups', 'y', 'sentiment', 'masks', 'seeds', 'cohort')}
        for method, source in dict(COMPACT, rmnet_frequency='rmnet_core').items():
            for suffix in ('', '_score'):
                evidence[method + suffix] = ref[source + suffix]
    predictions = load_matched(uniform_runs, evidence)
    for key, suffix in (('posterior', ''), ('score', '_score')):
        evidence['rmnet_uniform' + suffix] = np.stack([packet[key][1:3] for packet in predictions])
    original_predictions = load_matched(original_runs, evidence)
    np.testing.assert_array_equal(evidence['rmnet_frequency'], [row['posterior'][1:3] for row in original_predictions])
    np.testing.assert_array_equal(evidence['rmnet_frequency_score'], [row['score'][1:3] for row in original_predictions])
    formatted = {name: [row['selection']['epoch'] for row in path]
                 for name, path in (('rmnet_frequency', original), ('rmnet_uniform', uniform))}
    evidence['selected_epochs'] = np.asarray(json.dumps(formatted))
    evidence['fit_class_counts'] = np.asarray(class_counts[0])
    return evidence


def frequency_adjustment(posterior, class_counts, visible):
    scaled = posterior / class_counts
    adjusted = scaled / scaled.sum(-1, keepdims=True)
    return np.where(visible[..., None], adjusted, posterior)


def summarize(evidence, resamples):
    label, group, intensity = (evidence[key] for key in ('y', 'groups', 'sentiment'))
    group_ids, membership = np.unique(group, return_inverse=True)
    seeds = evidence['seeds']
    losses, results = {}, {}
    for method in MODELS:
        posterior = evidence[method].astype(np.float64)
        rps = ranked_probability_score(posterior, label).mean(axis=1)
        signed, neutral = (np.square(coordinates(posterior) - outcomes(label)) / 4.).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(signed + neutral, rps.mean(), atol=5e-8)
        by_polarity = [float(rps[:, label == j].mean()) for j in range(3)]
        results[method] = dict(rps=float(rps.mean()), signed_rps=float(signed), neutral_rps=float(neutral),
                               accuracy=float((posterior.argmax(-1) == label).mean()),
                               mae=float(np.abs(evidence[method + '_score'] - intensity).mean()),
                               rps_by_polarity=by_polarity, balanced_rps=float(np.mean(by_polarity)))
        losses[method] = rps
    compact = losses['compact_uniform'] - losses['compact_frequency']
    rmnet = losses['rmnet_uniform'] - losses['rmnet_frequency']
    weights = dict(compact=compact, rmnet=rmnet, interaction=rmnet - compact)
    matched = {name: losses[f'compact_{name}'] - losses[f'rmnet_{name}'] for name in ('uniform', 'frequency')}
    gaps = {name: paired_reduction(values, membership, resamples, 4) for name, values in matched.items()}
    reported = losses['compact_uniform'] - losses['rmnet_frequency']
    np.testing.assert_allclose(reported, matched['uniform'] + rmnet, atol=1e-13)
    visible = evidence['masks'][1:3, 1].any(-1)
    posthoc = {}
    for name in ('compact', 'rmnet'):
        original = evidence[f'{name}_uniform'].astype(np.float64)
        tilted = frequency_adjustment(original, evidence['fit_class_counts'], visible)
        np.testing.assert_array_equal(tilted[:, ~visible], original[:, ~visible])
        risk = ranked_probability_score(tilted, label).mean(axis=1)
        posthoc[name] = dict(rps=float(risk.mean()),
                             training_gain=paired_reduction(risk - losses[f'{name}_frequency'], membership,
                                                            resamples, 4))
    rng = np.random.default_rng(29)
    seed_draws = rng.multinomial(len(seeds), np.repeat(1 / len(seeds), len(seeds)), resamples)
    group_draws = rng.multinomial(len(group_ids), np.repeat(1 / len(group_ids), len(group_ids)), resamples)
    return dict(cohort=evidence['cohort'].item(), n_videos=len(group_ids), n_clips=len(label),
                seeds=seeds.tolist(), resamples=resamples, methods=results,
                weighting_effects={name: paired_reduction(values, membership, resamples, 6)
                                   for name, values in weights.items()}, matched_gaps=gaps,
                balanced_rps_effects_exploratory={name: balanced_reduction(values, label, membership, seed_draws,
                                                                         group_draws, 6)
                                                  for name, values in weights.items()},
                fit_class_counts=evidence['fit_class_counts'].tolist(),
                inference_only_frequency_exploratory=posthoc,
                published_gap=float(reported.mean()),
                selected_epochs=json.loads(evidence['selected_epochs'].item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', choices=COHORTS)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and supply --cohort with --runs.')
    if args.runs:
        evidence = collect(args.runs, args.cohort)
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    summary = summarize(evidence, args.resamples)
    if args.write_evidence:
        np.savez_compressed(args.write_evidence, **evidence)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, weight in summary['weighting_effects'].items():
        low, high = weight['ci95_familywise']
        print(f'{name}: {weight["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
