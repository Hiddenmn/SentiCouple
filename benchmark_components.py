"""Attribute matched forecast gaps to no-vision contrasts and their visual changes."""

import argparse
import json
from math import factorial
from pathlib import Path

import numpy as np

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_seed_effects import run_paths, verify_training
from score_anchor_effects import SEEDS
from text_weighted_loss import RUN_NAME


REFERENCES = ('rmnet_uniform', 'lnln_joint')
COMPONENTS = ('signed_no_vision', 'neutral_no_vision', 'signed_change', 'neutral_change')
N_COMPONENTS = len(COMPONENTS)
ALL_COMPONENTS = (1 << N_COMPONENTS) - 1


def logit_coordinates(posterior):
    if np.any(posterior <= 0):
        raise ValueError('The decomposition requires positive archived probabilities.')
    negative, neutral, positive = np.moveaxis(np.log(posterior.astype(np.float64)), -1, 0)
    return np.stack(((positive - negative) / 2., neutral - (positive + negative) / 2.), axis=-1)


def posterior(signed, neutral):
    logits = np.stack((-signed, neutral, signed), axis=-1)
    mass = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return mass / mass.sum(axis=-1, keepdims=True)


def components(forecast):
    full = logit_coordinates(forecast)
    no_vision = np.broadcast_to(full[:, 2:3], full[:, :2].shape)
    change = full[:, :2] - no_vision
    return (*np.moveaxis(no_vision, -1, 0), *np.moveaxis(change, -1, 0))


def collect(runs, cohort):
    model = RUN_NAME[cohort]
    paths = run_paths(runs, SEEDS, cohort)
    compare = {'compact': paths[model], 'lnln_joint': paths['lnln_joint'],
               'rmnet_uniform': [runs / f'{cohort}_rmnet_uniform_seed{s}' for s in SEEDS]}
    reference = read_predictions(compare['compact'][0])
    if np.any(reference['masks'][3]):
        raise ValueError('The last observation view must have no audio or vision.')
    data = {key: reference[key] for key in ('ids', 'groups', 'y', 'masks')}
    for name, source in compare.items():
        saved = load_matched(source, reference)
        selected = load_metadata(source)
        run_name = dict(compact=model, rmnet_uniform='rmnet_core', lnln_joint='lnln_joint')[name]
        verify_training(run_name, selected, SEEDS, cohort)
        if name == 'rmnet_uniform' and any(x['config'].get('rmnet_class_weighting') != 'uniform' for x in selected):
            raise ValueError('Use the uniform-class-weight RM-Net refits.')
        key = 'intensity' if name == 'compact' else 'posterior'
        data[name] = np.stack([packet[key][1:4] for packet in saved])
    return dict(data, cohort=np.asarray(cohort), seeds=np.asarray(SEEDS))


def recombine(compact, reference, label):
    left, right = components(compact), components(reference)
    risks, accuracy = [], []
    for subset in range(ALL_COMPONENTS + 1):
        eta, nu, delta, offset = (right[i] if subset & (1 << i) else left[i] for i in range(N_COMPONENTS))
        p = posterior(eta + delta, nu + offset)
        if subset in (0, ALL_COMPONENTS):
            expected = compact if subset == 0 else reference
            np.testing.assert_allclose(p, expected[:, :2], atol=2e-7, rtol=5e-7)
        risks.append(ranked_probability_score(p, label).mean(axis=1))
        accuracy.append(float((p.argmax(-1) == label).mean()))
    contributions = {}
    for i, name in enumerate(COMPONENTS):
        bit = 1 << i
        value = np.zeros_like(risks[0])
        for subset in range(ALL_COMPONENTS + 1):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = factorial(size) * factorial(N_COMPONENTS - size - 1) / factorial(N_COMPONENTS)
            value += weight * (risks[subset] - risks[subset | bit])
        contributions[name] = value
    np.testing.assert_allclose(sum(contributions.values()), risks[0] - risks[-1], atol=1e-14)
    return risks, accuracy, contributions


def summarize(data, resamples):
    y, groups, compact = (data[name] for name in ('y', 'groups', 'compact'))
    _, membership = np.unique(groups, return_inverse=True)
    summary = {}
    target = outcomes(y)
    for name in REFERENCES:
        comparison = data[name]
        rps, accuracy, phi = recombine(compact, comparison, y)
        no_vision = phi['signed_no_vision'] + phi['neutral_no_vision']
        visible_change = phi['signed_change'] + phi['neutral_change']
        terms = ((np.square(coordinates(compact[:, :2]) - target) -
                  np.square(coordinates(comparison[:, :2]) - target)) / 4.).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(terms.sum(), np.mean(rps[0] - rps[-1]), atol=4e-8)
        forecasts = {'compact': 0, 'reference': ALL_COMPONENTS, 'replace_no_vision': 3, 'replace_changes': 12}
        summary[name] = dict(
            contributions={part: paired_reduction(value, membership, resamples, 8) for part, value in phi.items()},
            grouped={'no_vision': paired_reduction(no_vision, membership, resamples, 4),
                     'visual_changes': paired_reduction(visible_change, membership, resamples, 4)},
            recombined={method: dict(rps=float(rps[i].mean()), accuracy=accuracy[i]) for method, i in forecasts.items()},
            no_vision_rps={method: float(ranked_probability_score(probs[:, 2].astype(np.float64), y).mean())
                           for method, probs in (('compact', compact), ('reference', comparison))},
            gap=float((rps[0] - rps[-1]).mean()), signed_rps_gap=float(terms[0]), neutral_rps_gap=float(terms[1]))
    return dict(cohort=data['cohort'].item(), n_clips=len(y), n_videos=len(np.unique(groups)),
                seeds=data['seeds'].tolist(), resamples=resamples, comparisons=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', choices=RUN_NAME)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and provide --cohort with --runs.')
    if args.runs:
        data = collect(args.runs, args.cohort)
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            data = dict(packet)
    report = summarize(data, args.resamples)
    if args.write_evidence:
        partial = args.write_evidence.with_suffix('.npz.part')
        with partial.open('wb') as file:
            np.savez_compressed(file, **data)
        partial.replace(args.write_evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for comparator, values in report['comparisons'].items():
        print(f'{comparator}: total {values["gap"]:+.6f}')
        for name, estimate in values['grouped'].items():
            lower, upper = estimate['ci95_familywise']
            print(f'  {name}: {estimate["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]')


if __name__ == '__main__':
    main()
