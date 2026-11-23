"""Decompose the matched benchmark gap under probability averaging over seeds."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from ablate import ranked_probability_score
from polarity_pool import coordinates, outcomes


COMPACT = 'original_max_shared'
REFERENCES = ('tfr_joint', 'lnln_joint', 'rmnet_uniform')
SIZES = (1, 2, 3, 5, 10)


def estimate(posterior, label, membership, seed_weights, video_weights, weighted_count):
    count, gaps, clips, _ = posterior.shape
    target = outcomes(label)
    residual = (coordinates(posterior) - target) / 2
    risks = np.square(residual)
    ensemble_risk = np.square(residual.mean(0)).mean(axis=(0, 1))
    average_risk = risks.mean(axis=(0, 1, 2))
    np.testing.assert_allclose(average_risk.sum(), ranked_probability_score(posterior, label).mean(),
                               rtol=0, atol=5e-8)
    np.testing.assert_allclose(average_risk - ensemble_risk,
                               np.square(residual - residual.mean(0)).mean(axis=(0, 1, 2)), atol=1e-14)

    # A source video's Gram matrix retains the quadratic dependence on seed weights.
    grouped = np.empty((membership.max() + 1, count, count), dtype=np.float64)
    for group in range(len(grouped)):
        error = residual[:, :, membership == group]
        grouped[group] = np.einsum('avnc,bvnc->ab', error, error) / gaps
    diagonal = np.diagonal(grouped, axis1=1, axis2=2)
    single_by_video = seed_weights @ diagonal.T
    outer = (seed_weights[:, :, None] * seed_weights[:, None, :]).reshape(len(seed_weights), -1)
    ensemble_by_video = outer @ grouped.reshape(len(grouped), -1).T
    sampled = [np.sum(value * video_weights, axis=1) / weighted_count
               for value in (single_by_video, ensemble_by_video)]
    np.testing.assert_allclose(np.trace(grouped.sum(0)) / (count * clips), average_risk.sum(), atol=1e-13)
    np.testing.assert_allclose(grouped.sum() / (count**2 * clips), ensemble_risk.sum(), atol=1e-13)

    trajectory = {}
    for size in SIZES:
        fraction = (count - size) / (size * (count - 1))
        signed, neutral = ensemble_risk + fraction * (average_risk - ensemble_risk)
        trajectory[str(size)] = dict(rps=float(signed + neutral), signed_rps=float(signed), neutral_rps=float(neutral))
    # Enumerate all two-model subsets on a few clips to check the finite-population identity.
    example = posterior[:, :, :8]
    two_model = np.stack([(example[i] + example[j]) / 2 for i, j in combinations(range(count), 2)])
    loss = ranked_probability_score(two_model, label[:8]).mean()
    single = risks[:, :, :8].sum(-1).mean()
    full = np.square(residual[:, :, :8].mean(0)).sum(-1).mean()
    predicted = full + (count - 2) / (2 * (count - 1)) * (single - full)
    np.testing.assert_allclose(loss, predicted, rtol=0, atol=5e-8)

    return dict(curve=trajectory, signed_diversity=float(average_risk[0]-ensemble_risk[0]),
                neutral_diversity=float(average_risk[1]-ensemble_risk[1]),
                accuracy_one=float((posterior.argmax(-1) == label).mean()),
                accuracy_all=float((posterior.mean(0).argmax(-1) == label).mean())), sampled


def summarize(data, resamples):
    groups, label, seeds = (data[key] for key in ('groups', 'y', 'seeds'))
    if len(seeds) != SIZES[-1] or len(np.unique(seeds)) != len(seeds):
        raise ValueError('The ensemble analysis requires all ten paired training seeds.')
    videos, membership = np.unique(groups, return_inverse=True)
    n = len(seeds)
    sizes = np.bincount(membership)
    rng = np.random.default_rng(29)
    seed_weights = rng.multinomial(n, np.full(n, 1 / n), size=resamples) / n
    video_weights = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=resamples)
    weighted_count = video_weights @ sizes

    methods, samples = {}, {}
    parameters = json.loads(data['parameters'].item())
    for name in (COMPACT, *REFERENCES):
        posterior = np.ascontiguousarray(data[name], dtype=np.float64)
        np.testing.assert_allclose(posterior.sum(-1), 1., rtol=0, atol=2e-7)
        methods[name], samples[name] = estimate(posterior, label, membership, seed_weights,
                                                video_weights, weighted_count)
        methods[name]['parameters'] = parameters[name][0]
    comparisons = {}
    tail = .025 / (2 * len(REFERENCES))

    def effect(estimate, draws):
        return dict(estimate=estimate, ci95_familywise=np.quantile(draws, (tail, 1-tail)).tolist())

    for name in REFERENCES:
        before, after = [methods[COMPACT]['curve'][size]['rps'] - methods[name]['curve'][size]['rps']
                         for size in ('1', str(n))]
        sampled_before, sampled_after = [a-b for a, b in zip(samples[COMPACT], samples[name], strict=True)]
        delta = {axis: methods[COMPACT][axis + '_diversity'] - methods[name][axis + '_diversity']
                 for axis in ('signed', 'neutral')}
        np.testing.assert_allclose(sum(delta.values()), before-after, atol=1e-14)
        comparisons[name] = dict(gap_one=effect(before, sampled_before), gap_all=effect(after, sampled_after),
                                 gap_reduction=effect(before-after, sampled_before-sampled_after),
                                 gap_reduction_by_coordinate=delta)
    return dict(cohort=data['cohort'].item(), seeds=seeds.tolist(), n_clips=len(label), n_videos=len(videos),
                ensemble_sizes=list(SIZES), resamples=resamples, methods=methods, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Use a positive number of resamples.')
    with np.load(args.evidence, allow_pickle=False) as packet:
        report = summarize(packet, args.resamples)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, comparison in report['comparisons'].items():
        for label, result in comparison.items():
            if 'estimate' in result:
                lower, upper = result['ci95_familywise']
                print(f'{name} / {label}: {result["estimate"]:+.6f} [{lower:+.6f}, {upper:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
