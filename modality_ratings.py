"""Relate held-out log-odds corrections to independent human modality ratings."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from ablate import ranked_probability_score
from aggregate import bootstrap_mean
from coupling import load_cohorts, logit_coordinates, video_resamples


def ratings(path, ids, intensity):
    # Read the checksum-verified public CH-SIMS v2 feature packet.
    with path.open('rb') as stream:
        splits = pickle.load(stream)
    records = {}
    for packet in splits.values():
        columns = (np.asarray(packet[key]).reshape(-1) for key in
                   ('id', 'regression_labels', 'regression_labels_T', 'regression_labels_V'))
        records.update((str(clip), (overall, text, vision)) for clip, overall, text, vision in zip(*columns, strict=True))
    selected = np.asarray([records[clip] for clip in ids], dtype=np.float64)
    np.testing.assert_allclose(selected[:, 0], intensity, rtol=0, atol=1e-7)
    if not np.isfinite(selected).all():
        raise ValueError('Expected finite overall, text, and vision ratings.')
    return selected[:, 1], selected[:, 2]


def conditional_slope(vision, correction, text, membership, weights):
    _, stratum = np.unique(text, return_inverse=True)
    _, n_videos = weights.shape
    n_strata = stratum.max() + 1
    rows = membership * n_strata + stratum

    def means_by_group(values=None):
        totals = np.bincount(rows, weights=values, minlength=n_videos * n_strata).reshape(n_videos, n_strata)
        return weights @ totals

    n = means_by_group().clip(1)
    sum_v, sum_d = means_by_group(vision), means_by_group(correction)
    within_covariance = means_by_group(vision * correction) - sum_v * sum_d / n
    within_variance = means_by_group(vision ** 2) - sum_v ** 2 / n
    denominator = within_variance.sum(axis=1)
    if denominator[0] <= 0:
        raise ValueError('Visual ratings have no variation within text strata.')
    valid = denominator > 0
    return within_covariance.sum(axis=1)[valid] / denominator[valid]


def support_difference(gains, membership, weights, supported, unsupported):
    means = []
    for selected in (supported, unsupported):
        count = np.bincount(membership[selected], minlength=weights.shape[1])
        total = np.bincount(membership[selected], weights=gains[selected], minlength=weights.shape[1])
        sizes = weights @ count
        means.append(np.divide(weights @ total, sizes, out=np.full(len(weights), np.nan), where=sizes > 0))
    differences = means[0] - means[1]
    return dict(rps_reduction_difference=float(differences[0]),
                ci95=np.nanquantile(differences[1:], [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--controls', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.controls and len(args.controls) != len(args.runs)):
        parser.error('Use positive resamples and one control directory per original run.')
    packets, _ = load_cohorts(args.runs, args.controls)
    reference = packets[0]
    labels, groups = reference['y'], reference['groups']
    text, vision = ratings(args.data, reference['ids'], reference['sentiment'])
    membership, bootstrap, _ = video_resamples(groups, args.resamples)
    weights = np.concatenate([np.ones((1, bootstrap.shape[1]), dtype=bootstrap.dtype), bootstrap])
    arms = ['anchor', 'fusion', 'independent_intensity', 'intensity']
    if args.controls:
        arms.append('intensity_shuffled')
    probabilities = {arm: np.stack([packet[arm][1:3] for packet in packets]) for arm in arms}
    signed = {arm: logit_coordinates(p)[0].mean(axis=(0, 1)) for arm, p in probabilities.items()}
    corrections = {arm: signed[arm] - signed['anchor'] for arm in arms[1:]}
    contrasts = {f'shared_minus_{arm}': corrections['intensity'] - corrections[arm]
                 for arm in corrections if arm != 'intensity'}
    slopes = {}
    for arm, corrections_in_arm in {**corrections, **contrasts}.items():
        beta = conditional_slope(vision, corrections_in_arm, text, membership, weights)
        slopes[arm] = dict(slope=float(beta[0]), ci95=np.quantile(beta[1:], [.025, .975]).tolist())
    losses = {arm: ranked_probability_score(p, labels).mean(axis=(0, 1)) for arm, p in probabilities.items()}
    text_polarity, visual_polarity = np.sign(text).astype(int) + 1, np.sign(vision).astype(int) + 1
    disagree = text_polarity != visual_polarity
    supported = disagree & (visual_polarity == labels)
    unsupported = disagree & (visual_polarity != labels)
    subgroups = {}
    for name, select in [('agree', ~disagree), ('vision_agrees_with_outcome', supported),
                         ('vision_disagrees_with_outcome', unsupported)]:
        subgroups[name] = dict(n_clips=int(select.sum()), n_videos=len(np.unique(groups[select])),
                              rps={arm: float(result[select].mean()) for arm, result in losses.items()} if select.any() else {})
        if select.any():
            effects = {arm: (losses['anchor'] - result)[select] for arm, result in losses.items() if arm != 'anchor'}
            subgroups[name]['relative_to_anchor'] = {
                arm: dict(rps_reduction=float(gain.mean()), ci95=bootstrap_mean(gain, membership[select], bootstrap))
                for arm, gain in effects.items()}
    report = dict(n_clips=len(labels), n_videos=int(len(np.unique(groups))), n_text_ratings=len(np.unique(text)),
                  n_visual_ratings=len(np.unique(vision)), within_text_rating=slopes, rps_by_agreement=subgroups)
    if supported.any() and unsupported.any():
        report['difference_by_visual_support'] = support_difference(losses['anchor'] - losses['intensity'],
                                                                   membership, weights, supported, unsupported)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
