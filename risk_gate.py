"""Fit anchor-mixture weights to tuning ranked probability score."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import adjust_holm, paired_result, ranked_probability_score
from coupling import load_cohorts, rps_summary, video_resamples


def risk_coefficients(anchor, fused, labels):
    cumulative = anchor.cumsum(-1)[..., :2]
    delta = fused.cumsum(-1)[..., :2] - cumulative
    residual = cumulative - (labels[..., None] <= np.arange(2))
    return np.square(delta).mean(-1), (2 * residual * delta).mean(-1)


def rps_weight(quadratic, linear):
    if quadratic.size == 0 or quadratic.mean() <= 1e-14:
        return 0.
    return float(np.clip(-linear.mean() / (2 * quadratic.mean()), 0, 1))


def fit_weights(anchor, fused, label):
    labels = np.broadcast_to(label, anchor.shape[:-1])
    quadratic, linear = risk_coefficients(anchor, fused, labels)
    global_weight = rps_weight(quadratic, linear)
    changed = anchor.argmax(-1) != fused.argmax(-1)
    by_change = [rps_weight(quadratic[mask], linear[mask]) for mask in (~changed, changed)]
    class_limits = [1.]
    for polarity in range(3):
        member = labels == polarity
        if member.any() and quadratic[member].mean() > 1e-14:
            class_limits.append(float(np.clip(-linear[member].mean() / quadratic[member].mean(), 0, 1)))
    return dict(global_rps=global_weight, by_decision_change=by_change,
                class_constrained=min(global_weight, *class_limits))


def mixtures(anchor, fused, weights):
    changed = (anchor.argmax(-1) != fused.argmax(-1)).astype(int)
    per_condition = np.asarray(weights['by_decision_change'])[changed]
    gap = fused - anchor
    return dict(anchor=anchor, shared=fused, global_rps=anchor + weights['global_rps'] * gap,
                by_decision_change=anchor + per_condition[..., None] * gap,
                class_constrained=anchor + weights['class_constrained'] * gap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    test, tune = load_cohorts(args.runs, None)
    weights = [fit_weights(packet['anchor'][1:3], packet['intensity'][1:3], packet['y']) for packet in tune]
    alternatives = [mixtures(packet['anchor'][1:3], packet['intensity'][1:3], w)
                    for packet, w in zip(test, weights)]
    labels = test[0]['y']
    posterior = {name: np.stack([entry[name] for entry in alternatives]) for name in alternatives[0]}
    scores = {name: rps_summary(values, labels) for name, values in posterior.items()}
    losses = {name: ranked_probability_score(values, labels).mean(axis=(0, 1))
              for name, values in posterior.items()}
    cluster_samples = video_resamples(test[0]['groups'], args.resamples)
    contrasts = {name: paired_result(losses['shared'] - loss, *cluster_samples)
                 for name, loss in losses.items() if name not in ('anchor', 'shared')}
    adjust_holm(contrasts)
    result = dict(tuning_weights=weights, scores=scores, versus_shared=contrasts,
                  n_clips=len(labels), n_videos=int(len(np.unique(test[0]['groups']))))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Tuning weights:', weights)
    print('Held-out RPS:', {name: score['rps'] for name, score in scores.items()})
    print('Differences:', contrasts)


if __name__ == '__main__':
    main()
