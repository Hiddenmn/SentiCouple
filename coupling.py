"""Measure what shared supervision changes in the signed sentiment contrast."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax

from ablate import adjust_holm, paired_result, ranked_probability_score, read_predictions, seed_video_interval
from aggregate import f1_by_run


SIGN = np.array([-1, 0, 1])
CLASS_NAMES = ('negative', 'neutral', 'positive')
BASE_ARMS = ('anchor', 'fusion', 'independent_intensity', 'intensity')
CONTROL_ARMS = ('intensity_shuffled', 'intensity_latent', 'intensity_centered')


def logit_coordinates(posterior):
    negative, neutral, positive = np.moveaxis(np.log(posterior.astype(np.float64).clip(1e-12, 1)), -1, 0)
    return (positive - negative) / 2, neutral - (positive + negative) / 2


def probabilities(signed, neutral):
    return softmax(np.stack((-signed, neutral, signed), axis=-1), axis=-1)


def rps_summary(posterior, label):
    clip_rps = ranked_probability_score(posterior, label).mean(axis=(0, 1))
    by_class = {name: float(clip_rps[label == i].mean())
                for i, name in enumerate(CLASS_NAMES) if (label == i).any()}
    return dict(rps=float(clip_rps.mean()), accuracy=float((posterior.argmax(-1) == label).mean()),
                rps_by_class=by_class,
                balanced_rps=float(np.mean(list(by_class.values()))))


def signed_exchange(shared, comparator, label):
    """Exchange class-odds coordinates within matched seed and gap pairs."""
    shared_sign, shared_neutral = logit_coordinates(shared)
    other_sign, other_neutral = logit_coordinates(comparator)
    mean_difference = (shared_sign - other_sign).mean(axis=-1, keepdims=True)
    hybrids = dict(shared_with_comparator_mean=probabilities(shared_sign - mean_difference, shared_neutral),
                   comparator_with_shared_mean=probabilities(other_sign + mean_difference, other_neutral),
                   shared_with_comparator_sign=probabilities(other_sign, shared_neutral),
                   comparator_with_shared_sign=probabilities(shared_sign, other_neutral))
    return dict(mean_signed_difference=float(mean_difference.mean()),
                **{arm: rps_summary(values, label) for arm, values in hybrids.items()})


def cohort_summary(posterior, label):
    anchor_sign, _ = logit_coordinates(posterior['anchor'])
    comparisons = [arm for arm in ('independent_intensity', *CONTROL_ARMS) if arm in posterior]
    return dict(n_clips=len(label), class_counts=np.bincount(label, minlength=3).tolist(),
                methods={arm: rps_summary(values, label) for arm, values in posterior.items()},
                signed_exchange={arm: signed_exchange(posterior['intensity'], posterior[arm], label)
                                 for arm in comparisons},
                mean_half_log_odds_update={arm: float((logit_coordinates(values)[0] - anchor_sign).mean())
                                           for arm, values in posterior.items()})


def gradient_opposition(packets, posterior, label, target):
    """Measure the directions requested by the two losses at the fitted predictions."""
    shared = posterior['intensity']
    score = np.stack([packet['intensity_score'][1:3] for packet in packets])
    class_grad = shared @ SIGN - SIGN[label]
    opposite = class_grad * (score - target) < 0
    visible = packets[0]['masks'][1:3].any(axis=(1, 3))
    gain = (ranked_probability_score(posterior['independent_intensity'], label) -
            ranked_probability_score(shared, label))
    result = {}
    for category, name in enumerate(CLASS_NAMES):
        group = np.broadcast_to(visible & (label == category), opposite.shape)
        if group.any():
            entry = dict(n_predictions=int(group.sum()), fraction_opposed=float(opposite[group].mean()))
            for direction, mask in [('opposed', opposite), ('other', ~opposite)]:
                selected = group & mask
                entry[f'rps_reduction_{direction}'] = float(gain[selected].mean()) if selected.any() else None
            result[name] = entry
    return result


def fit_signed_intercept(signed, neutral, label):
    """Fit a single additive half log-odds intercept by tuning log loss."""
    observed_sign = SIGN[label].mean()
    return brentq(lambda offset: (probabilities(signed + offset, neutral) @ SIGN).mean() - observed_sign,
                  -50, 50)


def evaluate_intercepts(tune, evaluation, y_tune, y_eval):
    summaries = {}
    for arm in BASE_ARMS:
        eval_signed, eval_neutral = logit_coordinates(evaluation[arm])
        tune_signed, tune_neutral = logit_coordinates(tune[arm])
        offsets = np.array([fit_signed_intercept(signed, neutral, y_tune)
                            for signed, neutral in zip(tune_signed, tune_neutral)])
        fitted = probabilities(eval_signed + offsets[:, None, None], eval_neutral)
        summaries[arm] = dict(offset_by_seed=offsets.tolist(), **rps_summary(fitted, y_eval))
    return summaries


def load_cohorts(runs, controls):
    test_packets = [read_predictions(run) for run in runs]
    tuning_packets = []
    for run in runs:
        with np.load(run / 'tuning.npz', allow_pickle=False) as packet:
            tuning_packets.append(dict(packet))
    matched_keys = ('ids', 'groups', 'y', 'sentiment', 'masks')
    for packet, tune in zip(test_packets, tuning_packets):
        for key in matched_keys:
            np.testing.assert_array_equal(packet[key], test_packets[0][key], err_msg=key)
        np.testing.assert_array_equal(tune['y'], tuning_packets[0]['y'])
    if controls:
        for reference, control in zip(test_packets, controls):
            packet = read_predictions(control)
            for key in (*matched_keys, 'anchor', 'intensity', 'intensity_score'):
                np.testing.assert_array_equal(packet[key], reference[key], err_msg=f'{control.name}: {key}')
            for arm in CONTROL_ARMS:
                reference[arm] = packet[arm]
                reference[f'{arm}_score'] = packet[f'{arm}_score']
    return test_packets, tuning_packets


def video_resamples(groups, resamples):
    videos, membership = np.unique(groups, return_inverse=True)
    n_videos = len(videos)
    rng = np.random.default_rng(29)
    bootstrap = rng.multinomial(n_videos, np.full(n_videos, 1 / n_videos), size=resamples)
    swap = 2 * rng.integers(0, 2, size=(resamples, n_videos), dtype=np.int8) - 1
    return membership, bootstrap, swap


def rps_condition_effects(packets, labels, groups, resamples):
    """Contrast shared and independent heads under common coverage and class weights."""
    membership, videos, _ = video_resamples(groups, resamples)
    loss = {arm: ranked_probability_score(np.stack([p[arm][:3] for p in packets]), labels)
            for arm in ('intensity', 'independent_intensity')}
    advantage = loss['independent_intensity'] - loss['intensity']
    native, gaps = advantage[:, 0], advantage[:, 1:3].mean(1)
    result = {name: dict(rps_reduction=float(effect.mean()),
                        ci95_seeds_and_videos=seed_video_interval(effect, membership, resamples))
              for name, effect in [('native', native), ('gaps', gaps), ('gaps_minus_native', gaps - native)]}
    n_seeds, n_groups = len(packets), videos.shape[1]
    rng = np.random.default_rng(359)
    seed_draws = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=resamples)
    bootstrap_by_class, point_by_class = [], []
    for label in np.unique(labels):
        rows = labels == label
        group = membership[rows]
        counts = videos @ np.bincount(group, minlength=n_groups)
        totals = np.stack([np.bincount(group, weights=row[rows], minlength=n_groups) for row in gaps])
        numerator = ((videos @ totals.T) * seed_draws).sum(axis=1)
        bootstrap_by_class.append(np.divide(numerator, n_seeds * counts,
                                            out=np.full(resamples, np.nan), where=counts > 0))
        point_by_class.append(gaps[:, rows].mean())
    replicates = np.mean(bootstrap_by_class, axis=0)
    result['class_balanced_gaps'] = dict(rps_reduction=float(np.mean(point_by_class)),
                                         ci95_seeds_and_videos=np.nanquantile(replicates, [.025, .975]).tolist(),
                                         complete_replicates=int(np.isfinite(replicates).sum()))
    return result


def paired_controls(posterior, label, groups, resamples):
    cluster_samples = video_resamples(groups, resamples)
    rps = {arm: ranked_probability_score(values, label) for arm, values in posterior.items()}
    differences = {
        'genuine_vs_shuffled': rps['intensity_shuffled'] - rps['intensity'],
        'centered_vs_absolute': rps['intensity_latent'] - rps['intensity_centered'],
    }
    contrasts = {name: dict(**paired_result(values.mean(axis=(0, 1)), *cluster_samples),
                            rps_reduction_by_seed=values.mean(axis=(1, 2)).tolist())
                 for name, values in differences.items()}
    adjust_holm(contrasts)
    return contrasts


def matched_donor_result(test_packets, posterior, label, groups, resamples):
    """Compute the effect of replacing genuine visual values at fixed observed bins."""
    arms = ('fusion', 'independent_intensity', 'intensity')
    donor_logs = {arm: np.stack([p[f'{arm}_reference_log'][1:3] for p in test_packets]) for arm in arms}
    eligible = np.isfinite(np.stack(list(donor_logs.values()))).all(axis=(0, 1, 2))
    if not eligible.any():
        raise ValueError('No held-out clips have donors in every paired observation condition.')
    cluster_samples = video_resamples(groups[eligible], resamples)
    clip_gaps = {}
    for arm in arms:
        actual_log = np.log(posterior[arm][..., np.arange(len(label)), label].clip(1e-8, 1))
        clip_gaps[arm] = (actual_log - donor_logs[arm])[..., eligible].mean(axis=(0, 1))
    contrasts = {arm: paired_result(clip_gaps['intensity'] - clip_gaps[arm], *cluster_samples,
                                   effect='additional_nll_reduction') for arm in arms[:2]}
    adjust_holm(contrasts)
    return dict(n_clips=int(eligible.sum()), n_videos=int(len(np.unique(groups[eligible]))),
                genuine_vs_donor_nll={arm: float(gain.mean()) for arm, gain in clip_gaps.items()},
                shared_vs_comparator=contrasts)


def label_seed_sensitivity(packets, runs, label, groups, resamples):
    """Resample paired model fits and source videos as distinct sampling units."""
    seeds = [json.loads((run / 'run.json').read_text(encoding='utf-8'))['config']['seed'] for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Training seeds must be distinct.')
    _, membership = np.unique(groups, return_inverse=True)
    arms = ('fusion', 'independent_intensity', 'intensity_shuffled', 'intensity')
    losses = {arm: np.stack([ranked_probability_score(packet[arm][1:3], label).mean(axis=0)
                             for packet in packets]) for arm in arms}
    contrasts = {}
    family_tail = .025 / (len(arms) - 1)
    for arm in arms[:-1]:
        gain = losses[arm] - losses['intensity']
        by_class = {name: float(gain[:, label == i].mean())
                    for i, name in enumerate(CLASS_NAMES) if (label == i).any()}
        contrasts[arm] = dict(rps_reduction=float(gain.mean()), rps_reduction_by_seed=gain.mean(axis=1).tolist(),
                              ci95_seeds_and_videos=seed_video_interval(gain, membership, resamples),
                              ci95_familywise=seed_video_interval(gain, membership, resamples,
                                                                 quantiles=(family_tail, 1 - family_tail)),
                              rps_reduction_by_class=by_class)
    return dict(seeds=seeds, rps={arm: float(loss.mean()) for arm, loss in losses.items()},
                contrasts=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--controls', nargs='+', type=Path)
    parser.add_argument('--label-repeats', nargs='+', type=Path)
    parser.add_argument('--condition-effects', action='store_true')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.controls and len(args.controls) != len(args.runs):
        parser.error('Supply one matched control run per training seed.')
    if args.label_repeats and not args.controls:
        parser.error('Supply the matched shuffled-label controls for the original training seeds.')
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    test_packets, tune_packets = load_cohorts(args.runs, args.controls)
    labels, groups, target = (test_packets[0][key] for key in ('y', 'groups', 'sentiment'))
    tune_labels = tune_packets[0]['y']
    arms = (*BASE_ARMS, *CONTROL_ARMS) if args.controls else BASE_ARMS
    test = {arm: np.stack([packet[arm][1:3] for packet in test_packets]) for arm in arms}
    tune = {arm: np.stack([packet[arm][1:3] for packet in tune_packets]) for arm in BASE_ARMS}
    report = dict(runs=[str(run) for run in args.runs], n_videos=int(len(np.unique(groups))),
                  tuning=cohort_summary(tune, tune_labels), evaluation=cohort_summary(test, labels),
                  gradient_opposition=gradient_opposition(test_packets, test, labels, target),
                  tune_fitted_intercepts=evaluate_intercepts(tune, test, tune_labels, labels))
    if args.condition_effects:
        report['condition_effects'] = rps_condition_effects(test_packets, labels, groups, args.resamples)
    for arm, posterior in test.items():
        metrics = report['evaluation']['methods'][arm]
        metrics['macro_f1'] = float(f1_by_run(labels, posterior).mean())
        if arm != 'anchor':
            scores = np.stack([packet[f'{arm}_score'][1:3] for packet in test_packets])
            metrics['mae'] = float(np.abs(scores - target).mean())
    if args.controls:
        report['control_runs'] = [str(run) for run in args.controls]
        report['inference_replicates'] = args.resamples
        report['paired_controls'] = paired_controls(test, labels, groups, args.resamples)
    if all('intensity_reference_log' in packet for packet in test_packets):
        report['inference_replicates'] = args.resamples
        report['visual_donor_comparison'] = matched_donor_result(test_packets, test, labels, groups, args.resamples)
    if args.label_repeats:
        repetitions = [read_predictions(run) for run in args.label_repeats]
        for packet in repetitions:
            for key in ('ids', 'groups', 'y', 'sentiment', 'masks'):
                np.testing.assert_array_equal(packet[key], test_packets[0][key], err_msg=key)
        report['training_seed_sensitivity'] = label_seed_sensitivity(
            [*test_packets, *repetitions], [*args.runs, *args.label_repeats], labels, groups, args.resamples)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
