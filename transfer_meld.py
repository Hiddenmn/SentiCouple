"""Evaluate the pinned OMG-Emotion classifiers on the untouched MELD test set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, seed_video_interval
from compare_benchmarks import load_metadata
from coupling import CLASS_NAMES, logit_coordinates, probabilities as from_coordinates, rps_summary
from ordinal_controls import factorial_contrasts
from score_anchor_effects import SEEDS
from transfer_moud import ORDINAL_ARMS, check_configuration, matched_input, replay_seed


def report_dialogue(forecasts, target, resamples):
    probabilities = {arm: np.stack([values[arm] for values in forecasts]) for arm in forecasts[0]}
    methods = {arm: dict(gapped=rps_summary(values[:, 1:3], target.label),
                         native=rps_summary(values[:, :1], target.label))
               for arm, values in probabilities.items()}
    loss = {arm: ranked_probability_score(values, target.label) for arm, values in probabilities.items()}
    gapped = {arm: values[:, 1:3].mean(axis=1) for arm, values in loss.items()}
    estimates = factorial_contrasts(gapped['intensity'], gapped['independent_intensity'],
                                    gapped['intensity_class_mean'], gapped['independent_class_mean'])
    primary = estimates['sharing_with_class_means']
    native = loss['independent_class_mean'][:, 0] - loss['intensity_class_mean'][:, 0]
    secondary = dict(versus_anchor=gapped['anchor'] - gapped['intensity_class_mean'],
                     versus_rps_only=gapped['fusion_rps'] - gapped['intensity_class_mean'],
                     versus_ce_plus_rps=gapped['fusion_ce_rps'] - gapped['intensity_class_mean'],
                     change_in_sharing_gain=estimates['change_in_sharing_gain'],
                     gaps_minus_native=primary - native)
    episodes, memberships = np.unique(target.groups, return_inverse=True)

    def estimate(values, tail):
        return dict(rps_reduction=float(values.mean()), by_seed=values.mean(axis=1).tolist(),
                    ci95=seed_video_interval(values, memberships, resamples, quantiles=(tail, 1-tail)))

    independent_sign, independent_neutral = logit_coordinates(probabilities['independent_class_mean'][:, 1:3])
    shared_sign, shared_neutral = logit_coordinates(probabilities['intensity_class_mean'][:, 1:3])
    swapped_sign = ranked_probability_score(from_coordinates(shared_sign, independent_neutral), target.label)
    swapped_neutral = ranked_probability_score(from_coordinates(independent_sign, shared_neutral), target.label)
    independent_loss = loss['independent_class_mean'][:, 1:3]
    shared_loss = loss['intensity_class_mean'][:, 1:3]
    sign_gain = ((independent_loss - swapped_sign) + (swapped_neutral - shared_loss)).mean() / 2
    neutral_gain = ((independent_loss - swapped_neutral) + (swapped_sign - shared_loss)).mean() / 2
    np.testing.assert_allclose(sign_gain + neutral_gain, primary.mean(), atol=2e-7)

    return dict(seeds=SEEDS, resamples=resamples, n_clips=len(target.ids), n_episodes=len(episodes),
                label_counts=np.bincount(target.label, minlength=3).tolist(), methods=methods,
                primary=estimate(primary, .025),
                secondary={key: estimate(values, .025/len(secondary)) for key, values in secondary.items()},
                logit_exchange=dict(signed_contrast=float(sign_gain), neutral_contrast=float(neutral_gain)))


def decompose_transfer_change(review, dialogue):
    """Separate differences in class proportions from differences in classwise gains."""
    def proportions_and_gains(report):
        counts = np.asarray(report['label_counts'])
        shared = report['methods']['intensity_class_mean']['gapped']['rps_by_class']
        independent = report['methods']['independent_class_mean']['gapped']['rps_by_class']
        gains = np.array([independent[name] - shared[name] for name in CLASS_NAMES])
        return counts / counts.sum(), gains

    review_proportion, review_gain = proportions_and_gains(review)
    dialogue_proportion, dialogue_gain = proportions_and_gains(dialogue)
    prevalence = (dialogue_proportion - review_proportion) @ ((dialogue_gain + review_gain) / 2)
    classwise = ((dialogue_proportion + review_proportion) / 2) @ (dialogue_gain - review_gain)
    return dict(change_rps_reduction=float(prevalence + classwise), class_proportion_term=float(prevalence),
                within_class_term=float(classwise), classwise_review_gains=review_gain.tolist(),
                classwise_dialogue_gains=dialogue_gain.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--moud-report', type=Path, help='Describe the change from the matched frozen review comparison.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose at least one resample.')

    reference = [args.runs / f'omg_pointwise_heads_seed{seed}' for seed in SEEDS]
    ordinal = [args.runs / f'omg_ordinal_controls_seed{seed}' for seed in SEEDS]
    independent = [args.runs / f'omg_independent_class_mean_seed{seed}' for seed in SEEDS]
    config = check_configuration(*(load_metadata(paths) for paths in (reference, ordinal, independent)))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    batches, source_ids, source_index, target, target_index = matched_input(config['data'], args.target, config, device)
    if len(target.ids) != 2610 or len(np.unique(target.groups)) != 143:
        raise ValueError('Expected the 2,610 MELD test utterances from 143 episode clusters.')
    forecasts = []
    for seed, original, control, independent_mean in zip(SEEDS, reference, ordinal, independent):
        comparisons = [(control, ORDINAL_ARMS), (independent_mean, ('independent_class_mean',))]
        forecasts.append(replay_seed(original, comparisons, config, batches, source_ids, source_index, target_index))
        print(f'Reproduced source predictions and transferred seed {seed}.', flush=True)
    report = report_dialogue(forecasts, target, args.resamples)
    if args.moud_report:
        review = json.loads(args.moud_report.read_text(encoding='utf-8'))
        if review['seeds'] != list(SEEDS):
            raise ValueError('The transfer comparison requires the same source seeds.')
        report['change_from_moud'] = decompose_transfer_change(review, report)
    with args.target.open('rb') as stream:
        report['feature_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('primary', 'secondary')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
