"""Compare SentiCouple and UCMR-Net with text, audio and vision."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions
from benchmark_mosei import ROOT, SEEDS
from measure import macro_f1_from_confusion
from mosei_results import profile_all, summarize_benchmark


COHORTS = {
    'sims_v2': 'sims_v2_shared_score',
    'mosei': 'mosei_three_stream',
}


def neutral_margin(posterior):
    log_p = np.log(posterior.clip(1e-12, 1.0))
    return np.maximum(log_p[..., 0], log_p[..., 2]) - log_p[..., 1]


def neutral_offset_labels(posterior, offset):
    polar = np.where(posterior[..., 2] > posterior[..., 0], 2, 0)
    return np.where(neutral_margin(posterior) <= offset, 1, polar)


def video_confusions(y, decisions, membership, n_videos):
    shape = decisions.shape[:-1]
    actual = 9 * membership + 3 * y
    counts = np.stack([
        np.bincount(actual + classes, minlength=9 * n_videos)
        for classes in decisions.reshape(-1, y.size)
    ])
    return counts.reshape(*shape, n_videos, 3, 3)


def tune_neutral_offset(y, posterior):
    margins = neutral_margin(posterior)
    # One threshold across both gap widths. Include the unshifted decision.
    offsets = np.unique(np.r_[0., np.nextafter(margins.min(), -np.inf),
                              np.quantile(margins, np.linspace(0., 1., 101))])
    decisions = neutral_offset_labels(posterior, offsets[:, None, None])
    confusion = video_confusions(y, decisions, np.zeros_like(y), 1).squeeze(-3)
    scores = macro_f1_from_confusion(confusion).mean(-1)
    tied = offsets[scores == scores.max()]
    return float(tied[np.abs(tied).argmin()])


def categorical_comparison(paths, resamples):
    first = read_predictions(paths['SentiCouple'][0])
    label = first['y']
    videos, membership = np.unique(first['groups'], return_inverse=True)
    counts, offsets = {}, {}
    for name, directories in paths.items():
        key = 'intensity' if name == 'SentiCouple' else 'posterior'
        decisions, offsets[name] = [], []
        for directory in directories:
            prediction = read_predictions(directory)
            for field in ('ids', 'groups', 'y', 'masks'):
                np.testing.assert_array_equal(prediction[field], first[field], err_msg=directory.name)
            posterior = prediction[key][1:3]
            with np.load(directory / 'tuning.npz', allow_pickle=False) as tuning:
                offset = tune_neutral_offset(tuning['y'], tuning[key][1:3])
            decisions.append(np.stack([posterior.argmax(-1),
                                       neutral_offset_labels(posterior, offset)]))
            offsets[name].append(offset)
        counts[name] = video_confusions(label, np.stack(decisions), membership, len(videos))

    rng = np.random.default_rng(451)
    group_draws = rng.multinomial(len(videos), np.full(len(videos), 1 / len(videos)), size=resamples)
    fit_count = len(paths['SentiCouple'])
    fit_draws = rng.multinomial(fit_count, np.full(fit_count, 1 / fit_count), size=resamples)
    means, draws = {}, {}
    for name, matrices in counts.items():
        totals = matrices.sum(-3)
        recall = np.diagonal(totals, axis1=-2, axis2=-1) / totals.sum(-1).clip(min=1)
        means[name] = dict(macro_f1=macro_f1_from_confusion(totals).mean(axis=(0, 2)).tolist(),
                           recall=recall.mean(axis=(0, 2)).tolist(),
                           neutral_offsets=offsets[name])
        by_video = np.moveaxis(matrices, -3, 0)
        sampled = (group_draws @ by_video.reshape(len(videos), -1)).reshape(
            resamples, *by_video.shape[1:])
        by_fit = macro_f1_from_confusion(sampled).mean(-1)
        draws[name] = (by_fit * fit_draws[..., None]).sum(1) / fit_count
    difference = draws['SentiCouple'] - draws['ucmr-net']
    mean_gain = np.subtract(means['SentiCouple']['macro_f1'], means['ucmr-net']['macro_f1'])
    return dict(rules=['argmax', 'tuning_neutral_offset'], models=means,
                shared_minus_ucmr_macro_f1=mean_gain.tolist(),
                ci95_familywise=np.quantile(difference, [.0125, .9875], axis=0).T.tolist())


def evaluate(runs, cohort, resamples, profile_upstream=None, *, categorical_check=False):
    shared = [runs / f'{COHORTS[cohort]}_seed{seed}' for seed in SEEDS]
    reference = [runs / f'{cohort}_three_stream_ucmr_net_seed{seed}' for seed in SEEDS]
    if not read_predictions(shared[0])['masks'][1:3, 0].any():
        raise ValueError('The comparison requires observed audio in the evaluation views.')
    paths = {'SentiCouple': shared, 'ucmr-net': reference}
    result = summarize_benchmark(paths, SEEDS, resamples)
    if categorical_check:
        result['categorical_comparison'] = categorical_comparison(paths, resamples)
    if profile_upstream is not None:
        if cohort == 'sims_v2':
            data = ROOT / 'data/ch_sims_v2_unaligned.pkl'
            group_reference = ROOT / 'data/ch_sims_unaligned_39.pkl'
            eval_groups = None
        else:
            data = ROOT / 'data/mosei_aligned_50.pkl'
            group_reference = None
            eval_groups = ROOT / 'splits/mosei_eval_videos.txt'
        result['timing'] = profile_all(paths, {'ucmr-net': profile_upstream}, data, eval_groups,
                                       cohort=cohort, group_reference=group_reference, visual_only=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--cohort', choices=tuple(COHORTS), required=True)
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--profile-upstream', type=Path, help='Time seed 97 on CUDA using the pinned UCMR-Net checkout.')
    parser.add_argument('--categorical-check', action='store_true',
                        help='Compare argmax and a tuning-only neutral-class offset.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    result = evaluate(args.runs, args.cohort, args.resamples, args.profile_upstream,
                      categorical_check=args.categorical_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
