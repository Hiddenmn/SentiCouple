"""Estimate paired ordinal ablation effects while keeping source videos intact."""

import argparse
import json
from pathlib import Path

import numpy as np

from aggregate import combine, f1_by_run


BASELINES = ['fusion', 'intensity', 'ordered']
FULL = 'ordered_intensity'
ENCODER_CONTROLS = ('intensity_unscaled', 'intensity_pointwise')
ARCHITECTURE_CONTROLS = (*ENCODER_CONTROLS, 'intensity_stream_query', 'intensity_signed_only',
                         'intensity_direct_logits')
HEAD_COMPARISONS = ('shared-score', 'visual-minimal', 'external-valence', 'calibrated-heads')
READOUT_CONTROLS = {'attention-query': ('intensity', 'intensity_stream_query'),
                    'neutral-mixing': ('intensity_mean_polarity', 'intensity_mean_polarity')}


def read_predictions(run):
    with np.load(run / 'predictions.npz', allow_pickle=False) as packet:
        return dict(packet)


def ranked_probability_score(posterior, label):
    cumulative = posterior.cumsum(-1)[..., :2]
    observed = label[:, None] <= np.arange(2)
    return np.square(cumulative - observed).mean(-1)


def paired_result(per_clip_gain, membership, bootstrap, swap, effect='rps_reduction'):
    n_videos = bootstrap.shape[1]
    totals = np.bincount(membership, weights=per_clip_gain, minlength=n_videos)
    count = np.bincount(membership, minlength=n_videos)
    difference = per_clip_gain.mean()
    replicates = (bootstrap @ totals) / (bootstrap @ count)
    null = (swap @ totals) / len(per_clip_gain)
    p_value = (1 + np.sum(np.abs(null) >= abs(difference))) / (len(null) + 1)
    return {effect: float(difference), 'ci95': np.quantile(replicates, [.025, .975]).tolist(),
            'p_value': float(p_value)}


def adjust_holm(contrasts):
    adjusted = 0.0
    for rank, contrast in enumerate(sorted(contrasts.values(), key=lambda x: x['p_value'])):
        adjusted = min(1, max(adjusted, (len(contrasts) - rank) * contrast['p_value']))
        contrast['p_holm'] = adjusted


def seed_video_interval(gain, membership, resamples, seed=29, quantiles=(.025, .975)):
    """Resample paired fits and source videos, retaining their crossed structure."""
    count = np.bincount(membership)
    totals = np.stack([np.bincount(membership, weights=row, minlength=len(count)) for row in gain])
    return grouped_seed_video_interval(totals, count, resamples, seed, quantiles)


def grouped_seed_video_interval(totals, count, resamples, seed=29, quantiles=(.025, .975)):
    """Use the per-video loss sums and clip counts for crossed resampling."""
    n_seeds, n_videos = totals.shape
    rng = np.random.default_rng(seed)
    seeds = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=resamples)
    videos = rng.multinomial(n_videos, np.full(n_videos, 1 / n_videos), size=resamples)
    replicates = ((videos @ totals.T) * seeds).sum(axis=1) / (n_seeds * (videos @ count))
    return np.quantile(replicates, quantiles).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--mode', required=True, choices=['restarts', 'folds'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--comparison', choices=['ordinal', *HEAD_COMPARISONS, *READOUT_CONTROLS,
                                                 'streams', 'visual-score', 'visual-encoder', 'architecture'],
                        default='ordinal')
    parser.add_argument('--comparator-runs', nargs='+', type=Path)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    external = args.comparison == 'external-valence'
    readout_control = READOUT_CONTROLS.get(args.comparison)
    if external and args.mode != 'restarts':
        parser.error('The external valence comparison uses paired training seeds.')
    packets = [read_predictions(run) for run in args.runs]
    architecture_controls = ()
    if args.comparison in ('visual-encoder', 'architecture'):
        if args.mode != 'restarts':
            parser.error('Architecture controls compare paired fits on the same holdout.')
        architecture_controls = (ARCHITECTURE_CONTROLS if args.comparison == 'architecture'
                                 else ENCODER_CONTROLS)
    if args.comparison in ('streams', 'visual-score') or readout_control or architecture_controls:
        if not args.comparator_runs or len(args.comparator_runs) != len(args.runs):
            parser.error('Supply a matched comparator run for each run.')
        for packet, comparator_run in zip(packets, args.comparator_runs):
            reference = read_predictions(comparator_run)
            for key in ('ids', 'groups', 'y', 'sentiment', 'masks', 'anchor'):
                np.testing.assert_array_equal(packet[key], reference[key])
            if readout_control:
                original, name = readout_control
                packet[name] = reference[original]
                packet[f'{name}_score'] = reference[f'{original}_score']
            elif architecture_controls:
                np.testing.assert_allclose(packet['intensity'], reference['intensity'], atol=3e-5, rtol=5e-5,
                                           err_msg='The repeated shared head must match the archived run.')
                packet['intensity'], packet['intensity_score'] = reference['intensity'], reference['intensity_score']
                for arm in architecture_controls:
                    packet[arm], packet[f'{arm}_score'] = reference[arm], reference[f'{arm}_score']
            else:
                desired = ('intensity_vision', 'intensity_vision_score') if args.comparison == 'visual-score' else (
                    'fusion', 'intensity', 'fusion_score', 'intensity_score')
                for key in desired:
                    packet[key] = reference[key]
    if args.mode == 'restarts':
        for packet in packets[1:]:
            for key in ('ids', 'groups', 'y', 'sentiment', 'masks'):
                np.testing.assert_array_equal(packet[key], packets[0][key])
    ids, groups, label, scores = (
        combine(packets, key, args.mode, 0)[0] for key in ('ids', 'groups', 'y', 'sentiment'))
    n_groups = len(np.unique(groups))
    if len(ids) != len(np.unique(ids)) or (args.mode == 'folds' and
            n_groups != sum(len(np.unique(packet['groups'])) for packet in packets)):
        raise ValueError('Expected disjoint held-out original videos across folds.')
    _, membership = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(29)
    bootstrap = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=args.resamples)
    swap = rng.integers(0, 2, size=(args.resamples, n_groups), dtype=np.int8) * 2 - 1
    methods, clip_rps, clip_mae = {}, {}, {}
    seed_clip_rps, seed_clip_mae = {}, {}
    if args.comparison == 'ordinal':
        arms = [*BASELINES, FULL]
    elif args.comparison in HEAD_COMPARISONS:
        arms = ['fusion', 'intensity', 'independent_intensity']
        if external:
            arms.append('intensity_shuffled')
    elif args.comparison == 'streams':
        arms = ['fusion', 'intensity', 'intensity_audio', 'intensity_vision']
    elif readout_control:
        arms = ['intensity', readout_control[1]]
    elif architecture_controls:
        arms = ['intensity', *architecture_controls]
    else:
        arms = ['fusion_vision', 'intensity_vision', 'independent_vision']
    for arm in arms:
        posterior = combine(packets, arm, args.mode, 1)[:, 1:3]
        intensity = combine(packets, f'{arm}_score', args.mode, 1)[:, 1:3]
        individual_rps = ranked_probability_score(posterior, label)
        seed_clip_rps[arm] = individual_rps.mean(axis=1)
        clip_rps[arm] = individual_rps.mean(axis=(0, 1))
        absolute_error = np.abs(intensity - scores)
        seed_clip_mae[arm] = absolute_error.mean(axis=1)
        clip_mae[arm] = absolute_error.mean(axis=(0, 1))
        methods[arm] = dict(rps=float(clip_rps[arm].mean()), accuracy=float((posterior.argmax(-1) == label).mean()),
                            macro_f1=float(f1_by_run(label, posterior).mean()),
                            nll=float(-np.log(posterior[..., np.arange(len(label)), label].clip(1e-8)).mean()),
                            mae=float(clip_mae[arm].mean()),
                            rps_by_seed=individual_rps.mean(axis=(1, 2)).tolist(),
                            rps_by_block=individual_rps.mean(axis=(0, 2)).tolist())
        signed = label != 1
        if signed.any():
            predicted = posterior.argmax(-1)[..., signed]
            methods[arm]['opposite_sign_rate'] = float((np.abs(predicted - label[signed]) == 2).mean())
        print(arm, methods[arm], flush=True)
    if args.comparison == 'ordinal':
        candidate = FULL
        contrasts = {arm: paired_result(clip_rps[arm] - clip_rps[candidate], membership, bootstrap, swap)
                     for arm in BASELINES}
    elif args.comparison in (*HEAD_COMPARISONS, 'neutral-mixing'):
        candidate = 'intensity'
        comparator = 'intensity_mean_polarity' if args.comparison == 'neutral-mixing' else 'independent_intensity'
        contrasts = {
            f'{comparator}_rps': paired_result(
                clip_rps[comparator] - clip_rps[candidate], membership, bootstrap, swap),
            f'{comparator}_mae': paired_result(
                clip_mae[comparator] - clip_mae[candidate], membership, bootstrap, swap, 'mae_reduction'),
        }
        if args.comparison in ('visual-minimal', 'external-valence', 'calibrated-heads'):
            contrasts['fusion_rps'] = paired_result(
                clip_rps['fusion'] - clip_rps[candidate], membership, bootstrap, swap)
        if args.comparison == 'calibrated-heads':
            contrasts['fusion_mae'] = paired_result(
                clip_mae['fusion'] - clip_mae[candidate], membership, bootstrap, swap, 'mae_reduction')
        if external:
            contrasts['intensity_shuffled_rps'] = paired_result(
                clip_rps['intensity_shuffled'] - clip_rps[candidate], membership, bootstrap, swap)
        if args.mode == 'restarts' and len(packets) > 1:
            family_tail = .025 / (len(contrasts) - int(external))
            for name, result in contrasts.items():
                arm, metric = name.rsplit('_', 1)
                losses = seed_clip_rps if metric == 'rps' else seed_clip_mae
                gain = losses[arm] - losses[candidate]
                result['reduction_by_seed'] = gain.mean(axis=-1).tolist()
                result['ci95_seeds_and_videos'] = seed_video_interval(gain, membership, args.resamples)
                if not external or name != 'independent_intensity_rps':
                    result['ci95_familywise'] = seed_video_interval(gain, membership, args.resamples,
                                                                    quantiles=(family_tail, 1 - family_tail))
    elif args.comparison == 'streams':
        candidate = 'intensity'
        contrasts = {arm: paired_result(clip_rps[arm] - clip_rps[candidate], membership, bootstrap, swap)
                     for arm in ('fusion', 'intensity_audio', 'intensity_vision')}
    elif args.comparison == 'attention-query':
        candidate = 'intensity'
        contrasts = {'stream_query_rps': paired_result(clip_rps['intensity_stream_query'] - clip_rps[candidate],
                                                       membership, bootstrap, swap)}
    elif architecture_controls:
        candidate = 'intensity'
        unscaled, pointwise = ENCODER_CONTROLS
        pairs = {arm: (arm, candidate) for arm in architecture_controls}
        pairs['pointwise_vs_unscaled'] = (pointwise, unscaled)
        family_tail = .025 / len(pairs)
        contrasts = {}
        for name, (control, model) in pairs.items():
            gain = seed_clip_rps[control] - seed_clip_rps[model]
            result = paired_result(gain.mean(axis=0), membership, bootstrap, swap)
            result['reduction_by_seed'] = gain.mean(axis=-1).tolist()
            if len(gain) > 1:
                result['ci95_seeds_and_videos'] = seed_video_interval(gain, membership, args.resamples)
                result['ci95_familywise'] = seed_video_interval(gain, membership, args.resamples,
                                                                quantiles=(family_tail, 1 - family_tail))
            contrasts[f'{name}_rps'] = result
    else:
        candidate = 'intensity_vision'
        contrasts = {
            'fusion_vision_rps': paired_result(
                clip_rps['fusion_vision'] - clip_rps[candidate], membership, bootstrap, swap),
            'independent_vision_rps': paired_result(
                clip_rps['independent_vision'] - clip_rps[candidate], membership, bootstrap, swap),
            'independent_vision_mae': paired_result(
                clip_mae['independent_vision'] - clip_mae[candidate], membership, bootstrap, swap, 'mae_reduction'),
        }
    if external:
        adjust_holm({name: result for name, result in contrasts.items() if name != 'independent_intensity_rps'})
    else:
        adjust_holm(contrasts)
    for arm, result in contrasts.items():
        print(arm if architecture_controls else f'{candidate} vs {arm}', result, flush=True)
    report = dict(n_clips=len(ids), n_videos=n_groups, runs=len(packets), mode=args.mode,
                  inference_replicates=args.resamples, arms=methods, contrasts=contrasts)
    if external:
        seeds = [json.loads((run / 'run.json').read_text(encoding='utf-8'))['config']['seed'] for run in args.runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError('The external comparison requires distinct seeds.')
        report['seeds'] = seeds
        report['primary_contrast'] = 'independent_intensity_rps'
        report['class_counts'] = np.bincount(label, minlength=3).tolist()
    if args.comparison in ('visual-minimal', 'external-valence'):
        anchor = combine(packets, 'anchor', args.mode, 1)[:, 1:3]
        anchor_rps = ranked_probability_score(anchor, label).mean(axis=(0, 1))
        exploratory = paired_result(anchor_rps - clip_rps[candidate], membership, bootstrap, swap)
        report['anchor_comparison'] = dict(rps=float(anchor_rps.mean()), contrast=exploratory)
    if args.comparison in (*HEAD_COMPARISONS, 'visual-score', 'neutral-mixing'):
        per_clip = clip_rps[comparator] - clip_rps[candidate]
        report['rps_reduction_by_class'] = {
            name: dict(n=int((label == i).sum()), rps_reduction=float(per_clip[label == i].mean()))
            for i, name in enumerate(('negative', 'neutral', 'positive')) if (label == i).any()
        }
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
