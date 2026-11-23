"""Compare held-out forecasts made on identical source videos and masks."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import adjust_holm, paired_result, ranked_probability_score, read_predictions, seed_video_interval
from aggregate import f1_by_run
from coupling import rps_summary, video_resamples


def summarize(packets, probability_key, score_key, y, intensity):
    forecast = np.stack([packet[probability_key][1:3] for packet in packets])
    scores = np.stack([packet[score_key][1:3] for packet in packets])
    return summarize_forecasts(forecast, scores, y, intensity)


def summarize_forecasts(forecast, scores, y, intensity):
    rps = ranked_probability_score(forecast, y)
    mae = np.abs(scores - intensity)
    summary = dict(**rps_summary(forecast, y), mae=float(mae.mean()),
                   macro_f1=float(f1_by_run(y, forecast).mean()),
                   rps_by_gap=rps.mean(axis=(0, 2)).tolist(), rps_by_seed=rps.mean(axis=(1, 2)).tolist())
    return summary, rps.mean(axis=1), mae.mean(axis=1)


def load_matched(paths, reference):
    """Align by seed at the caller and require exactly the same evaluation clips."""
    packets = [read_predictions(path) for path in paths]
    for packet in packets:
        for key in ('ids', 'groups', 'y', 'sentiment', 'masks'):
            np.testing.assert_array_equal(packet[key], reference[key], err_msg=key)
    return packets


def load_metadata(paths):
    return [json.loads((path / 'run.json').read_text(encoding='utf-8')) for path in paths]


def describe_published(summary, metadata):
    summary['selected_epochs'] = [item['selection']['epoch'] for item in metadata]
    summary['trainable_parameters'] = [item['trainable_parameters'] for item in metadata]
    summary['upstream_revision'] = metadata[0]['upstream_revision']
    if 'frozen_parameters' in metadata[0]:
        summary['frozen_parameters'] = [item['frozen_parameters'] for item in metadata]


def summarize_run(packets, metadata, arm, y, intensity):
    if 'posterior' in packets[0]:
        summary, rps, mae = summarize(packets, 'posterior', 'score', y, intensity)
        describe_published(summary, metadata)
    else:
        summary, rps, mae = summarize(packets, arm, f'{arm}_score', y, intensity)
        summary['selected_epochs'] = [meta['selection'][arm]['epoch'] for meta in metadata]
        summary['fusion_parameters'] = [meta['selection'][arm]['parameters'] for meta in metadata]
    return summary, rps, mae


def seed_clip_accuracy(packets, arm, label):
    probability_key = 'posterior' if 'posterior' in packets[0] else arm
    correct = np.stack([packet[probability_key][1:3].argmax(-1) == label for packet in packets])
    return correct.mean(axis=1)


def verify_anchored_pair(shared, independent):
    for left, right in zip(shared, independent):
        if right['config']['head'] != 'anchored-independent':
            continue
        for key, value in left['config'].items():
            if key not in ('head', 'output') and right['config'].get(key) != value:
                raise ValueError(f'Anchored heads must be trained with the same {key}.')
        if left['upstream_revision'] != right['upstream_revision']:
            raise ValueError('Anchored heads must use the same upstream revision.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path, nargs='+', required=True)
    parser.add_argument('--baseline', action='append', nargs='+', metavar='NAME_OR_DIRECTORY', required=True)
    parser.add_argument('--candidate', default='intensity')
    parser.add_argument('--local-heads', type=Path, nargs='+',
                        help='Compare the anchored-head contrast with the same two heads on the local encoder.')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--accuracy-family-size', type=int,
                        help='Number of paired accuracy contrasts in a shared comparison family.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 1 or any(len(row) != len(args.shared) + 1 for row in args.baseline):
        parser.error('Supply a name and one matched directory per shared seed for each baseline.')
    names = [row[0] for row in args.baseline]
    if len(names) != len(set(names)) or args.candidate in names:
        parser.error('Method names must be distinct.')
    accuracy_family = len(names) if args.accuracy_family_size is None else args.accuracy_family_size
    if accuracy_family < len(names):
        parser.error('The accuracy family must include every supplied baseline.')
    if args.local_heads and len(args.local_heads) != len(args.shared):
        parser.error('Supply one matching local model run for each anchored-head pair.')

    reference = read_predictions(args.shared[0])
    packets = load_matched(args.shared, reference)
    y, intensity, groups = (reference[key] for key in ('y', 'sentiment', 'groups'))
    shared_meta = load_metadata(args.shared)
    seeds = [meta['config']['seed'] for meta in shared_meta]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate shared training seeds.')
    cluster_samples = video_resamples(groups, args.resamples)
    candidate, rps, mae = summarize_run(packets, shared_meta, args.candidate, y, intensity)
    accuracy = seed_clip_accuracy(packets, args.candidate, y)
    methods = {args.candidate: candidate}
    contrasts, accuracy_contrasts = {}, {}
    local_gains = None
    if args.local_heads:
        if (shared_meta[0]['config'].get('head') != 'anchored-shared' or
                [meta['config']['seed'] for meta in load_metadata(args.local_heads)] != seeds):
            parser.error('Local heads must match an anchored shared candidate in seed order.')
        local = load_matched(args.local_heads, reference)
        _, shared_rps, shared_mae = summarize(local, 'intensity', 'intensity_score', y, intensity)
        _, independent_rps, independent_mae = summarize(local, 'independent_intensity',
                                                        'independent_intensity_score', y, intensity)
        local_gains = {'rps': independent_rps - shared_rps, 'mae': independent_mae - shared_mae}
    for name, *directories in args.baseline:
        paths = list(map(Path, directories))
        meta = load_metadata(paths)
        if [item['config']['seed'] for item in meta] != seeds:
            raise ValueError(f'{name} must follow the order of the shared seeds.')
        if shared_meta[0]['config'].get('head') == 'anchored-shared':
            verify_anchored_pair(shared_meta, meta)
        baseline_packets = load_matched(paths, reference)
        baseline, base_rps, base_mae = summarize_run(baseline_packets, meta, 'intensity', y, intensity)
        methods[name] = baseline
        accuracy_difference = accuracy - seed_clip_accuracy(baseline_packets, 'intensity', y)
        accuracy_effect = dict(difference=float(accuracy_difference.mean()))
        if len(seeds) > 1:
            tail = .025 / accuracy_family
            accuracy_effect['ci95_familywise'] = seed_video_interval(accuracy_difference, cluster_samples[0],
                                                                     args.resamples, quantiles=(tail, 1 - tail))
        accuracy_contrasts[name] = accuracy_effect
        for metric, values in [('rps', base_rps - rps), ('mae', base_mae - mae)]:
            effect = paired_result(values.mean(axis=0), *cluster_samples, effect=f'{metric}_reduction')
            effect['reduction_by_seed'] = values.mean(axis=-1).tolist()
            if len(seeds) > 1:
                effect['ci95_seeds_and_videos'] = seed_video_interval(values, cluster_samples[0], args.resamples)
            contrasts[f'{name}_{metric}'] = effect
            if local_gains is not None and meta[0]['config'].get('head') == 'anchored-independent':
                interaction = local_gains[metric] - values
                difference = paired_result(interaction.mean(axis=0), *cluster_samples,
                                           effect=f'{metric}_reduction_difference')
                difference['reduction_by_seed'] = interaction.mean(axis=-1).tolist()
                if len(seeds) > 1:
                    difference['ci95_seeds_and_videos'] = seed_video_interval(
                        interaction, cluster_samples[0], args.resamples)
                contrasts[f'{name}_{metric}_local_minus_published'] = difference
    adjust_holm(contrasts)
    result = dict(n_clips=len(y), n_videos=int(len(np.unique(groups))), seeds=seeds,
                  inference_replicates=args.resamples, methods=methods, contrasts=contrasts,
                  accuracy_contrasts=accuracy_contrasts, accuracy_family_size=accuracy_family)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for name, metrics in methods.items():
        print(f'{name:21s} RPS={metrics["rps"]:.6f} MAE={metrics["mae"]:.6f} macro-F1={metrics["macro_f1"]:.4f}')
    for name, contrast in contrasts.items():
        print(name, contrast)


if __name__ == '__main__':
    main()
