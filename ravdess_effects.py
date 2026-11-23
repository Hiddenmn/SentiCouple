"""Compare paired RAVDESS forecasts from held-out actors."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import ranked_probability_score, read_predictions, seed_video_interval
from prepare_ravdess import ACTOR_FOLDS
from score_anchor_effects import SEEDS


HEADS = ('anchor', 'fusion', 'intensity', 'independent_intensity', 'intensity_shuffled')
CLASS_MEAN_HEADS = ('intensity_class_mean', 'independent_class_mean')
MATCH_KEYS = ('ids', 'groups', 'y', 'sentiment', 'masks')


def matched_forecasts(root, class_means=False):
    fits = []
    for seed in SEEDS:
        folds = []
        for fold, actors in enumerate(ACTOR_FOLDS):
            run = root / f'ravdess_fold{fold}_seed{seed}'
            packet = read_predictions(run)
            metadata = json.loads((run / 'run.json').read_text(encoding='utf-8'))
            if metadata['config']['seed'] != seed or set(packet['groups']) != {f'actor{a:02d}' for a in actors}:
                raise ValueError(f'Wrong training seed or held-out actors for {run}.')
            if fits:
                for key in MATCH_KEYS:
                    np.testing.assert_array_equal(packet[key], fits[0][fold][key], err_msg=f'{run}: {key}')
            if class_means:
                coarse = root / f'ravdess_class_mean_fold{fold}_seed{seed}'
                control = read_predictions(coarse)
                settings = json.loads((coarse / 'run.json').read_text(encoding='utf-8'))['config']
                original = metadata['config']
                for key in (settings.keys() | original.keys()) - {'arms', 'output'}:
                    if settings.get(key) != original.get(key):
                        raise ValueError(f'{coarse} has a different setting for {key}.')
                for key in (*MATCH_KEYS, 'anchor'):
                    np.testing.assert_array_equal(control[key], packet[key], err_msg=f'{coarse}: {key}')
                for name in (*CLASS_MEAN_HEADS, *(f'{arm}_score' for arm in CLASS_MEAN_HEADS),
                             'intensity_class_mean_uncalibrated_score'):
                    packet[name] = control[name]
            folds.append(packet)
        fits.append(folds)
    return fits


def losses(fits, labels, scores, class_means=False):
    rps, mae = {}, {}
    heads = (*HEADS, *CLASS_MEAN_HEADS) if class_means else HEADS
    uncalibrated = ('intensity_uncalibrated', 'intensity_class_mean_uncalibrated') if class_means else (
        'intensity_uncalibrated',)
    for head in heads:
        probabilities = [np.concatenate([fold[head][1:3] for fold in folds], axis=1) for folds in fits]
        rps[head] = ranked_probability_score(np.stack(probabilities), labels).mean(axis=1)
    for name in (*heads, *uncalibrated):
        intensities = [np.concatenate([fold[f'{name}_score'][1:3] for fold in folds], axis=1) for folds in fits]
        mae[name] = np.abs(np.stack(intensities) - scores).mean(axis=1)
    return rps, mae


def estimate(gain, actor_index, resamples, weights=1., tail=.025):
    weighted = gain * weights
    counts = np.bincount(actor_index)
    actor_means = np.bincount(actor_index, weights=weighted.mean(axis=0)) / counts
    return dict(reduction=float(weighted.mean()), by_seed=weighted.mean(axis=1).tolist(),
                by_actor=actor_means.tolist(),
                ci=seed_video_interval(weighted, actor_index, resamples, quantiles=(tail, 1 - tail)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--class-means', action='store_true', help='Include the matched class mean heads.')
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose at least one bootstrap resample.')

    fits = matched_forecasts(args.runs, args.class_means)
    ids, actors, labels, scores = [np.concatenate([fold[key] for fold in fits[0]])
                                   for key in ('ids', 'groups', 'y', 'sentiment')]
    names, actor_index = np.unique(actors, return_inverse=True)
    if len(ids) != 1056 or len(np.unique(ids)) != len(ids) or len(names) != 24:
        raise ValueError('Each of the 1,056 trials must have exactly one held-out prediction per seed.')
    counts = np.stack([np.bincount(labels[actor_index == i], minlength=3) for i in range(len(names))])
    np.testing.assert_array_equal(counts, np.broadcast_to([32, 4, 8], counts.shape))
    weights = len(labels) / (3 * counts.sum(0)[labels])
    rps, mae = losses(fits, labels, scores, args.class_means)
    methods = {name: dict(rps=float(values.mean()), balanced_rps=float((values * weights).mean()),
                          mae=float(mae[name].mean()),
                          rps_by_class=[float(values[:, labels == c].mean()) for c in range(3)])
               for name, values in rps.items()}
    methods['intensity_uncalibrated'] = dict(mae=float(mae['intensity_uncalibrated'].mean()))

    shared = rps['intensity']
    primary = estimate(rps['independent_intensity'] - shared, actor_index, args.resamples, weights)
    tail = .025 / 3
    secondary = dict(
        versus_anchor_balanced_rps=estimate(rps['anchor'] - shared, actor_index, args.resamples, weights, tail),
        versus_shuffle_balanced_rps=estimate(rps['intensity_shuffled'] - shared,
                                             actor_index, args.resamples, weights, tail),
        calibration_mae=estimate(mae['intensity_uncalibrated'] - mae['intensity'],
                                 actor_index, args.resamples, tail=tail))
    class_only = estimate(rps['fusion'] - shared, actor_index, args.resamples, weights)
    result = dict(clips=len(ids), actors=names.tolist(), actor_folds=ACTOR_FOLDS, seeds=SEEDS,
                  labels=counts.sum(0).tolist(), resamples=args.resamples, methods=methods,
                  primary_shared_versus_independent_balanced_rps=primary, secondary=secondary,
                  descriptive_shared_versus_class_only_balanced_rps=class_only)
    contrasts = {'shared_versus_independent': primary, **secondary, 'versus_class_only': class_only}
    if args.class_means:
        shared_coarse = rps['intensity_class_mean']
        coarse_gain = rps['independent_class_mean'] - shared_coarse
        coarse_primary = estimate(coarse_gain, actor_index, args.resamples, weights)
        coarse_secondary = dict(
            change_in_sharing_gain=estimate(coarse_gain - (rps['independent_intensity'] - shared),
                                             actor_index, args.resamples, weights, tail),
            versus_anchor_balanced_rps=estimate(rps['anchor'] - shared_coarse, actor_index,
                                                args.resamples, weights, tail),
            calibration_mae=estimate(mae['intensity_class_mean_uncalibrated'] - mae['intensity_class_mean'],
                                     actor_index, args.resamples, tail=tail))
        result['class_mean'] = dict(primary=coarse_primary, secondary=coarse_secondary)
        methods['intensity_class_mean_uncalibrated'] = dict(mae=float(mae['intensity_class_mean_uncalibrated'].mean()))
        contrasts.update({f'class_mean_{key}': value for key, value in {'primary': coarse_primary,
                                                                       **coarse_secondary}.items()})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: {'reduction': effect['reduction'], 'ci': effect['ci']}
                      for name, effect in contrasts.items()}, indent=2))


if __name__ == '__main__':
    main()
