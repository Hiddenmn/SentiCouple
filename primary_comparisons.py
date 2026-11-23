"""Reproduce sharing, score calibration, and architecture comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import (ARCHITECTURE_CONTROLS, ENCODER_CONTROLS, grouped_seed_video_interval,
                   ranked_probability_score, seed_video_interval)
from calibrate import scale_intensity
from cremad_effects import summarize as summarize_cremad
from prepare_ravdess import ACTOR_FOLDS
from ravdess_effects import estimate
from score_anchor_effects import SEEDS
from score_gain import reduction_intervals


def read_summary(runs, name):
    return json.loads((runs / name).read_text(encoding='utf-8'))


def ch_sims(runs):
    reference = read_summary(runs, 'sims_v2_compact_visual.json')
    target = reference['supervision']['versus_independent_rps']
    with np.load(runs / 'dispersion_pooling_sims_v2_evidence.npz', allow_pickle=False) as data:
        y, videos = data['y'], data['groups']
        np.testing.assert_array_equal(data['seeds'], SEEDS)
        _, membership = np.unique(videos, return_inverse=True)
        independent = ranked_probability_score(data['original_max_independent'], y).mean(1)
        shared = ranked_probability_score(data['original_max_shared'], y).mean(1)
    np.testing.assert_array_equal((len(y), np.unique(videos).size),
                                  (reference['n_clips'], reference['n_videos']))
    gain = independent - shared
    # Two supervision contrasts and two outcomes comprise the published family.
    tail = .025 / len(reference['supervision'])
    interval = seed_video_interval(gain, membership, reference['resamples'], quantiles=(tail, 1 - tail))
    np.testing.assert_allclose([gain.mean(), *interval], [target['reduction'], *target['ci95_familywise']],
                               atol=1e-9, rtol=0)
    return dict(metric='RPS', shared=float(shared.mean()), independent=float(independent.mean()),
                reduction=float(gain.mean()), interval=interval, family_size=len(reference['supervision']))


def ravdess(runs):
    reference = read_summary(runs, 'ravdess_effects.json')
    target = reference['primary_shared_versus_independent_balanced_rps']
    with np.load(runs / 'rater_precision_ravdess_evidence.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['seeds'], SEEDS)
        for fold, actors in enumerate(ACTOR_FOLDS):
            np.testing.assert_array_equal(np.unique(data[f'fold{fold}_groups']),
                                          sorted(f'actor{actor:02d}' for actor in actors))
        y, actors = (np.concatenate([data[f'fold{fold}_{key}'] for fold in range(len(ACTOR_FOLDS))])
                     for key in ('y', 'groups'))
        independent, shared = (
            ranked_probability_score(np.concatenate([data[f'fold{fold}_{head}'] for fold in range(len(ACTOR_FOLDS))],
                                                    axis=2), y).mean(1)
            for head in ('independent', 'shared')
        )
    names, membership = np.unique(actors, return_inverse=True)
    np.testing.assert_array_equal(names, reference['actors'])
    np.testing.assert_array_equal(np.bincount(y, minlength=3), reference['labels'])
    for index in range(len(names)):
        np.testing.assert_array_equal(np.bincount(y[membership == index], minlength=3), [32, 4, 8])
    weights = len(y) / (3 * np.bincount(y)[y])
    result = estimate(independent - shared, membership, reference['resamples'], weights)
    np.testing.assert_allclose([result['reduction'], *result['ci']], [target['reduction'], *target['ci']],
                               atol=1e-9, rtol=0)
    return dict(metric='balanced RPS', shared=float((shared * weights).mean()),
                independent=float((independent * weights).mean()), reduction=result['reduction'],
                interval=result['ci'], family_size=1)


def cremad(runs):
    reference = read_summary(runs, 'cremad_effects.json')
    with np.load(runs / 'cremad_evidence.npz', allow_pickle=False) as evidence:
        result = summarize_cremad(evidence, reference['resamples'])
    if result != reference:
        raise ValueError('CREMA-D estimates differ from the archived analysis.')
    effect = result['balanced_rps_reductions']['independent_intensity']
    methods = result['methods']
    return dict(metric='balanced RPS', shared=methods['intensity']['balanced_rps'],
                independent=methods['independent_intensity']['balanced_rps'], reduction=effect['reduction'],
                interval=effect['ci95_familywise'], family_size=1)


def score_calibration(runs):
    reference = read_summary(runs, 'score_gain_effects.json')
    tail = .025 / reference['family_size']
    summary = {}
    with np.load(runs / 'primary_score_calibration_evidence.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['seeds'], SEEDS)
        for corpus, expected in reference['cohorts'].items():
            prefix = f'{corpus}__'
            scale = float(data[prefix + 'scale'].item())
            target = data[prefix + 'sentiment']
            groups = data[prefix + 'groups']
            base = np.arctanh((data[prefix + 'anchor'] @ data[prefix + 'centers']).clip(-.9999, .9999)
                              .astype(np.float64))
            errors = {}
            for head in ('shared', 'independent'):
                score = data[prefix + head + '_score']
                gain = data[prefix + head + '_gain']
                np.testing.assert_array_equal(gain, expected['methods'][head]['gain_by_seed'])
                corrected = scale_intensity(base, np.arctanh(score.astype(np.float64) / scale),
                                            gain[:, None, None], scale)
                errors[head] = dict(old=np.abs(score - target), calibrated=np.abs(corrected - target))
            _, membership = np.unique(groups, return_inverse=True)
            effects = reduction_intervals(errors, membership, reference['resamples'], tail)
            for key, effect in effects.items():
                archived = expected['reductions'][key]
                np.testing.assert_allclose(
                    [effect['mae_reduction'], *effect['ci95_familywise']],
                    [archived['mae_reduction'], *archived['ci95_familywise']],
                    rtol=0, atol=4e-7)
            calibrated = effects['shared_old']
            summary[corpus] = dict(metric='MAE', reduction=calibrated['mae_reduction'],
                                   interval=calibrated['ci95_familywise'], family_size=reference['family_size'])
    return summary


def architecture(runs):
    reference = read_summary(runs, 'sims_v2_architecture_comparison.json')
    contrasts = {f'{arm}_rps': (arm, 'intensity') for arm in ARCHITECTURE_CONTROLS}
    unscaled, pointwise = ENCODER_CONTROLS
    contrasts['pointwise_vs_unscaled_rps'] = (pointwise, unscaled)
    resamples = reference['inference_replicates']
    tail = .025 / len(contrasts)
    summary = {}
    with np.load(runs / 'primary_architecture_losses.npz', allow_pickle=False) as data:
        counts = data['counts']
        np.testing.assert_array_equal(data['seeds'], SEEDS)
        np.testing.assert_array_equal((counts.sum(), len(counts)), (reference['n_clips'], reference['n_videos']))
        for name, (control, model) in contrasts.items():
            totals = data[control] - data[model]
            reduction = float(totals.sum(axis=1).mean() / counts.sum())
            marginal = grouped_seed_video_interval(totals, counts, resamples)
            adjusted = grouped_seed_video_interval(totals, counts, resamples, quantiles=(tail, 1 - tail))
            archived = reference['contrasts'][name]
            np.testing.assert_allclose([reduction, *marginal, *adjusted],
                                       [archived['rps_reduction'], *archived['ci95_seeds_and_videos'],
                                        *archived['ci95_familywise']], atol=2e-8, rtol=0)
            summary[name] = dict(reduction=reduction, interval=adjusted)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    args = parser.parse_args()
    result = dict(ch_sims_v2=ch_sims(args.runs), ravdess=ravdess(args.runs), cremad=cremad(args.runs),
                  score_calibration=score_calibration(args.runs), architecture=architecture(args.runs))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
