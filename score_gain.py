"""Fit a tuning-only scale for the frozen shift in each intensity head."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import read_predictions, seed_video_interval
from calibrate import fit_intensity_gain, scale_intensity
from compare_benchmarks import load_metadata
from data import load_clips, observation_views
from gap_sensitivity import MASK_SEEDS, RATES
from networks import AudiovisualResidual, TextAnchor, intensity_anchor
from score_anchor_effects import HEADS, SEEDS
from train import BatchSource, anchor_outputs, predict_fusion


def readout_predictions(run, config, source, split, archived, score):
    anchor = TextAnchor(source.text.shape[-1], config['bins'], config['width'], visual_only=True).to(source.device)
    anchor.load_state_dict(torch.load(run / 'anchor.pt', map_location=source.device, weights_only=True))
    saved = anchor_outputs(anchor, source, config['batch'])
    tune, test = split['tune'], split['eval']
    class_base = intensity_anchor(saved[1], source.class_centers).cpu().numpy()
    scale = source.intensity_scale
    result = {}
    with np.load(run / 'tuning.npz', allow_pickle=False) as tuning:
        np.testing.assert_array_equal(tuning['y'], source.label[tune].cpu().numpy())
        np.testing.assert_allclose(saved[1][:4, tune].softmax(-1).cpu().numpy(), tuning['anchor'],
                                   rtol=2e-5, atol=4e-6)
        for head, arm in HEADS.items():
            model = AudiovisualResidual(0, source.vision.shape[-1], config['width'], attention_bins=config['bins'],
                                        independent_intensity=head == 'independent',
                                        text_query=config['backbone'] == 'attention',
                                        visual_encoding=config['visual_encoding']).to(source.device)
            model.load_state_dict(torch.load(run / f'{arm}.pt', map_location=source.device, weights_only=True))
            tuner, _, _, tune_logit = predict_fusion(model, source, tune, saved, config['batch'], (1, 2),
                                                    return_score_logits=True)
            np.testing.assert_allclose(tuner, tuning[arm][1:3], rtol=2e-5, atol=4e-6)
            gain = fit_intensity_gain(class_base[1:3, tune], tune_logit, score[tune], scale)
            predicted, _, _, eval_logit = predict_fusion(
                model, source, test, saved, config['batch'], range(len(source.views)), return_score_logits=True)
            np.testing.assert_allclose(predicted[:4], archived[arm], rtol=2e-5, atol=4e-6)
            original = archived[f'{arm}_score']
            raw = np.tanh(eval_logit) * scale
            np.testing.assert_allclose(raw[:4], original, rtol=2e-5, atol=8e-6)
            raw[:4] = original
            corrected = raw if gain == 1. else scale_intensity(class_base[:, test], eval_logit, gain, scale)
            np.testing.assert_allclose(corrected[3], original[3], rtol=2e-5, atol=8e-6)
            result[head] = dict(gain=gain, old=np.abs(original[1:3] - score[test]),
                                calibrated=np.abs(corrected[1:3] - score[test]))
            if len(source.views) > 4:
                shape = (len(MASK_SEEDS), len(RATES), len(test))
                result[head]['old_new_gaps'] = np.abs(raw[4:] - score[test]).reshape(shape).mean(axis=0)
                result[head]['calibrated_new_gaps'] = np.abs(corrected[4:] - score[test]).reshape(shape).mean(axis=0)
    return result


def reduction_intervals(errors, membership, resamples, tail):
    reductions = {}
    for name, comparator in [('shared_old', errors['shared']['old']),
                             ('independent_calibrated', errors['independent']['calibrated'])]:
        difference = comparator - errors['shared']['calibrated']
        average = difference.mean(axis=1)
        reductions[name] = dict(mae_reduction=float(average.mean()), by_seed=average.mean(axis=1).tolist(),
                                per_width=difference.mean(axis=(0, 2)).tolist(),
                                ci95_seeds_and_videos=seed_video_interval(average, membership, resamples),
                                ci95_familywise=seed_video_interval(average, membership, resamples,
                                                                    quantiles=(tail, 1 - tail)))
    return reductions


def cohort_experiment(paths, seeds, device, resamples, tail, new_gaps):
    configs = [entry['config'] for entry in load_metadata(paths)]
    config = configs[0]
    keys = ('data', 'split', 'split_seed', 'mask_seed', 'group_reference', 'eval_groups', 'exclude_eval',
            'bins', 'batch', 'width', 'inputs', 'backbone', 'visual_encoding', 'stream_text')
    for seed, current in zip(seeds, configs):
        if (current['seed'] != seed or current['inputs'] != 'visual' or current['backbone'] == 'temporal' or
                current.get('score_anchor', 'class-mean') != 'class-mean' or
                current.get('calibrate_score_gain', False) or
                any(current.get(key) != config.get(key) for key in keys) or
                current.get('neutral_margin', 0.) != config.get('neutral_margin', 0.)):
            raise ValueError('Use matched visual checkpoints with class-mean anchors.')
    clips, split = load_clips(config['data'], config['bins'], config['split'], config['split_seed'],
                              group_reference=config.get('group_reference'), eval_groups=config.get('eval_groups'),
                              exclude_eval=config.get('exclude_eval'), neutral_margin=config.get('neutral_margin', 0.))
    views = observation_views(clips, [.35, .65], config['mask_seed'])
    if new_gaps:
        additional = [observation_views(clips, RATES, seed)[1:-1] for seed in MASK_SEEDS]
        views = np.concatenate([views, *additional])
    views[:, 0] = False
    source = BatchSource(clips, views, split['fit'], device, visual_only=True, stream_text=config['stream_text'])
    test = split['eval']
    matched = dict(ids=clips.ids[test], groups=clips.groups[test], y=clips.label[test],
                   sentiment=clips.sentiment[test], masks=views[:4, :, test])
    runs = []
    for path, current in zip(paths, configs):
        archived = read_predictions(path)
        for key, expected in matched.items():
            np.testing.assert_array_equal(archived[key], expected, err_msg=key)
        runs.append(readout_predictions(path, current, source, split, archived, clips.sentiment))
        print(f'{path.name}: exact inputs and archived forecasts reconciled', flush=True)
    methods = {}
    errors = {}
    for head in HEADS:
        errors[head] = {setting: np.stack([run[head][setting] for run in runs]) for setting in ('old', 'calibrated')}
        methods[head] = dict(gain_by_seed=[run[head]['gain'] for run in runs],
                             mae={setting: float(values.mean()) for setting, values in errors[head].items()},
                             mae_by_gap={setting: values.mean(axis=(0, 2)).tolist()
                                         for setting, values in errors[head].items()})
    _, membership = np.unique(matched['groups'], return_inverse=True)
    summary = dict(n_clips=len(test), n_videos=int(membership.max()) + 1, methods=methods,
                   reductions=reduction_intervals(errors, membership, resamples, tail))
    if new_gaps:
        errors = {head: {setting: np.stack([run[head][f'{setting}_new_gaps'] for run in runs])
                         for setting in ('old', 'calibrated')} for head in HEADS}
        summary['new_gaps'] = dict(mask_seeds=MASK_SEEDS, widths=RATES,
                                   reductions=reduction_intervals(errors, membership, resamples, tail))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', action='append', nargs=2, required=True, metavar=('NAME', 'RUN_PREFIX'))
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--new-gaps', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or args.resamples < 1 or
            len({row[0] for row in args.cohort}) != len(args.cohort)):
        parser.error('Supply distinct cohort names, at least two paired seeds and positive resampling.')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    family = 2 * len(args.cohort)
    reports = {}
    for name, prefix in args.cohort:
        paths = [Path(f'{prefix}{seed}') for seed in args.seeds]
        reports[name] = cohort_experiment(paths, args.seeds, torch.device(args.device), args.resamples, .025 / family,
                                          args.new_gaps)
    report = dict(seeds=list(args.seeds), resamples=args.resamples, family_size=family, cohorts=reports)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
