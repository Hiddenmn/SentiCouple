"""Evaluate frozen OMG-Emotion polarity predictors on the MOUD review corpus."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions, seed_video_interval
from compare_benchmarks import load_metadata
from coupling import rps_summary
from data import Clips, load_clips, observation_views, transcode
from networks import AudiovisualResidual, TextAnchor
from ordinal_controls import CONTROLS, factorial_contrasts
from risk_gate import fit_weights, mixtures, risk_coefficients, rps_weight
from score_anchor_effects import SEEDS
from train import (INDEPENDENT_SCORE_ARMS, NEUTRAL_READOUT_ARMS, BatchSource, anchor_outputs,
                   predict_fusion, visual_component_options)


ORIGINAL_ARMS = ('fusion', 'intensity', 'independent_intensity')
TRIAL_ARMS = ('intensity_capped', 'intensity_uniform')
ORDINAL_ARMS = tuple(CONTROLS.values())
GAPS = (.35, .65)


def check_configuration(reference, *comparators):
    config = reference[0]['config']
    if (config['split'] != 'official' or config['inputs'] != 'visual' or config['backbone'] != 'temporal'
            or config['visual_encoding'] != 'pointwise' or config['neutral_margin'] != .10):
        raise ValueError('The transfer uses the official OMG-Emotion pointwise visual checkpoints.')
    for collection in (reference, *comparators):
        if len(collection) != len(SEEDS):
            raise ValueError('Provide the source checkpoints for every paired seed.')
        for seed, record in zip(SEEDS, collection):
            current = record['config']
            if current['seed'] != seed:
                raise ValueError('Checkpoint training seeds must match the protocol order.')
            for key in config.keys() - {'seed', 'arms', 'output'}:
                if current.get(key) != config[key]:
                    raise ValueError(f'Checkpoint configuration differs for {key}.')
    return config


def matched_input(source_path, target_path, config, device, *, target_neutral_margin=0.):
    source, split = load_clips(source_path, config['bins'], 'official', config['split_seed'],
                               neutral_margin=config['neutral_margin'])
    with np.load(target_path, allow_pickle=False) as packet:
        target = transcode(packet, config['bins'], target_neutral_margin)
    if len(np.unique(target.ids)) != len(target.ids):
        raise ValueError('Target review utterance identifiers must be unique.')
    target_index = len(source.ids) + np.arange(len(target.ids))
    combined = Clips(**{key: np.concatenate((value, getattr(target, key))) for key, value in vars(source).items()})
    source_views = observation_views(source, GAPS, config['mask_seed'])
    target_views = observation_views(target, GAPS, 1777)
    masks = np.concatenate((source_views, target_views), axis=2)
    masks[:, 0] = False
    batches = BatchSource(combined, masks, split['fit'], device, visual_only=True,
                          stream_text=config['stream_text'])
    return batches, source.ids[split['eval']], split['eval'], target, target_index


def replay_seed(original, comparisons, config, batches, source_ids, source_index, target_index):
    device = batches.device
    anchor = TextAnchor(batches.text.shape[-1], config['bins'], config['width'], visual_only=True).to(device)
    anchor.load_state_dict(torch.load(original / 'anchor.pt', map_location=device, weights_only=True))
    saved = anchor_outputs(anchor, batches, config['batch'])
    joint_index = np.concatenate((source_index, target_index))
    n_source = len(source_index)
    source_anchor = saved[1][:, source_index].softmax(-1).cpu().numpy()
    anchor_probability = saved[1][:, target_index].softmax(-1).cpu().numpy()
    forecasts = {'anchor': anchor_probability}

    for run, arms in ((original, ORIGINAL_ARMS), *comparisons):
        original_packet = read_predictions(run)
        np.testing.assert_array_equal(original_packet['ids'], source_ids)
        np.testing.assert_array_equal(original_packet['masks'], batches.views[:, :, source_index].cpu().numpy())
        np.testing.assert_allclose(original_packet['anchor'], source_anchor, rtol=2e-5, atol=4e-6)
        for arm in arms:
            model = AudiovisualResidual(0, batches.vision.shape[-1], config['width'],
                                        independent_intensity=arm in INDEPENDENT_SCORE_ARMS,
                                        visual_encoding=config['visual_encoding'],
                                        neutral_readout=NEUTRAL_READOUT_ARMS.get(arm, 'softmax'),
                                        **visual_component_options(arm)).to(device)
            weights = torch.load(run / f'{arm}.pt', map_location=device, weights_only=True)
            model.load_state_dict(weights)
            probability, _, _, _ = predict_fusion(model, batches, joint_index, saved, config['batch'], range(4))
            np.testing.assert_allclose(original_packet[arm], probability[:, :n_source], rtol=3e-5, atol=6e-6)
            forecasts[arm] = probability[:, n_source:]
            np.testing.assert_allclose(forecasts[arm][-1], anchor_probability[-1], rtol=2e-6, atol=1e-7)
    return forecasts


def report_transfer(runs, target, resamples):
    probabilities = {key: np.stack([run[key] for run in runs]) for key in runs[0]}
    per_view = {key: ranked_probability_score(values, target.label) for key, values in probabilities.items()}
    gap_loss = {key: values[:, 1:3].mean(axis=1) for key, values in per_view.items()}
    methods = {key: dict(gapped=rps_summary(values[:, 1:3], target.label),
                         native=rps_summary(values[:, :1], target.label),
                         rps_by_gap=per_view[key][:, 1:3].mean(axis=(0, 2)).tolist())
               for key, values in probabilities.items()}
    shared_gain = gap_loss['independent_intensity'] - gap_loss['intensity']
    native_gain = per_view['independent_intensity'][:, 0] - per_view['intensity'][:, 0]
    source_groups, membership = np.unique(target.groups, return_inverse=True)
    secondary = dict(versus_anchor=gap_loss['anchor'] - gap_loss['intensity'],
                     gaps_minus_native=shared_gain - native_gain,
                     clipping=gap_loss['intensity'] - gap_loss['intensity_capped'])

    def estimate(values, tail):
        return dict(rps_reduction=float(values.mean()), by_seed=values.mean(axis=1).tolist(),
                    ci95=seed_video_interval(values, membership, resamples, quantiles=(tail, 1 - tail)))

    result = dict(seeds=SEEDS, n_clips=len(target.ids), n_videos=len(source_groups), methods=methods,
                  label_counts=np.bincount(target.label, minlength=3).tolist(),
                  primary_shared_versus_independent=estimate(shared_gain, .025),
                  secondary={key: estimate(value, .025 / len(secondary)) for key, value in secondary.items()})
    if all(arm in gap_loss for arm in ORDINAL_ARMS):
        result['ordinal_controls'] = {
            name: estimate(gap_loss[arm] - gap_loss['intensity'], .025 / len(ORDINAL_ARMS))
            for name, arm in CONTROLS.items()
        }
    if 'independent_class_mean' in gap_loss:
        contrasts = factorial_contrasts(gap_loss['intensity'], gap_loss['independent_intensity'],
                                        gap_loss['intensity_class_mean'], gap_loss['independent_class_mean'])
        result['class_mean_factorial'] = {name: estimate(value, .025 / len(contrasts))
                                          for name, value in contrasts.items()}
    return result


def diagnose_anchor_mixture(runs, original, label):
    weights, transferred = [], []
    for run, checkpoint in zip(runs, original):
        with np.load(checkpoint / 'tuning.npz', allow_pickle=False) as tuning:
            coefficients = fit_weights(tuning['anchor'][1:3], tuning['intensity'][1:3], tuning['y'])
        weights.append(coefficients)
        transferred.append(mixtures(run['anchor'][1:3], run['intensity'][1:3], coefficients))
    methods = {name: np.stack([result[name] for result in transferred]) for name in transferred[0]}
    quadratic, linear = risk_coefficients(methods['anchor'], methods['shared'], label)
    return dict(source_tuning_weights=weights,
                target_mean_linear=float(linear.mean()), target_mean_quadratic=float(quadratic.mean()),
                target_descriptive_minimizer=rps_weight(quadratic, linear),
                scores={name: rps_summary(values, label) for name, values in methods.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--reference-prefix', required=True)
    parser.add_argument('--trial-prefix', required=True)
    parser.add_argument('--ordinal-prefix', help='Also transfer the matched polarity and class mean controls.')
    parser.add_argument('--independent-class-mean-prefix', help='Complete the class mean head comparison.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose at least one bootstrap resample.')
    if args.independent_class_mean_prefix and not args.ordinal_prefix:
        parser.error('The class mean comparison requires --ordinal-prefix.')

    original = [Path(f'{args.reference_prefix}{seed}') for seed in SEEDS]
    trial_groups = [(args.trial_prefix, TRIAL_ARMS)]
    if args.ordinal_prefix:
        trial_groups.append((args.ordinal_prefix, ORDINAL_ARMS))
    if args.independent_class_mean_prefix:
        trial_groups.append((args.independent_class_mean_prefix, ('independent_class_mean',)))
    comparator_paths = [([Path(f'{prefix}{seed}') for seed in SEEDS], arms)
                        for prefix, arms in trial_groups]
    config = check_configuration(load_metadata(original), *(load_metadata(paths) for paths, _ in comparator_paths))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    batches, source_ids, source_index, target, target_index = matched_input(config['data'], args.target, config, device)
    forecasts = []
    for index, seed in enumerate(SEEDS):
        comparisons = [(paths[index], arms) for paths, arms in comparator_paths]
        forecasts.append(replay_seed(original[index], comparisons, config, batches, source_ids,
                                     source_index, target_index))
        print(f'Matched source and transferred frozen forecasts for seed {seed}.', flush=True)
    report = report_transfer(forecasts, target, args.resamples)
    report['exploratory_anchor_mixture'] = diagnose_anchor_mixture(forecasts, original, target.label)
    with args.target.open('rb') as stream:
        report['feature_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    displayed = dict(primary=report['primary_shared_versus_independent'], secondary=report['secondary'])
    for key in ('ordinal_controls', 'class_mean_factorial'):
        if key in report:
            displayed[key] = report[key]
    print(json.dumps(displayed, indent=2), flush=True)


if __name__ == '__main__':
    main()
