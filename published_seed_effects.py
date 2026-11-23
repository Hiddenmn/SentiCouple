"""Compare the shared heads with published fusion cores over matched video and seed splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from calibrate import fit_temperature, log_loss, temperature_scale
from compare_benchmarks import describe_published, load_matched, load_metadata, summarize_forecasts
from coupling import rps_summary
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_baselines import UPSTREAM
from score_anchor_effects import SEEDS


BASELINES = {'tfr_joint': 'tfr-net', 'lnln_joint': 'lnln', 'rmnet_core': 'rm-net'}
CANDIDATES = {'fixed_pooling': 'sims_v2_compact_pooled', 'attention': 'sims_v2_compact_attention'}
BENCHMARK_CANDIDATES = {'sims_v2': CANDIDATES, 'omg': {'pointwise': 'omg_pointwise_heads'}}


def run_paths(root, seeds, cohort='sims_v2'):
    paths = {name: [root / f'{prefix}_seed{seed}' for seed in seeds]
             for name, prefix in BENCHMARK_CANDIDATES[cohort].items()}
    for name in BASELINES:
        paths[name] = [root / f'{cohort}_{name}_seed{seed}' for seed in seeds]
    if cohort == 'sims_v2' and 97 in seeds:
        paths['tfr_joint'][seeds.index(97)] = root / 'sims_v2_tfr_joint_seed97_cuda'
    return paths


def verify_training(name, metadata, seeds, cohort='sims_v2', *, inputs='visual'):
    if [row['config']['seed'] for row in metadata] != list(seeds):
        raise ValueError(f'{name} must match the paired training seeds.')
    if name not in BASELINES:
        backbone = 'temporal' if name == 'fixed_pooling' else 'stream-query'
        settings = dict(backbone=backbone, visual_encoding='unscaled', split='new-videos', inputs=inputs,
                        calibrate_score_gain=True, split_seed=1729, mask_seed=1777)
        if cohort == 'omg':
            settings.update(backbone='temporal', visual_encoding='pointwise', split='official', neutral_margin=.10,
                            selection='rps', score_anchor='class-mean', device='cuda', eval_groups=None)
        if any(row['config'][key] != value for row in metadata for key, value in settings.items()):
            raise ValueError(f'{name} has a checkpoint with different training settings.')
        return
    model = BASELINES[name]
    settings = dict(model=model, head='joint', corpus='sims', split='new-videos',
                    inputs=inputs, device='cuda')
    if cohort == 'omg':
        settings.update(corpus='omg', split='official', neutral_margin=.10, eval_groups=None)
    if model == 'rm-net':
        settings.update(epochs=48, patience=10, batch=64, lr=.0001, weight_decay=.001)
    else:
        settings.update(epochs=32, patience=8, batch=32, lr=.001, weight_decay=.01)
    for row, seed in zip(metadata, seeds, strict=True):
        if row['upstream_revision'] != UPSTREAM[model] or any(
                row['config'][key] != value for key, value in settings.items()):
            raise ValueError(f'{name} has a checkpoint with different training settings.')
        if cohort == 'omg':
            reference = Path(row['config']['match_run'])
            expected = f'omg_pointwise_heads_seed{seed}'
            if reference.name != expected:
                raise ValueError(f'{name} must verify its inputs against {expected}.')
            history = row['tune_rps_by_epoch']
            if row['selection']['epoch'] != int(np.argmin(history)) + 1 or row['selection']['rps'] != min(history):
                raise ValueError(f'{name} seed {seed} must use the best tuning RPS checkpoint.')


def scaled_risk(tuning, forecast, tuning_labels, label):
    temperatures, tune_losses, heldout_losses, risks = [], [], [], []
    for tune, raw in zip(tuning.astype(np.float64), forecast.astype(np.float64)):
        temperature = fit_temperature(tune, tuning_labels)
        temperatures.append(float(np.exp(temperature)))
        tune_losses.append(log_loss(temperature_scale(tune, temperature), tuning_labels))
        calibrated = temperature_scale(raw, temperature)
        np.testing.assert_array_equal(calibrated.argmax(-1), raw.argmax(-1))
        heldout_losses.append(log_loss(calibrated, label))
        risks.append(ranked_probability_score(calibrated, label).mean(axis=0))
    return np.stack(risks), dict(temperature=temperatures, tuning_log_loss=float(np.mean(tune_losses)),
                                 heldout_log_loss=float(np.mean(heldout_losses)))


def count_anchor(path):
    parameters = torch.load(path / 'anchor.pt', map_location='cpu', weights_only=True)
    return sum(value.numel() for value in parameters.values())


def collect_forecasts(root, seeds, cohort='sims_v2'):
    paths = run_paths(root, seeds, cohort)
    candidates = BENCHMARK_CANDIDATES[cohort]
    reference_path = paths[next(iter(candidates))][0]
    reference = read_predictions(reference_path)
    with np.load(reference_path / 'tuning.npz', allow_pickle=False) as tune:
        tuning_labels = tune['y']
    records, probabilities, scores, tuning = {}, [], [], []
    extra = {}
    for name, runs in paths.items():
        metadata = load_metadata(runs)
        verify_training(name, metadata, seeds, cohort)
        packets = load_matched(runs, reference)
        if name in candidates:
            key, score_key = 'intensity', 'intensity_score'
            selected = [row['selection']['intensity'] for row in metadata]
            counts = [row['parameters'] for row in selected]
            record = dict(selected_epochs=[row['epoch'] for row in selected], fusion_parameters=counts,
                          trainable_parameters=[count + count_anchor(run) for count, run in zip(counts, runs)],
                          intensity_multiplier=[row['score_gain'] for row in selected])
            for packet in packets:
                np.testing.assert_array_equal(packet[key][-1], packet['anchor'][-1])
            if cohort == 'omg':
                extra = dict(cohort=np.asarray(cohort), anchor=np.stack([packet['anchor'][1:3] for packet in packets]))
        else:
            key, score_key = 'posterior', 'score'
            record = {}
            describe_published(record, metadata)
        forecast = np.stack([packet[key][1:3] for packet in packets])
        if name in BASELINES:
            risk = ranked_probability_score(forecast, reference['y']).mean(axis=(1, 2))
            np.testing.assert_allclose(risk, [row['heldout_rps'] for row in metadata], rtol=2e-6, atol=1e-7)
        tune_probabilities = []
        for run in runs:
            with np.load(run / 'tuning.npz', allow_pickle=False) as packet:
                np.testing.assert_array_equal(packet['y'], tuning_labels, err_msg=run.name)
                tune_probabilities.append(packet[key][1:3])
        if cohort == 'omg' and name in BASELINES:
            tune_rps = ranked_probability_score(np.stack(tune_probabilities), tuning_labels).mean(axis=(1, 2))
            np.testing.assert_allclose(tune_rps, [row['selection']['rps'] for row in metadata], rtol=2e-6, atol=1e-7)
        records[name] = record
        probabilities.append(forecast)
        scores.append(np.stack([packet[score_key][1:3] for packet in packets]))
        tuning.append(np.stack(tune_probabilities))
    population = {key: reference[key] for key in ('ids', 'groups', 'y', 'sentiment', 'masks')}
    return dict(**population, **extra, tuning_labels=tuning_labels, seeds=np.asarray(seeds), methods=np.asarray(list(paths)),
                records=np.asarray(json.dumps(records)), forecast=np.stack(probabilities),
                score=np.stack(scores), tuning=np.stack(tuning))


def loss_summaries(data):
    labels, intensity, tuning_labels = data['y'], data['sentiment'], data['tuning_labels']
    names = data['methods'].tolist()
    records = json.loads(data['records'].item())
    methods, risks, errors, correct, scaled = {}, {}, {}, {}, {}
    for name, forecast, score, tune in zip(names, data['forecast'], data['score'], data['tuning'], strict=True):
        method, risks[name], errors[name] = summarize_forecasts(forecast, score, labels, intensity)
        correct[name] = (forecast.argmax(-1) == labels).mean(axis=1)
        scaled[name], calibration = scaled_risk(tune, forecast, tuning_labels, labels)
        components = (np.square(coordinates(forecast) - outcomes(labels)) / 4).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(components.sum(), method['rps'], rtol=1e-6)
        method.update(records[name], **calibration, scaled_rps=float(scaled[name].mean()),
                      rps_components=dict(signed=float(components[0]), neutral=float(components[1])))
        methods[name] = method
    return methods, risks, errors, correct, scaled


def evaluate(data, resamples):
    labels, tuning_labels = data['y'], data['tuning_labels']
    groups, membership = np.unique(data['groups'], return_inverse=True)
    if (len(labels), len(groups), len(tuning_labels)) != (2122, 85, 325):
        raise ValueError('Expected the CH-SIMS v2 evaluation and tuning cohorts with held-out original videos.')
    if data['methods'].tolist() != [*CANDIDATES, *BASELINES]:
        raise ValueError('Expected all five declared methods in order.')
    methods, risks, errors, correct, scaled = loss_summaries(data)

    fixed = 'fixed_pooling'
    primary, secondary, sensitivity = {}, {}, {}
    for name in BASELINES:
        primary[name] = paired_reduction(risks[name] - risks[fixed], membership, resamples, 3)
        secondary[name] = {
            'accuracy': paired_reduction(correct[fixed] - correct[name], membership, resamples, 6),
            'mae': paired_reduction(errors[name] - errors[fixed], membership, resamples, 6),
        }
        sensitivity[name] = {
            'attention_raw': paired_reduction(risks[name] - risks['attention'], membership, resamples, 9),
            'attention_scaled': paired_reduction(scaled[name] - scaled['attention'], membership, resamples, 9),
            'fixed_pooling_scaled': paired_reduction(scaled[name] - scaled[fixed], membership, resamples, 9),
        }
    return dict(seeds=data['seeds'].tolist(), resamples=resamples, n_clips=len(labels), n_videos=len(groups),
                tuning_clips=len(tuning_labels), class_counts=np.bincount(labels, minlength=3).tolist(),
                methods=methods, primary_rps=primary, secondary=secondary, sensitivity_rps=sensitivity)


def evaluate_omg(data, resamples):
    labels, tuning_labels = data['y'], data['tuning_labels']
    groups, membership = np.unique(data['groups'], return_inverse=True)
    if (len(labels), len(groups), len(tuning_labels)) != (834, 65, 138):
        raise ValueError('Expected the original held-out OMG-Emotion videos and tuning clips.')
    names = data['methods'].tolist()
    if data['cohort'].item() != 'omg' or names != ['pointwise', *BASELINES]:
        raise ValueError('Expected the matched pointwise and three published OMG-Emotion forecasts.')
    methods, risks, errors, correct, scaled = loss_summaries(data)
    text_anchor = rps_summary(data['anchor'], labels)
    anchor_rps = ranked_probability_score(data['anchor'], labels).mean(axis=1)
    np.testing.assert_allclose(anchor_rps.mean(), text_anchor['rps'], atol=1e-7)
    by_gap = dict(zip(names, (ranked_probability_score(values, labels) for values in data['forecast'])))
    primary, secondary, sensitivity, anchor, gaps = {}, {}, {}, {}, {}
    candidate = 'pointwise'
    for name in BASELINES:
        primary[name] = paired_reduction(risks[name] - risks[candidate], membership, resamples, 3)
        secondary[name] = {
            'accuracy': paired_reduction(correct[candidate] - correct[name], membership, resamples, 6),
            'mae': paired_reduction(errors[name] - errors[candidate], membership, resamples, 6),
        }
        sensitivity[name] = paired_reduction(scaled[name] - scaled[candidate], membership, resamples, 3)
        anchor[name] = paired_reduction(anchor_rps - risks[name], membership, resamples, 3)
        difference = by_gap[name] - by_gap[candidate]
        gaps[name] = paired_reduction(difference[:, 1] - difference[:, 0], membership, resamples, 3)
    return dict(cohort='omg', seeds=data['seeds'].tolist(), resamples=resamples, n_clips=len(labels),
                n_videos=len(groups), tuning_clips=len(tuning_labels),
                class_counts=np.bincount(labels, minlength=3).tolist(), methods=methods,
                text_anchor=text_anchor, primary_rps=primary, secondary=secondary,
                sensitivity_rps=sensitivity, above_text_anchor=anchor, gap_interaction=gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', type=Path)
    source.add_argument('--forecasts', type=Path)
    parser.add_argument('--cohort', choices=BENCHMARK_CANDIDATES, default='sims_v2')
    parser.add_argument('--write-forecasts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose a positive number of bootstrap draws.')
    if args.write_forecasts and not args.runs:
        parser.error('Export the forecast packet from the run directories.')
    if args.runs:
        data = collect_forecasts(args.runs, SEEDS, args.cohort)
    else:
        with np.load(args.forecasts, allow_pickle=False) as forecast:
            data = dict(forecast)
    if data['seeds'].tolist() != list(SEEDS):
        raise ValueError('Expected all ten declared seeds in order.')
    report = evaluate_omg(data, args.resamples) if args.cohort == 'omg' else evaluate(data, args.resamples)
    if args.write_forecasts:
        temporary = args.write_forecasts.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(args.write_forecasts)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, method in report['methods'].items():
        print(f'{name:15s} RPS={method["rps"]:.6f}, scaled RPS={method["scaled_rps"]:.6f}, '
              f'MAE={method["mae"]:.6f}, accuracy={method["accuracy"]:.4f}')
    for name, effect in report['primary_rps'].items():
        print(name, effect['reduction'], effect['ci95_familywise'])


if __name__ == '__main__':
    main()
