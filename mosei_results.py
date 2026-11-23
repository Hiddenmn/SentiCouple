"""Summarize and time the matched text-and-vision benchmarks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import read_predictions, seed_video_interval
from benchmark import model_inputs
from benchmark_mosei import ROOT, SEEDS, SETTINGS, verify_development_split
from compare_benchmarks import load_matched, load_metadata, summarize, summarize_run
from coupling import video_resamples
from data import load_clips, observation_views
from networks import AudiovisualResidual, TextAnchor, intensity_anchor, sentiment_shift
from polarity_pool import coordinates, outcomes
from published_baselines import UPSTREAM, build_published
from published_inference import batch_layout, load_weights, measure
from published_seed_effects import BASELINES, run_paths, scaled_risk
from score_anchor_effects import SEEDS as TEN_SEEDS
from train import BatchSource, visual_component_options


def runs_for(reference_runs, matched_runs, seeds, models):
    paths = {'SentiCouple': [reference_runs / f'mosei_fresh_visual_seed{seed}' for seed in seeds]}
    for name in models:
        if name == 'lnln':
            paths[name] = [reference_runs / f'mosei_lnln_regression_seed{seed}' for seed in seeds]
        else:
            paths[name] = [matched_runs / f'{name.replace("-", "_")}_seed{seed}' for seed in seeds]
    return paths


def earlier_corpus_runs(reference_runs, matched_runs, seeds, models, cohort):
    original = run_paths(reference_runs, seeds, cohort)
    shared = 'fixed_pooling' if cohort == 'sims_v2' else 'pointwise'
    paths = {'SentiCouple': original[shared]}
    reference_methods = {model: name for name, model in BASELINES.items()}
    for model in models:
        if model in reference_methods:
            paths[model] = original[reference_methods[model]]
        else:
            prefix = f'{cohort}_{model.replace("-", "_")}'
            paths[model] = [matched_runs / f'{prefix}_seed{seed}' for seed in seeds]
    return paths


def mosi_runs(reference_runs, matched_runs, seeds, models):
    paths = {'SentiCouple': [reference_runs / f'mosi_matched_shared_seed{seed}' for seed in seeds]}
    for model in models:
        prefix = f'mosi_matched_{model.replace("-", "_")}'
        paths[model] = [matched_runs / f'{prefix}_seed{seed}' for seed in seeds]
    return paths


def paired_summary(gains, membership, resamples, family_size):
    result = {}
    for key, gain in gains.items():
        effect = dict(mean=float(gain.mean()), by_seed=gain.mean(1).tolist())
        if len(gain) > 1:
            tail = .025 / family_size
            effect['ci95_familywise'] = seed_video_interval(
                gain, membership, resamples, quantiles=(tail, 1 - tail))
        result[key] = effect
    return result


def summarize_benchmark(paths, seeds, resamples):
    reference = read_predictions(paths['SentiCouple'][0])
    label, intensity, groups = (reference[key] for key in ('y', 'sentiment', 'groups'))
    with np.load(paths['SentiCouple'][0] / 'tuning.npz', allow_pickle=False) as tuning:
        tuning_labels = tuning['y']
    membership = video_resamples(groups, resamples)[0]
    results, differences = {}, {}
    for name, directories in paths.items():
        packets = load_matched(directories, reference)
        metadata = load_metadata(directories)
        if [m['config']['seed'] for m in metadata] != list(seeds):
            raise ValueError(f'{name} forecasts are in the wrong seed order.')
        result, rps, mae = summarize_run(packets, metadata, 'intensity', label, intensity)
        key = 'intensity' if name == 'SentiCouple' else 'posterior'
        forecasts = np.stack([packet[key][1:3] for packet in packets])
        tune_forecasts = []
        for directory in directories:
            with np.load(directory / 'tuning.npz', allow_pickle=False) as tuning:
                np.testing.assert_array_equal(tuning['y'], tuning_labels, err_msg=directory.name)
                tune_forecasts.append(tuning[key][1:3])
        tune_forecasts = np.stack(tune_forecasts)
        scaled, calibration = scaled_risk(tune_forecasts, forecasts, tuning_labels, label)
        signed, neutral = (np.square(coordinates(forecasts) - outcomes(label)) / 4).mean((0, 1, 2))
        np.testing.assert_allclose(signed + neutral, result['rps'], atol=1e-7)
        result.update(**calibration, scaled_rps=float(scaled.mean()),
                      rps_components=dict(signed=float(signed), neutral=float(neutral)))
        results[name] = result
        if name == 'SentiCouple':
            shared_rps, shared_mae, shared_scaled = rps, mae, scaled
            independent, indep_rps, indep_mae = summarize(
                packets, 'independent_intensity', 'independent_intensity_score', label, intensity)
            supervision = dict(independent_metrics=independent, gain=paired_summary(
                dict(rps=indep_rps - rps, mae=indep_mae - mae), membership, resamples, 1))
        else:
            for meta in metadata:
                if meta['config']['model'] != name or meta['upstream_revision'] != UPSTREAM[name]:
                    raise ValueError(f'{name} did not use the pinned reference implementation.')
                history = meta['tune_rps_by_epoch']
                if (meta['selection']['epoch'] != int(np.argmin(history)) + 1
                        or meta['selection']['rps'] != min(history)):
                    raise ValueError(f'{name} did not select the best tuning RPS checkpoint.')
            np.testing.assert_allclose(rps.mean(axis=-1), [m['heldout_rps'] for m in metadata], atol=1e-7)
            gains = dict(rps=rps - shared_rps, mae=mae - shared_mae, scaled_rps=scaled - shared_scaled)
            differences[name] = paired_summary(gains, membership, resamples, len(paths) - 1)
        print(f'{name:12s} RPS {result["rps"]:.6f}  MAE {result["mae"]:.6f}  '
              f'macro-F1 {result["macro_f1"]:.4f}', flush=True)
    return dict(clips=len(label), videos=int(len(np.unique(groups))), tuning_clips=len(tuning_labels),
                seeds=list(seeds), resamples=resamples,
                methods=results, reference_minus_shared=differences, shared_versus_independent=supervision)


def predictor_for(run, name, source, upstream):
    meta, = load_metadata([run])
    config = meta['config']
    audio_dim = source.audio.shape[-1]
    visual_only = audio_dim == 0
    if name == 'SentiCouple':
        bins, width = (config[key] for key in ('bins', 'width'))
        anchor = load_weights(TextAnchor(source.text.shape[-1], bins, width, visual_only=visual_only),
                              run / 'anchor.pt', source.device)
        fusion = AudiovisualResidual(audio_dim, source.vision.shape[-1], width,
                                     attention_bins=bins if config['backbone'] != 'temporal' else None,
                                     text_query=config['backbone'] == 'attention',
                                     visual_encoding=config.get('visual_encoding', 'normalized'),
                                     **visual_component_options('intensity'))
        fusion = load_weights(fusion, run / 'intensity.pt', source.device)
        gain = meta['selection']['intensity'].get('score_gain', 1.)

        def predict(inputs):
            observed, visible = inputs
            context, prior = anchor(observed[0], *visible)
            fused = fusion(context, prior, *observed[1:], *visible[1:])
            shift = sentiment_shift(fused, prior)
            score = (intensity_anchor(prior, source.class_centers) + gain * shift).tanh()
            return fused.softmax(-1), score * source.intensity_scale

        parameters = sum(p.numel() for model in (anchor, fusion) for p in model.parameters() if p.requires_grad)
        return predict, parameters, ('intensity', 'intensity_score')

    dimensions = (source.text.shape[-1], max(1, audio_dim), source.vision.shape[-1])
    lengths = (source.text.shape[1], source.audio.shape[1], source.vision.shape[1])
    head = config['head']
    model = build_published(config['model'], upstream, config['corpus'], dimensions, lengths, head, visual_only)
    model = load_weights(model, run / 'model.pt', source.device)
    if head == 'regression':
        center, scale, lower, log_gap, slope = meta['selection']['calibration']
        cuts = torch.tensor([lower, lower + np.exp(log_gap)], dtype=torch.float32, device=source.device)

        def predict(inputs):
            output, _ = model(*inputs)
            score = output[:, 0]
            cumulative = (cuts - slope * ((score[:, None] - center) / scale)).sigmoid()
            return torch.stack((cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0],
                                1 - cumulative[:, 1]), dim=1), score
    else:
        def predict(inputs):
            output, _ = model(*inputs)
            return output[:, :3].softmax(-1), output[:, 3].tanh() * source.intensity_scale

    return predict, meta['trainable_parameters'], ('posterior', 'score')


@torch.inference_mode()
def profile_all(paths, upstream, data, eval_groups, *, cohort='mosei', group_reference=None, visual_only=True):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    split_name = 'new-videos' if cohort == 'sims_v2' else 'official'
    neutral_margin = .10 if cohort == 'omg' else 0.
    clips, split = load_clips(data, 24, split_name, 1729, eval_groups=eval_groups,
                             group_reference=group_reference, neutral_margin=neutral_margin)
    if cohort == 'mosei':
        verify_development_split(paths['SentiCouple'][0], clips, split, 97,
                                 inputs='visual' if visual_only else 'audiovisual')
    views = observation_views(clips, [.35, .65], 1777)
    if visual_only:
        views[:, 0] = False
    source = BatchSource(clips, views, split['fit'], device, visual_only=visual_only, stream_text=True)
    held = split['eval']
    predictors, archived, timings = {}, {}, {}
    for name, (run, *_) in paths.items():
        packet = read_predictions(run)
        np.testing.assert_array_equal(packet['ids'], clips.ids[held])
        np.testing.assert_array_equal(packet['masks'], views[:, :, held])
        predict, parameters, keys = predictor_for(run, name, source, upstream.get(name))
        batch = load_metadata([run])[0]['config']['batch']
        for condition in (1, 2):
            for start in range(0, len(held), batch):
                rows = slice(start, start + batch)
                indices = torch.as_tensor(held[rows], device=device)
                conditions = torch.full_like(indices, condition)
                actual = predict(model_inputs(source, conditions, indices))
                for values, key in zip(actual, keys, strict=True):
                    np.testing.assert_allclose(values.cpu().numpy(), packet[key][condition, rows],
                                               atol=1e-4, rtol=1e-4, err_msg=name)
        predictors[name] = predict
        archived[name] = [packet[key][1:3] for key in keys]
        timings[name] = dict(trainable_parameters=parameters, batches={}, max_batch_difference={})
        print(f'{name}: replayed the held-out scores at the original batch size {batch}', flush=True)
    return profile_cached(source, held, predictors, archived, timings)


def profile_cached(source, held, predictors, archived, timings):
    device = source.device
    for size, layout in batch_layout(len(held)):
        maximum = {name: [0., 0.] for name in predictors}
        batches = []
        for rows, conditions in layout:
            indices = torch.as_tensor(held[rows], device=device)
            batch = model_inputs(source, torch.as_tensor(conditions, device=device), indices)
            for name, predict in predictors.items():
                estimates = [value.cpu().numpy() for value in predict(batch)]
                for index, (actual, expected) in enumerate(zip(estimates, archived[name], strict=True)):
                    if not np.isfinite(actual).all():
                        raise ValueError(f'{name} produced nonfinite outputs at batch size {size}.')
                    difference = np.max(np.abs(actual - expected[conditions - 1, rows]))
                    maximum[name][index] = max(maximum[name][index], float(difference))
            batches.append(batch)
        for name, (probabilities, scores) in maximum.items():
            timings[name]['max_batch_difference'][size] = dict(probability=probabilities, intensity=scores)
        for order, names in (('forward', list(predictors)), ('reverse', list(reversed(predictors)))):
            for name in names:
                timings[name]['batches'].setdefault(size, {})[order] = measure(predictors[name], batches)
    return dict(device=torch.cuda.get_device_name(device), seed=97, methods=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('mosei', 'sims_v2', 'omg', 'mosi'), default='mosei')
    parser.add_argument('--models', choices=('lnln', *SETTINGS), nargs='+', default=['lnln', *SETTINGS])
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--reference-runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--matched-runs', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--group-reference', type=Path)
    parser.add_argument('--eval-groups', type=Path)
    parser.add_argument('--profile-upstream', type=Path, nargs='+',
                        help='Profile seed 97 on CUDA with the pinned checkouts in model order.')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    is_mosei = options.cohort == 'mosei'
    default_seeds = SEEDS if options.cohort in ('mosei', 'mosi') else TEN_SEEDS
    seeds = options.seeds or default_seeds
    if options.resamples < 1 or len(set(seeds)) != len(seeds) or len(set(options.models)) != len(options.models):
        parser.error('Use positive resampling counts and distinct models and seeds.')
    if not set(seeds).issubset(default_seeds):
        parser.error(f'Seeds for {options.cohort} must come from {default_seeds}.')
    if is_mosei:
        matched = options.matched_runs or ROOT / 'runs/mosei_matched'
        paths = runs_for(options.reference_runs, matched, seeds, options.models)
    else:
        matched = options.matched_runs or options.reference_runs
        if options.cohort == 'mosi':
            paths = mosi_runs(options.reference_runs, matched, seeds, options.models)
        else:
            paths = earlier_corpus_runs(options.reference_runs, matched, seeds, options.models, options.cohort)
    result = summarize_benchmark(paths, seeds, options.resamples)
    if options.profile_upstream:
        if seeds[0] != 97 or len(options.profile_upstream) != len(options.models):
            parser.error('Profile seed 97 first and supply one upstream checkout for each reference model.')
        feature_files = {'mosei': 'mosei_aligned_50.pkl', 'sims_v2': 'ch_sims_v2_unaligned.pkl',
                         'omg': 'omg_unaligned.pkl', 'mosi': 'mosi_unaligned_50.pkl'}
        data = options.data or ROOT / 'data' / feature_files[options.cohort]
        eval_groups = options.eval_groups or (ROOT / 'splits/mosei_eval_videos.txt' if is_mosei else None)
        group_reference = options.group_reference
        if options.cohort == 'sims_v2' and group_reference is None:
            group_reference = ROOT / 'data/ch_sims_unaligned_39.pkl'
        result['timing'] = profile_all(paths, dict(zip(options.models, options.profile_upstream, strict=True)),
                                       data, eval_groups, cohort=options.cohort, group_reference=group_reference)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(options.output)


if __name__ == '__main__':
    main()
