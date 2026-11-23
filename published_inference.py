"""Time matched CH-SIMS v2 and OMG-Emotion predictors on cached input features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import read_predictions
from benchmark import model_inputs
from compare_benchmarks import load_metadata
from data import load_clips, observation_views
from networks import AudiovisualResidual, TextAnchor, intensity_anchor, sentiment_shift
from published_baselines import build_published
from published_seed_effects import BASELINES, BENCHMARK_CANDIDATES, CANDIDATES, run_paths, verify_training
from train import BatchSource, visual_component_options


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.to(device).eval()


def predictor(name, run, metadata, upstream, source, arm='intensity'):
    dimensions = (source.text.shape[-1], max(1, source.audio.shape[-1]), source.vision.shape[-1])
    lengths = (source.text.shape[1], source.audio.shape[1], source.vision.shape[1])
    scale = source.intensity_scale
    visual_only = source.audio.shape[-1] == 0
    config = metadata['config']
    if name in BASELINES:
        model = build_published(BASELINES[name], upstream, config['corpus'], dimensions, lengths,
                                'joint', visual_only)
        model = load_weights(model, run / 'model.pt', source.device)

        def predict(inputs):
            output, _ = model(*inputs)
            return output[:, :3].softmax(-1), output[:, 3].tanh() * scale

        return predict

    bins, width = config['bins'], config['width']
    anchor = load_weights(TextAnchor(dimensions[0], bins, width, visual_only=visual_only),
                          run / 'anchor.pt', source.device)
    attention_bins = bins if config['backbone'] == 'stream-query' else None
    residual = AudiovisualResidual(source.audio.shape[-1], dimensions[-1], width,
                                    attention_bins=attention_bins, text_query=False,
                                    visual_encoding=config['visual_encoding'], **visual_component_options(arm))
    residual = load_weights(residual, run / f'{arm}.pt', source.device)
    gain = metadata['selection'][arm]['score_gain']

    def predict(inputs):
        observed, visible = inputs
        context, logits = anchor(observed[0], *visible)
        fused = residual(context, logits, *observed[1:], *visible[1:])
        shift = sentiment_shift(fused, logits)
        score = (intensity_anchor(logits, source.class_centers) + gain * shift).tanh() * scale
        return fused.softmax(-1), score

    return predict


def timed_cuda_forward(predict, batch):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(4):
        predict(batch)
    stop.record()
    return start, stop


def timing_summary(times):
    return dict(batches=len(times), median_ms=float(np.median(times)), iqr_ms=np.quantile(times, [.25, .75]).tolist())


def measure(predict, batches):
    # Warm both the model and CUDA before recording the timed forwards.
    for pass_index in range(32):
        predict(batches[pass_index % len(batches)])
    torch.cuda.synchronize()
    spans = [timed_cuda_forward(predict, batch) for batch in batches]
    torch.cuda.synchronize()
    times = np.array([start.elapsed_time(stop) / 4 for start, stop in spans])
    return timing_summary(times)


def measure_interleaved(predictors, batches, repeat):
    names = list(predictors)
    spans = {name: [] for name in names}
    for index in range(32):
        batch = batches[index % len(batches)]
        for predict in predictors.values():
            predict(batch)
    torch.cuda.synchronize()
    for index, batch in enumerate(batches):
        order = names if (index + repeat) % 2 == 0 else reversed(names)
        for name in order:
            spans[name].append(timed_cuda_forward(predictors[name], batch))
    torch.cuda.synchronize()
    times = {name: np.array([start.elapsed_time(stop) / 4 for start, stop in records])
             for name, records in spans.items()}
    differences = times['fixed_pooling'] - times['visual_projection']
    return dict(methods={name: timing_summary(values) for name, values in times.items()},
                paired_ms=timing_summary(differences),
                projection_faster_batches=int(np.count_nonzero(differences > 0)))


def observed_features(args, device):
    if args.cohort == 'omg':
        clips, split = load_clips(args.data, 24, 'official', 1729, neutral_margin=.10)
    else:
        clips, split = load_clips(args.data, 24, 'new-videos', 1729, group_reference=args.group_reference)
    views = observation_views(clips, [.35, .65], 1777)
    views[:, 0] = False
    source = BatchSource(clips, views, split['fit'], device, visual_only=True, stream_text=True)
    return clips, split['eval'], views, source


def batch_layout(n):
    sizes = ((1, 64), (32, 16))
    if n < 32:
        raise ValueError('A timing cohort needs at least 32 distinct clips.')
    ordered = np.resize(np.random.default_rng(2317).permutation(n), sum(size * count for size, count in sizes))
    begin = 0
    for size, count in sizes:
        end = begin + size * count
        positions = ordered[begin:end].reshape(count, size)
        conditions = 1 + np.arange(begin, end).reshape(count, size) % 2
        yield size, zip(positions, conditions)
        begin = end


@torch.inference_mode()
def profile(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    if args.method == 'visual_projection':
        run, arm, reference = (args.runs / 'sims_v2_visual_components_seed97',
                               'intensity_projection', 'fixed_pooling')
    else:
        run, arm, reference = run_paths(args.runs, (97,), args.cohort)[args.method][0], 'intensity', args.method
    metadata, = load_metadata([run])
    verify_training(reference, [metadata], (97,), args.cohort)
    clips, test, views, source = observed_features(args, device)
    archived = read_predictions(run)
    np.testing.assert_array_equal(archived['ids'], clips.ids[test])
    np.testing.assert_array_equal(archived['masks'], views[:, :, test])
    predict = predictor(args.method, run, metadata, args.upstream, source, arm)
    probability_key, score_key = ('posterior', 'score') if args.method in BASELINES else (arm, f'{arm}_score')
    summaries = {}
    for size, layout in batch_layout(len(test)):
        batches = []
        probability_differences, score_differences = [], []
        for rows, gaps in layout:
            indices = torch.as_tensor(test[rows], device=device)
            gap = torch.as_tensor(gaps, device=device)
            batch = model_inputs(source, gap, indices)
            forecast, score = predict(batch)
            probability_differences.append(np.abs(archived[probability_key][gaps, rows] - forecast.cpu().numpy()).max())
            score_differences.append(np.abs(archived[score_key][gaps, rows] - score.cpu().numpy()).max())
            batches.append(batch)
        probability_error = float(np.max(probability_differences))
        score_error = float(np.max(score_differences))
        # Batch shape can change the float32 kernels used by a public backbone.
        if not (probability_error <= 1e-4 and score_error <= 1e-4):
            raise ValueError(f'Batch {size} forecast replay differs by {probability_error:g} in probability '
                             f'and {score_error:g} in intensity.')
        summaries[str(size)] = dict(**measure(predict, batches),
                                    max_probability_difference=probability_error, max_intensity_difference=score_error)
    return dict(method=args.method, cohort=args.cohort, seed=97, device=torch.cuda.get_device_name(device),
                torch=str(torch.__version__), times=summaries)


@torch.inference_mode()
def profile_visual_pair(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    clips, test, views, source = observed_features(args, device)
    reference = run_paths(args.runs, (97,))['fixed_pooling'][0]
    methods = {'fixed_pooling': (reference, 'intensity'),
               'visual_projection': (args.runs / 'sims_v2_visual_components_seed97', 'intensity_projection')}
    predictors, forecasts = {}, {}
    for name, (run, arm) in methods.items():
        metadata, = load_metadata([run])
        verify_training('fixed_pooling', [metadata], (97,))
        packet = read_predictions(run)
        np.testing.assert_array_equal(packet['ids'], clips.ids[test])
        np.testing.assert_array_equal(packet['masks'], views[:, :, test])
        predictors[name] = predictor(name, run, metadata, None, source, arm)
        forecasts[name] = packet[arm], packet[f'{arm}_score']
    results = {}
    for size, layout in batch_layout(len(test)):
        batches = []
        replay_errors = {name: np.zeros(2) for name in methods}
        for rows, gaps in layout:
            indices = torch.as_tensor(test[rows], device=device)
            batch = model_inputs(source, torch.as_tensor(gaps, device=device), indices)
            batches.append(batch)
            for name in methods:
                predicted = predictors[name](batch)
                expected = [array[gaps, rows] for array in forecasts[name]]
                error = [float(np.max(np.abs(p.cpu().numpy() - e))) for p, e in zip(predicted, expected)]
                replay_errors[name] = np.maximum(replay_errors[name], error)
        for error in replay_errors.values():
            if np.max(error) > 1e-4:
                raise ValueError(f'Forecast replay for batch size {size} differed by {np.max(error):g}.')
        results[str(size)] = dict(repeats=[measure_interleaved(predictors, batches, repeat) for repeat in range(3)],
                                  replay_error={name: error.tolist() for name, error in replay_errors.items()})
    return dict(method=args.method, seed=97, device=torch.cuda.get_device_name(device),
                torch=str(torch.__version__), times=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    visual_controls = ('visual_projection', 'visual_pooling_pair')
    parser.add_argument('--method', required=True,
                        choices=(*CANDIDATES, 'pointwise', *BASELINES, *visual_controls))
    parser.add_argument('--cohort', choices=BENCHMARK_CANDIDATES, default='sims_v2')
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--group-reference', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    methods = (*BENCHMARK_CANDIDATES[args.cohort], *BASELINES)
    if args.cohort == 'sims_v2':
        methods += visual_controls
        if args.group_reference is None:
            parser.error('Supply the original CH-SIMS packet as --group-reference for CH-SIMS v2.')
    if args.method not in methods:
        parser.error(f'{args.method} is not a profiled method for {args.cohort}.')
    if args.method in BASELINES and not args.upstream:
        parser.error('Supply the pinned checkout for the published core.')
    result = profile_visual_pair(args) if args.method == 'visual_pooling_pair' else profile(args)
    report = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
