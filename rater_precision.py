"""Compare precision-weighted score supervision with matched score heads."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from cremad_effects import POLARITIES, balanced_reduction
from data import load_clips, observation_views, shuffle_within_score_ranks
from networks import TextAnchor
from ordinal_controls import paired_reduction
from prepare_ravdess import POLARITY as RAVDESS_POLARITY
from published_inference import load_weights
from score_anchor_effects import SEEDS
from train import BatchSource, anchor_outputs, calibrated_head_forecasts, fit_fusion
from view_stability import RUNS


COHORTS = ('cremad', 'ravdess')
TREATMENTS = ('precision', 'permuted_precision')
REFERENCES = {'shared': 'intensity', 'independent': 'independent_intensity', 'anchor': 'anchor'}
TESTS = ('shared', 'permuted_precision')


def ravdess_ratings(path):
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    ratings = {}
    try:
        rows = workbook['Speech_Intensity'].iter_rows(min_row=3, values_only=True)
        header = next(rows)
        name = header.index('Filename')
        ratings_columns = [header.index(f'R{i}') for i in range(1, 11)]
        for row in rows:
            if not isinstance(row[name], str):
                continue
            clip = Path(row[name]).stem
            modality, channel, emotion, *_ = map(int, clip.split('-'))
            if (modality, channel) != (1, 1) or emotion not in RAVDESS_POLARITY:
                continue
            raw = np.asarray([row[column] for column in ratings_columns], dtype=np.float64)
            if clip in ratings or not np.all((1 <= raw) & (raw <= 5)):
                raise ValueError(f'Unexpected rating for RAVDESS clip {clip}.')
            ratings[clip] = raw * RAVDESS_POLARITY[emotion] / 5.
    finally:
        workbook.close()
    return ratings


def measurement_variance(clips, votes):
    by_clip = [votes[str(clip).split('$_$', 1)[1]] for clip in clips.ids]
    counts = np.asarray([len(values) for values in by_clip])
    if np.any(counts < 2):
        raise ValueError('Each clip needs at least two distinct ratings.')
    np.testing.assert_allclose(clips.sentiment, [np.mean(values) for values in by_clip], rtol=3e-7, atol=5e-8)
    return np.asarray([np.var(values, ddof=1) for values in by_clip]) / counts, counts


def precision_weights(variance, labels, fit):
    weights = np.ones(len(variance), dtype=np.float32)
    for polarity in range(3):
        cell = fit[labels[fit] == polarity]
        if not len(cell):
            raise ValueError('All polarities must occur in the fitting set.')
        floor = variance[cell].mean()
        if floor > 0:
            relative = floor / (floor + variance[cell])
            weights[cell] = relative / relative.mean()
    return weights


def fit_weighted(args, source, clips, split, saved_anchor, banks, weights):
    model, selection = fit_fusion(args, 'intensity', source, split, saved_anchor, clips.label, banks,
                                  score_weights=torch.as_tensor(weights, device=source.device))
    posterior, score, gain = calibrated_head_forecasts(model, source, clips, split, saved_anchor, args.batch)
    return posterior, score, dict(selection, score_gain=gain)


def refit_fold(paths, seeds, votes, device):
    metadata = load_metadata(paths)
    reference = metadata[0]['config']
    for seed, run in zip(seeds, metadata, strict=True):
        config = run['config']
        if config['seed'] != seed or any(config[key] != value for key, value in reference.items()
                                        if key not in ('seed', 'output')):
            raise ValueError('A checkpoint uses different training settings.')
    if reference['inputs'] != 'visual' or reference['score_anchor'] != 'class-mean':
        raise ValueError('Use the original visual model with class-mean anchor intensity.')
    clips, split = load_clips(reference['data'], reference['bins'], reference['split'], reference['split_seed'],
                              reference['fold'], neutral_margin=reference['neutral_margin'])
    fit, tune, held = (split[key] for key in ('fit', 'tune', 'eval'))
    views = observation_views(clips, [.35, .65], reference['mask_seed'])
    views[:, 0] = False
    source = BatchSource(clips, views, fit, device, visual_only=True, stream_text=reference['stream_text'])
    banks = np.full((len(views), len(clips.ids), 1), -1, dtype=np.int32)
    variance, counts = measurement_variance(clips, votes)
    weights = precision_weights(variance, clips.label, fit)
    original = read_predictions(paths[0])
    for key, expected in (('ids', clips.ids[held]), ('groups', clips.groups[held]), ('y', clips.label[held]),
                          ('sentiment', clips.sentiment[held]), ('masks', views[:, :, held])):
        np.testing.assert_array_equal(original[key], expected)
    packets = load_matched(paths, original)
    results = {key: original[key] for key in ('ids', 'groups', 'y', 'sentiment')}
    for name, arm in REFERENCES.items():
        for kind in ('', '_score'):
            results[name + kind] = np.stack([packet[arm + kind][1:3] for packet in packets])
    statistics = {}
    for label, name in enumerate(POLARITIES):
        cell = fit[clips.label[fit] == label]
        statistics[name] = dict(n=len(cell), votes_min=int(counts[cell].min()), votes_max=int(counts[cell].max()),
                                mean_variance=float(variance[cell].mean()),
                                weight_quantiles=np.quantile(weights[cell], [0, .1, .5, .9, 1]).tolist(),
                                mean_weight=float(weights[cell].mean()),
                                effective_fraction=float(weights[cell].sum() ** 2 / np.square(weights[cell]).sum()
                                                         / len(cell)))
    results['measurement'] = np.asarray(json.dumps(statistics))

    selections = []
    forecasts = {name: [] for name in TREATMENTS}
    for seed, run, packet in zip(seeds, paths, packets, strict=True):
        args = argparse.Namespace(**dict(reference, seed=seed))
        anchor = TextAnchor(source.text.shape[-1], args.bins, args.width, visual_only=True)
        saved_anchor = anchor_outputs(load_weights(anchor, run / 'anchor.pt', device), source, args.batch)
        np.testing.assert_allclose(saved_anchor[1][:, held].softmax(-1).cpu().numpy(), packet['anchor'],
                                   atol=3e-6, rtol=2e-5)
        with np.load(run / 'tuning.npz', allow_pickle=False) as archived:
            np.testing.assert_array_equal(archived['y'], clips.label[tune])
            np.testing.assert_allclose(saved_anchor[1][:, tune].softmax(-1).cpu().numpy(), archived['anchor'],
                                       atol=3e-6, rtol=2e-5)
        placebo = shuffle_within_score_ranks(weights, clips.label, clips.sentiment, fit,
                                             np.random.default_rng(seed + 739))
        selection = {}
        for name, assignment in zip(TREATMENTS, (weights, placebo), strict=True):
            probs, score, selection[name] = fit_weighted(args, source, clips, split, saved_anchor, banks, assignment)
            forecasts[name].append((probs, score))
        selections.append(selection)
        print(f'{run.name}: completed two shared-head fits', flush=True)
    for name, pairs in forecasts.items():
        probabilities, scores = zip(*pairs, strict=True)
        results[name], results[f'{name}_score'] = np.stack(probabilities), np.stack(scores)
    results['selections'] = np.asarray(json.dumps(selections))
    return results


def collect(runs, cohort, seeds, votes, device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evidence = dict(cohort=np.asarray(cohort), seeds=np.asarray(seeds))
    for fold, prefix in enumerate(RUNS[cohort]):
        values = refit_fold([runs / f'{prefix}{seed}' for seed in seeds], seeds, votes, device)
        evidence.update({f'fold{fold}_{name}': value for name, value in values.items()})
    return evidence


def summarize(evidence, resamples):
    cohort, seeds = evidence['cohort'].item(), evidence['seeds']
    prefixes = [f'fold{i}_' for i in range(len(RUNS[cohort]))]
    y, group, sentiment = (np.concatenate([evidence[prefix + key] for prefix in prefixes])
                           for key in ('y', 'groups', 'sentiment'))
    n_groups = sum(len(np.unique(evidence[prefix + 'groups'])) for prefix in prefixes)
    if n_groups != len(np.unique(group)):
        raise ValueError('Each held-out actor should occur in only one fold.')
    _, membership = np.unique(group, return_inverse=True)
    methods, rps, mae = {}, {}, {}
    for name in (*TREATMENTS, *REFERENCES):
        forecast = np.concatenate([evidence[x + name] for x in prefixes], axis=2).astype(np.float64)
        score = np.concatenate([evidence[x + name + '_score'] for x in prefixes], axis=2)
        rps[name] = ranked_probability_score(forecast, y).mean(axis=1)
        mae[name] = np.abs(score - sentiment).mean(axis=1)
        class_rps = [float(rps[name][:, y == level].mean()) for level in range(3)]
        methods[name] = dict(rps=float(rps[name].mean()), balanced_rps=float(np.mean(class_rps)),
                             rps_by_class=class_rps, mae=float(mae[name].mean()),
                             accuracy=float((forecast.argmax(-1) == y).mean()))
    rng = np.random.default_rng(29)
    seed_draws = rng.multinomial(len(seeds), np.repeat(1 / len(seeds), len(seeds)), resamples)
    actor_draws = rng.multinomial(n_groups, np.repeat(1 / n_groups, n_groups), resamples)
    contrasts = {name: balanced_reduction(rps[name] - rps['precision'], y, membership,
                                           seed_draws, actor_draws, 4) for name in TESTS}
    mae_contrasts = {name: paired_reduction(mae[name] - mae['precision'], membership, resamples, 4) for name in TESTS}
    return dict(cohort=cohort, seeds=seeds.tolist(), resamples=resamples, n_clips=len(y), n_actors=n_groups,
                methods=methods, balanced_rps_reductions=contrasts, calibrated_mae_reductions=mae_contrasts,
                measurements=[json.loads(evidence[x + 'measurement'].item()) for x in prefixes],
                selections=[json.loads(evidence[x + 'selections'].item()) for x in prefixes])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--ratings', type=Path)
    parser.add_argument('--cohort', choices=COHORTS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--first-seed', action='store_true', help='Refit only the first seed for a smoke test.')
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not (args.cohort and args.ratings)) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples. With --runs, specify the cohort and its published ratings.')
    if args.runs:
        if args.cohort == 'cremad':
            from prepare_cremad import audience_ratings

            votes = audience_ratings(args.ratings)
        else:
            votes = ravdess_ratings(args.ratings)
        seeds = SEEDS[:1] if args.first_seed else SEEDS
        evidence = collect(args.runs, args.cohort, seeds, votes, torch.device(args.device))
    else:
        with np.load(args.evidence, allow_pickle=False) as archive:
            evidence = dict(archive)
    report = summarize(evidence, args.resamples)
    if args.write_evidence:
        np.savez_compressed(args.write_evidence, **evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for ref, value in report['balanced_rps_reductions'].items():
        low, high = value['ci95_familywise']
        print(f'{ref}: {value["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]')


if __name__ == '__main__':
    main()
