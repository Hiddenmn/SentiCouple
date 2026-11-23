"""Compare class weighting defined by a frozen text forecast with pooled weighting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ablate import ranked_probability_score, read_predictions
from aggregate import f1_by_run
from compare_benchmarks import load_matched, load_metadata
from data import load_clips, observation_views
from networks import TextAnchor
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_inference import load_weights
from published_seed_effects import BASELINES, BENCHMARK_CANDIDATES, run_paths, verify_training
from score_anchor_effects import SEEDS
from train import BatchSource, anchor_outputs, calibrated_head_forecasts, fit_fusion


RUN_NAME = {'sims_v2': 'fixed_pooling', 'omg': 'pointwise'}
ARMS = {
    'text_matched': 'intensity',
    'text_independent': 'independent_intensity',
    'text_weight_only': 'intensity',
    'text_offset_only': 'intensity',
    'global_matched': 'intensity',
    'inverse_frequency': 'intensity',
}
REFERENCES = {'anchor': 'anchor', 'shared': 'intensity', 'independent': 'independent_intensity'}
PRIMARY = ('shared', 'global_matched', 'text_independent')
COMPONENTS = ('text_weight_only', 'text_offset_only', 'inverse_frequency')


def normalize_weights(probabilities):
    inverse = 2. / (probabilities + 1. / probabilities.shape[-1])
    return inverse / (inverse * probabilities).sum(-1, keepdim=True)


def objective(base_weights, labels, fit, prior_shape, *, weight=True, offset=True):
    weights = base_weights.expand(prior_shape)
    observed = weights.gather(-1, labels[None, :, None].expand(prior_shape[0], -1, -1)).squeeze(-1)
    observed = observed / observed[:, fit].mean()
    correction = base_weights.log().expand(prior_shape) if offset else None

    def classify(logits, targets, views, indices):
        if correction is not None:
            logits = logits + correction[views, indices]
        values = F.cross_entropy(logits, targets, reduction='none')
        if weight:
            values = values * observed[views, indices]
        return values.mean()

    fitting = observed[:, fit]
    return classify, dict(minimum=float(fitting.min()), maximum=float(fitting.max()),
                          sd=float(fitting.std()), mean=float(fitting.mean()))


def study_objectives(priors, label, fit):
    probabilities = priors.softmax(-1)
    frequency = torch.bincount(label[fit], minlength=3).to(priors.dtype) / len(fit)
    if not torch.all(frequency > 0):
        raise ValueError('Both weighting rules require all three classes in the fitting cohort.')
    informative = normalize_weights(probabilities)
    global_weights = normalize_weights(frequency)
    contextual_matched = objective(informative, label, fit, priors.shape)
    return {
        'text_matched': contextual_matched,
        'text_independent': contextual_matched,
        'text_weight_only': objective(informative, label, fit, priors.shape, offset=False),
        'text_offset_only': objective(informative, label, fit, priors.shape, weight=False),
        'global_matched': objective(global_weights, label, fit, priors.shape),
        'inverse_frequency': objective(frequency.reciprocal(), label, fit, priors.shape, offset=False),
    }


def load_references(runs, cohort, seeds, compact_runs):
    with np.load(runs / f'{cohort}_published_predictions.npz', allow_pickle=False) as archive:
        baseline = dict(archive)
    baseline_rows = [baseline['seeds'].tolist().index(seed) for seed in seeds]
    matched = load_matched(compact_runs, baseline)
    reference = {key: baseline[key] for key in ('ids', 'groups', 'y', 'sentiment', 'masks')}
    for name, arm in REFERENCES.items():
        for suffix in ('', '_score'):
            reference[name + suffix] = np.stack([packet[arm + suffix][1:3] for packet in matched])
    if RUN_NAME[cohort] not in BENCHMARK_CANDIDATES[cohort]:
        raise ValueError('The matched compact model is absent.')
    indices = {name: baseline['methods'].tolist().index(name) for name in (*BASELINES, RUN_NAME[cohort])}
    np.testing.assert_array_equal(reference['shared'], baseline['forecast'][indices[RUN_NAME[cohort]], baseline_rows])
    for name in BASELINES:
        reference[name] = baseline['forecast'][indices[name], baseline_rows]
        reference[f'{name}_score'] = baseline['score'][indices[name], baseline_rows]
    return reference


def source_for(config, device):
    clips, split = load_clips(config['data'], config['bins'], config['split'], config['split_seed'], config['fold'],
                              group_reference=config['group_reference'], exclude_eval=config['exclude_eval'],
                              eval_groups=config['eval_groups'], neutral_margin=config['neutral_margin'])
    views = observation_views(clips, [.35, .65], config['mask_seed'])
    views[:, 0] = False
    source = BatchSource(clips, views, split['fit'], device, visual_only=True, stream_text=config['stream_text'])
    return clips, split, views, source


def anchor_for(run, source, split, config, reference, row):
    anchor = TextAnchor(source.text.shape[-1], config['bins'], config['width'], visual_only=True)
    anchor = load_weights(anchor, run / 'anchor.pt', source.device)
    saved = anchor_outputs(anchor, source, config['batch'])
    with np.load(run / 'tuning.npz', allow_pickle=False) as packet:
        tuning_label = source.label[split['tune']].cpu().numpy()
        np.testing.assert_array_equal(tuning_label, packet['y'])
        archived = packet['anchor']
        actual = saved[1][:len(archived), split['tune']].softmax(-1).cpu().numpy()
        np.testing.assert_allclose(actual, archived, atol=3e-6, rtol=2e-5)
    posterior = saved[1][1:3, split['eval']].softmax(-1).cpu().numpy()
    np.testing.assert_allclose(posterior, reference['anchor'][row], atol=3e-6, rtol=2e-5)
    return saved


def collect(runs, cohort, seeds, device, verify_original=False):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    name = RUN_NAME[cohort]
    compact_runs = run_paths(runs, seeds, cohort)[name]
    metadata = load_metadata(compact_runs)
    verify_training(name, metadata, seeds, cohort)
    reference = load_references(runs, cohort, seeds, compact_runs)
    settings = metadata[0]['config']
    clips, split, views, source = source_for(settings, device)
    held = split['eval']
    for key, values in (('ids', clips.ids[held]), ('groups', clips.groups[held]), ('y', clips.label[held]),
                        ('sentiment', clips.sentiment[held]), ('masks', views[:, :, held])):
        np.testing.assert_array_equal(values, reference[key], err_msg=key)
    banks = np.full((len(views), len(clips.ids), 1), -1, dtype=np.int32)
    forecasts = {name: [] for name in ARMS}
    selections = []
    for row, (seed, run, record) in enumerate(zip(seeds, compact_runs, metadata, strict=True)):
        config = record['config']
        if any(config[key] != value for key, value in settings.items() if key not in ('seed', 'output')):
            raise ValueError(f'Training settings differ for {run.name}.')
        if config['score_anchor'] != 'class-mean' or not config['calibrate_score_gain']:
            raise ValueError('Use the original calibrated visual head.')
        args = argparse.Namespace(**config)
        saved = anchor_for(run, source, split, config, reference, row)
        if verify_original:
            model, selected = fit_fusion(args, 'intensity', source, split, saved, clips.label, banks)
            posterior, score, _ = calibrated_head_forecasts(model, source, clips, split, saved, args.batch)
            np.testing.assert_allclose(posterior, reference['shared'][row], atol=4e-6, rtol=2e-5)
            np.testing.assert_allclose(score, reference['shared_score'][row], atol=4e-6, rtol=2e-5)
            if selected['epoch'] != record['selection']['intensity']['epoch']:
                raise ValueError('Refitting the original objective must recover the archived checkpoint.')
        objectives = study_objectives(saved[1], source.label, split['fit'])
        seed_record = {}
        for arm, head in ARMS.items():
            criterion, weights = objectives[arm]
            model, selected = fit_fusion(args, head, source, split, saved, clips.label, banks,
                                          class_objective=criterion)
            posterior, score, gain = calibrated_head_forecasts(model, source, clips, split, saved, args.batch)
            forecasts[arm].append((posterior, score))
            seed_record[arm] = dict(selected, score_gain=gain, fitting_weights=weights)
            print(f'{cohort} seed {seed}: {arm} complete', flush=True)
        selections.append(seed_record)
    for arm, pairs in forecasts.items():
        posteriors, scores = zip(*pairs, strict=True)
        reference[arm], reference[f'{arm}_score'] = np.stack(posteriors), np.stack(scores)
    return dict(reference, cohort=np.asarray(cohort), seeds=np.asarray(seeds),
                selections=np.asarray(json.dumps(selections)),
                sizes=np.asarray([len(split[role]) for role in ('fit', 'tune', 'eval')]))


def summarize(evidence, resamples):
    y, group, intensity = (evidence[key] for key in ('y', 'groups', 'sentiment'))
    _, membership = np.unique(group, return_inverse=True)
    losses, rps, accuracy, mae, methods = {}, {}, {}, {}, {}
    for name in (*ARMS, *REFERENCES, *BASELINES):
        posterior = evidence[name].astype(np.float64)
        np.testing.assert_allclose(posterior.sum(axis=-1), 1., rtol=0, atol=2e-7)
        losses[name] = ranked_probability_score(posterior, y)
        rps[name] = losses[name].mean(axis=1)
        accuracy[name] = (posterior.argmax(-1) == y).mean(axis=1)
        mae[name] = np.abs(evidence[name + '_score'] - intensity).mean(axis=1)
        by_polarity = [float(rps[name][:, y == j].mean()) for j in range(3)]
        axes = (np.square(coordinates(posterior) - outcomes(y)) / 4.).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(axes.sum(), rps[name].mean(), atol=5e-8)
        methods[name] = dict(rps=float(rps[name].mean()), balanced_rps=float(np.mean(by_polarity)),
                             rps_by_polarity=by_polarity, accuracy=float(accuracy[name].mean()),
                             mae=float(mae[name].mean()), macro_f1=float(f1_by_run(y, posterior).mean()),
                             signed_rps=float(axes[0]), neutral_rps=float(axes[1]))
    preferred = 'text_matched'
    comparisons = {key: paired_reduction(rps[key] - rps[preferred], membership, resamples, 6)
                   for key in (*PRIMARY, *COMPONENTS, *BASELINES)}
    accuracy_gain = {key: paired_reduction(accuracy[preferred] - accuracy[key], membership, resamples, 6)
                     for key in PRIMARY}
    mae_gain = {key: paired_reduction(mae[key] - mae[preferred], membership, resamples, 6) for key in PRIMARY}
    interaction = rps['text_weight_only'] + rps['text_offset_only'] - rps['shared'] - rps[preferred]
    surprise = np.take_along_axis(evidence['anchor'], y[None, None, :, None], axis=-1).squeeze(-1) < 1. / 3.
    gain_by_view = losses['shared'] - losses[preferred]
    by_surprise = {key: dict(n_predictions=int(mask.sum()), rps_reduction=float(gain_by_view[mask].mean()))
                   for key, mask in (('low_anchor_probability', surprise), ('other', ~surprise))}
    return dict(cohort=evidence['cohort'].item(), seeds=evidence['seeds'].tolist(), resamples=resamples,
                n_videos=len(np.unique(group)), fitting_tuning_evaluation=evidence['sizes'].tolist(),
                methods=methods, rps_reductions=comparisons, accuracy_gains=accuracy_gain, mae_reductions=mae_gain,
                factorial_interaction=paired_reduction(interaction, membership, resamples, 2),
                by_anchor_probability=by_surprise, selections=json.loads(evidence['selections'].item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', type=Path)
    source.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', choices=RUN_NAME)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--first-seed', action='store_true', help='Run the first seed and replay the original fit.')
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and provide --cohort with --runs.')
    if args.runs:
        seeds = SEEDS[:1] if args.first_seed else SEEDS
        data = collect(args.runs, args.cohort, seeds, torch.device(args.device), verify_original=args.first_seed)
    else:
        with np.load(args.evidence, allow_pickle=False) as archive:
            data = dict(archive)
    report = summarize(data, args.resamples)
    if args.write_evidence:
        np.savez_compressed(args.write_evidence, **data)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for key, result in report['rps_reductions'].items():
        low, high = result['ci95_familywise']
        print(f'{key}: {result["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
