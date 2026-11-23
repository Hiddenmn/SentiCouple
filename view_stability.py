"""Evaluate subset stability of the shared head's signed odds correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from aggregate import f1_by_run
from compare_benchmarks import load_metadata
from cremad_effects import POLARITIES, balanced_reduction
from data import load_clips, observation_views
from networks import AudiovisualResidual, TextAnchor
from published_inference import load_weights
from score_anchor_effects import SEEDS
from train import BatchSource, index_batches, visual_component_options


RUNS = {
    'cremad': ('cremad_heads_seed',),
    'ravdess': tuple(f'ravdess_fold{fold}_seed' for fold in range(4)),
    'sims_v2': ('sims_v2_compact_pooled_seed',),
    'omg': ('omg_pointwise_heads_seed',),
}
GRID = (0., .125, .25, .5, .75, 1., 1.5, 2., 3., 4., 6., 8., 12., 16.)
ARM = 'intensity'
REFERENCES = ('shared', 'support_threshold', 'signed_average')
FIELDS = ('anchor', 'shared', 'update', 'signed_average', 'posterior_average')


def restricted_views(views):
    """Remove each chronological quarter of the observed support from each gap."""
    masks = views[1:3, 1]
    support = masks.sum(-1, keepdims=True)
    rank = np.cumsum(masks, axis=-1) - 1
    block_id = 4 * rank // np.maximum(support, 1)
    restrictions = [masks & ((block_id != block) | (support < 4)) for block in range(4)]
    subviews = np.zeros((4, 2, *views.shape[1:]), dtype=bool)
    subviews[:, :, 1] = np.stack(restrictions)
    if np.any(subviews[:, :, 1] & ~masks):
        raise ValueError('A restricted mask reveals an unobserved bin.')
    return np.concatenate((views, subviews.reshape(8, *views.shape[1:])))


@torch.inference_mode()
def forecast(anchor, model, source, index, batch_size):
    views = ((1, 2), (4, 5), (6, 7), (8, 9), (10, 11))
    all_results = []
    for short_views in views:
        view_results = []
        for condition in short_views:
            batches = []
            for view, batch in index_batches(index, batch_size, source.device, condition):
                context, prior = source.anchor(anchor, view, batch)
                logits = source.fusion(model, context, prior, view, batch, batch)
                signed = logits[:, 2] - logits[:, 0]
                update = signed - (prior[:, 2] - prior[:, 0])
                batches.append(torch.column_stack((prior.softmax(-1), logits.softmax(-1),
                                                   update, signed.sigmoid())))
            view_results.append(torch.cat(batches))
        all_results.append(torch.stack(view_results))
    values = torch.stack(all_results).cpu().numpy()
    return dict(anchor=values[0, ..., :3], shared=values[0, ..., 3:6], update=values[..., 6],
                signed_average=values[..., 7].astype(np.float64).mean(0),
                posterior_average=values[..., 3:6].astype(np.float64).mean(0))


def replay_fold(paths, seeds, device):
    metadata = load_metadata(paths)
    settings = metadata[0]['config']
    for seed, path, run in zip(seeds, paths, metadata, strict=True):
        config = run['config']
        if config['seed'] != seed or any(settings[key] != config[key]
                                        for key in settings if key not in ('seed', 'output')):
            raise ValueError(f'Mismatched checkpoint settings for {path.name}.')
    if settings['inputs'] != 'visual' or settings['bins'] != 24:
        raise ValueError('This study expects a visual residual model with 24 bins.')
    clips, split = load_clips(settings['data'], settings['bins'], settings['split'], settings['split_seed'],
                              settings['fold'], group_reference=settings['group_reference'],
                              exclude_eval=settings['exclude_eval'], eval_groups=settings['eval_groups'],
                              neutral_margin=settings['neutral_margin'])
    views = observation_views(clips, [.35, .65], settings['mask_seed'])
    views[:, 0] = False
    source = BatchSource(clips, restricted_views(views), split['fit'], device, visual_only=True,
                         stream_text=settings['stream_text'])
    fold = {field: getattr(clips, field)[split['eval']] for field in ('ids', 'groups')}
    fold['eval_masks'] = views[:, :, split['eval']]
    records = {'tune': [], 'eval': []}
    for role in records:
        index = split[role]
        fold[f'{role}_y'] = clips.label[index]
        fold[f'{role}_support'] = views[1:3, 1, index].sum(-1)

    for run in paths:
        anchor = TextAnchor(source.text.shape[-1], settings['bins'], settings['width'], visual_only=True)
        model = AudiovisualResidual(0, source.vision.shape[-1], settings['width'], text_query=False,
                                    visual_encoding=settings['visual_encoding'], **visual_component_options(ARM))
        load_weights(anchor, run / 'anchor.pt', device)
        load_weights(model, run / f'{ARM}.pt', device)
        packet = read_predictions(run)
        np.testing.assert_array_equal(packet['ids'], fold['ids'])
        np.testing.assert_array_equal(packet['groups'], fold['groups'])
        np.testing.assert_array_equal(packet['y'], fold['eval_y'])
        np.testing.assert_array_equal(packet['masks'], fold['eval_masks'])
        with np.load(run / 'tuning.npz', allow_pickle=False) as saved:
            tune = {key: saved[key] for key in ('y', 'anchor', ARM)}
        np.testing.assert_array_equal(tune['y'], fold['tune_y'])
        np.testing.assert_array_equal(packet[ARM][-1], packet['anchor'][-1])
        for role, reference in (('tune', tune), ('eval', packet)):
            raw = forecast(anchor, model, source, split[role], settings['batch'])
            for name, key in (('anchor', 'anchor'), ('shared', ARM)):
                np.testing.assert_allclose(raw[name], reference[key][1:3], rtol=2e-5, atol=3e-6,
                                           err_msg=f'{run.name}: {role} {name}')
            empty = fold[f'{role}_support'] == 0
            np.testing.assert_array_equal(raw['shared'][empty], raw['anchor'][empty])
            np.testing.assert_array_equal(raw['update'][:, empty], 0.)
            records[role].append(raw)
        records['eval'][-1]['independent'] = packet['independent_intensity'][1:3]
        print(f'{run.name}: {len(split["tune"])} tuning and {len(split["eval"])} evaluation clips', flush=True)
    for role, runs in records.items():
        for field in runs[0]:
            fold[f'{role}_{field}'] = np.stack([run[field] for run in runs])
    if device.type == 'cuda':
        del source, anchor, model
        torch.cuda.empty_cache()
    return fold


def collect(runs, cohort, seeds, device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evidence = dict(cohort=np.asarray(cohort), seeds=np.asarray(seeds))
    for i, prefix in enumerate(RUNS[cohort]):
        paths = [runs / f'{prefix}{seed}' for seed in seeds]
        fold = replay_fold(paths, seeds, device)
        evidence.update({f'fold{i}_{name}': value for name, value in fold.items()})
        if cohort in ('sims_v2', 'omg'):
            with np.load(runs / f'{cohort}_published_predictions.npz', allow_pickle=False) as archive:
                for key in ('ids', 'groups', 'y'):
                    np.testing.assert_array_equal(archive[key], fold['eval_y' if key == 'y' else key])
                np.testing.assert_array_equal(archive['masks'], fold['eval_masks'])
                rows = [archive['seeds'].tolist().index(seed) for seed in seeds]
                reference = archive['methods'].tolist().index('fixed_pooling' if cohort == 'sims_v2' else 'pointwise')
                np.testing.assert_allclose(archive['forecast'][reference, rows], fold['eval_shared'],
                                           rtol=2e-5, atol=3e-6)
                for name in ('tfr_joint', 'lnln_joint', 'rmnet_core'):
                    idx = archive['methods'].tolist().index(name)
                    evidence[f'published_{name}'] = archive['forecast'][idx, rows]
    return evidence


def sensitivity(update, support):
    departure = update[1:].astype(np.float64) - update[0]
    return np.where(support >= 4, np.sqrt(np.square(departure).mean(axis=0)), 0.)


def support_scales(tune_scale, tune_support, support):
    pooled = np.zeros(support.shape, dtype=np.float64)
    for gap, (scales, counts) in enumerate(zip(tune_scale, tune_support, strict=True)):
        eligible = counts >= 4
        reference = np.mean(scales[eligible]) if eligible.any() else 0.
        for start, end in ((4, 8), (9, 16), (17, 24)):
            cell = (counts >= start) & (counts <= end)
            destination = (support[gap] >= start) & (support[gap] <= end)
            pooled[gap, destination] = np.mean(scales[cell]) if cell.any() else reference
    return pooled


def on_polar_mass(posterior, positive_given_polar, active):
    polar = (1. - posterior[..., 1].astype(np.float64))
    positive = polar * positive_given_polar
    replaced = np.stack((polar - positive, posterior[..., 1], positive), axis=-1)
    return np.where(active[..., None], replaced, posterior)


def threshold(posterior, update, scale, multiplier):
    step = np.minimum(multiplier * scale, np.abs(update))
    deltas = np.sign(update) * step
    logs = np.log(posterior[..., (0, 2)].astype(np.float64).clip(1e-35))
    odds = logs[..., 1] - logs[..., 0] - deltas
    conditional = (1. + np.tanh(odds / 2.)) / 2.
    return on_polar_mass(posterior, conditional, step > 0), step


def fit_threshold(tune, update, scale, y):
    weights = 1. / np.bincount(y, minlength=3)[y]
    risks = [np.average(ranked_probability_score(threshold(tune, update, scale, k)[0], y).mean(0),
                        weights=weights) for k in GRID]
    return GRID[int(np.argmin(risks))]


def apply_fold(fold, seeds):
    output = {name: [] for name in ('adaptive', *REFERENCES, 'anchor', 'independent', 'posterior_average')}
    selections, steps, scales = [], [], []
    for i in range(len(seeds)):
        tune_y = fold['tune_y']
        tune_support, eval_support = fold['tune_support'], fold['eval_support']
        tune, held = ({name: fold[f'{role}_{name}'][i] for name in FIELDS} for role in ('tune', 'eval'))
        tune_scale, eval_scale = (sensitivity(values['update'], support) for values, support in
                                 ((tune, tune_support), (held, eval_support)))
        scale_pairs = (
            ('adaptive', tune_scale, eval_scale),
            ('support_threshold', support_scales(tune_scale, tune_support, tune_support),
             support_scales(tune_scale, tune_support, eval_support)),
        )
        chosen = {}
        for name, tune_threshold, test_threshold in scale_pairs:
            k = fit_threshold(tune['shared'], tune['update'][0], tune_threshold, tune_y)
            prediction, step = threshold(held['shared'], held['update'][0], test_threshold, k)
            np.testing.assert_array_equal(prediction[..., 1], held['shared'][..., 1])
            np.testing.assert_array_equal(prediction[eval_support < 4], held['shared'][eval_support < 4])
            output[name].append(prediction)
            chosen[name] = k
            if name == 'adaptive':
                steps.append(step)
                scales.append(eval_scale)
        selections.append(chosen)
        output['signed_average'].append(on_polar_mass(held['shared'], held['signed_average'], eval_support >= 4))
        for name in ('anchor', 'shared', 'posterior_average'):
            output[name].append(held[name])
        output['independent'].append(fold['eval_independent'][i])
    return {name: np.stack(values) for name, values in output.items()}, selections, np.stack(steps), np.stack(scales)


def signed_contraction_ceiling(posterior, anchor, support, label):
    """The most RPS a label-aware contraction could remove at fixed neutral mass."""
    remaining_mass = 1. - posterior[..., 1]
    full = posterior[..., 2] - posterior[..., 0]
    anchor_polar_total = anchor[..., 2] + anchor[..., 0]
    anchor_balance = (anchor[..., 2] - anchor[..., 0]) / anchor_polar_total.clip(1e-35)
    target = label - 1.
    origin = remaining_mass * anchor_balance
    optimum = np.clip(target, np.minimum(full, origin), np.maximum(full, origin))
    gain = (np.square(full - target) - np.square(optimum - target)) / 4.
    return np.where(support >= 4, gain, 0.)


def summarize(evidence, resamples):
    cohort = evidence['cohort'].item()
    seeds = evidence['seeds'].tolist()
    forecasts, selections, diagnostics, labels, sources, ids, ceilings = {}, [], [], [], [], [], []
    for i in range(len(RUNS[cohort])):
        prefix = f'fold{i}_'
        fold = {key[len(prefix):]: value for key, value in evidence.items() if key.startswith(prefix)}
        predictions, selection, step, scale = apply_fold(fold, seeds)
        for name, values in predictions.items():
            forecasts.setdefault(name, []).append(values)
        selections.append(selection)
        diagnostics.append(dict(eligible=float((fold['eval_support'] >= 4).mean()),
                                trimmed=float((step > 0).mean()),
                                mean_log_odds_trim=float(step.mean()), mean_sensitivity=float(scale.mean())))
        labels.append(fold['eval_y'])
        sources.append(fold['groups'])
        ids.append(fold['ids'])
        ceilings.append(signed_contraction_ceiling(predictions['shared'].astype(np.float64),
                                                   predictions['anchor'].astype(np.float64),
                                                   fold['eval_support'], fold['eval_y']))
    if len(np.unique(np.concatenate(ids))) != sum(map(len, ids)):
        raise ValueError('The held-out folds must contain distinct clips.')
    if len(np.unique(np.concatenate(sources))) != sum(len(np.unique(x)) for x in sources):
        raise ValueError('A held-out actor or video appears in multiple folds.')
    y, groups = np.concatenate(labels), np.concatenate(sources)
    population, membership = np.unique(groups, return_inverse=True)
    forecasts = {name: np.concatenate(folds, axis=2) for name, folds in forecasts.items()}
    forecasts.update({key: value for key, value in evidence.items() if key.startswith('published_')})
    results, rps = {}, {}
    for name, posterior in forecasts.items():
        np.testing.assert_allclose(posterior.sum(-1), 1., rtol=0, atol=2e-7)
        losses = ranked_probability_score(posterior.astype(np.float64), y)
        correct = posterior.argmax(-1) == y
        rps[name] = losses.mean(axis=1)
        by_class = np.stack([losses[..., y == level].mean(axis=(0, 2)) for level in range(3)])
        results[name] = dict(rps=float(losses.mean()), balanced_rps=float(by_class.mean()),
                             balanced_rps_by_gap=by_class.mean(0).tolist(), rps_by_class=by_class.mean(1).tolist(),
                             accuracy=float(correct.mean()), macro_f1=float(f1_by_run(y, posterior).mean()),
                             accuracy_by_class=[float(correct[..., y == level].mean()) for level in range(3)])
    rng = np.random.default_rng(29)
    seed_draws = rng.multinomial(len(seeds), np.repeat(1 / len(seeds), len(seeds)), resamples)
    group_draws = rng.multinomial(len(population), np.repeat(1 / len(population), len(population)), resamples)
    reductions = {name: balanced_reduction(rps[name] - rps['adaptive'], y, membership, seed_draws,
                                           group_draws, 6, include_polarity=name == 'shared')
                  for name in REFERENCES}
    ceiling = np.concatenate(ceilings, axis=2)
    ceiling_by_class = [float(ceiling[..., y == level].mean()) for level in range(3)]
    return dict(cohort=cohort, n_clips=len(y), n_groups=len(population), seeds=seeds, resamples=resamples,
                class_order=list(POLARITIES), class_counts=np.bincount(y, minlength=3).tolist(),
                grid=list(GRID), selections_by_fold=selections, diagnostics_by_fold=diagnostics,
                methods=results, balanced_rps_reductions=reductions,
                contraction_ceiling=dict(balanced_rps=float(np.mean(ceiling_by_class)), by_class=ceiling_by_class))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', choices=RUNS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--first-seed', action='store_true', help='Run one checkpoint per fold for a smoke test.')
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and supply --cohort with --runs.')
    if args.runs:
        evidence = collect(args.runs, args.cohort, SEEDS[:1] if args.first_seed else SEEDS, torch.device(args.device))
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    report = summarize(evidence, args.resamples)
    if args.write_evidence:
        np.savez_compressed(args.write_evidence, **evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, result in report['balanced_rps_reductions'].items():
        low, high = result['ci95_familywise']
        print(f'{name}: {result["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
