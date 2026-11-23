"""Test group-cross-fitted native targets for signed and neutral gap predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F

from ablate import ranked_probability_score
from aggregate import f1_by_run
from compare_benchmarks import load_metadata
from data import shuffle_within_score_ranks
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_seed_effects import BASELINES, run_paths, verify_training
from score_anchor_effects import SEEDS
from text_weighted_loss import REFERENCES, RUN_NAME, anchor_for, load_references, source_for
from train import BatchSource, anchor_outputs, calibrated_head_forecasts, fit_anchor, fit_fusion, predict_fusion


N_FOLDS = 4
ARMS = {
    'signed_crossfit': ('intensity', (0,)),
    'neutral_crossfit': ('intensity', (1,)),
    'both_crossfit': ('intensity', (0, 1)),
    'signed_hard': ('intensity', (0,)),
    'signed_shuffled': ('intensity', (0,)),
    'signed_included': ('intensity', (0,)),
    'signed_independent': ('independent_intensity', (0,)),
}
CONTRAST_FAMILIES = (('shared',),
                     ('signed_hard', 'signed_shuffled', 'signed_included', 'signed_independent'),
                     ('neutral_crossfit', 'both_crossfit'), tuple(BASELINES))


def native_forecasts(clips, split, views, metadata, device):
    """Fit only on source-video complements and keep forecasts for fitting clips."""
    fit = split['fit']
    group = clips.groups[fit]
    columns = np.arange(len(fit))
    held_fold = np.full(len(fit), -1, dtype=np.int64)
    per_fold = list(GroupKFold(n_splits=N_FOLDS).split(columns, groups=group))
    posterior = np.empty((len(metadata), len(per_fold), len(fit), 3), dtype=np.float32)
    prior = np.empty_like(posterior)
    records = []

    for fold, (complement, withheld) in enumerate(per_fold):
        if np.intersect1d(group[complement], group[withheld]).size or np.any(held_fold[withheld] >= 0):
            raise ValueError('Teacher folds must keep source videos intact and cover each clip once.')
        held_fold[withheld] = fold
        inner_fit = fit[complement]
        inner = dict(fit=inner_fit, tune=split['tune'])
        source = BatchSource(clips, views[:1], inner_fit, device, visual_only=True,
                             stream_text=metadata[0]['config']['stream_text'])
        banks = np.full((1, len(clips.ids), 1), -1, dtype=np.int32)
        fold_records = []
        for i, record in enumerate(metadata):
            args = argparse.Namespace(**record['config'])
            anchor, anchor_choice = fit_anchor(args, source, inner, clips.label, tuning_views=(0,))
            saved = anchor_outputs(anchor, source, args.batch)
            teacher, choice = fit_fusion(args, 'intensity', source, inner, saved, clips.label, banks,
                                         tuning_views=(0,))
            values, _, _, _ = predict_fusion(teacher, source, fit, saved, args.batch, conditions=(0,))
            posterior[i, fold] = values[0]
            prior[i, fold] = saved[1][0, fit].softmax(-1).cpu().numpy()
            fold_records.append(dict(seed=args.seed, anchor_epoch=anchor_choice['epoch'],
                                     epoch=choice['epoch'], native_tune_rps=-choice['tune_score']))
            print(f'Teacher fold {fold + 1}/{N_FOLDS}, seed {args.seed} complete', flush=True)
        records.append(dict(teacher_training_clips=len(complement), target_clips=len(withheld),
                            withheld_videos=np.unique(group[withheld]).tolist(), choices=fold_records))

    if np.any(held_fold < 0):
        raise ValueError('Every fitting clip must receive an out-of-fold forecast.')
    included_fold = (held_fold + 1) % N_FOLDS
    for fold, (complement, _) in enumerate(per_fold):
        if not np.isin(columns[included_fold == fold], complement).all():
            raise ValueError('The inclusion control must use a teacher trained on each assigned video.')
    return dict(teacher_crossfit=posterior[:, held_fold, columns], teacher_anchor=prior[:, held_fold, columns],
                teacher_included=posterior[:, included_fold, columns], teacher_fold=held_fold,
                teacher_records=np.asarray(json.dumps(records)), fit_y=clips.label[fit],
                fit_ids=clips.ids[fit], fit_groups=group)


def fitting_targets(clips, fit, teacher, row, seed):
    hard = outcomes(clips.label[fit])
    crossfit = coordinates(teacher['teacher_crossfit'][row])
    included = coordinates(teacher['teacher_included'][row])
    shuffled = shuffle_within_score_ranks(crossfit, clips.label[fit], clips.sentiment[fit],
                                         np.arange(len(fit)), np.random.default_rng(seed + 739))
    soft = (crossfit + hard) * .5
    return dict(signed_crossfit=soft, neutral_crossfit=soft, both_crossfit=soft, signed_hard=hard,
                signed_shuffled=(shuffled + hard) * .5, signed_included=(included + hard) * .5,
                signed_independent=soft)


def gap_objective(source, fit, target, axes):
    auxiliary = torch.zeros((len(source.label), 2), device=source.device, dtype=torch.float32)
    auxiliary[fit] = torch.as_tensor(target, device=source.device, dtype=auxiliary.dtype)
    eligible = source.views[:, 1].any(-1)
    columns = list(axes)

    def objective(logits, labels, view, clips):
        p = logits.softmax(-1)
        values = torch.stack((p[:, 2] - p[:, 0], p[:, 1]), dim=-1)
        errors = (values[:, columns] - auxiliary[clips][:, columns]).square().sum(-1)
        active = (view > 0) & (view < 3) & eligible[view, clips]
        return F.cross_entropy(logits, labels) + .25 * (errors * active).sum() / active.sum().clamp_min(1)

    return objective


def collect(runs, cohort, seeds, device):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    name = RUN_NAME[cohort]
    original_runs = run_paths(runs, seeds, cohort)[name]
    metadata = load_metadata(original_runs)
    verify_training(name, metadata, seeds, cohort)
    reference = load_references(runs, cohort, seeds, original_runs)
    settings = metadata[0]['config']
    for path, record in zip(original_runs, metadata, strict=True):
        config = record['config']
        if any(config[key] != value for key, value in settings.items() if key not in ('seed', 'output')):
            raise ValueError(f'Training settings differ for {path.name}.')
        if config['score_anchor'] != 'class-mean' or not config['calibrate_score_gain']:
            raise ValueError('Use the original calibrated visual head.')

    clips, split, views, source = source_for(settings, device)
    fit, held = split['fit'], split['eval']
    for key, values in (('ids', clips.ids[held]), ('groups', clips.groups[held]), ('y', clips.label[held]),
                        ('sentiment', clips.sentiment[held]), ('masks', views[:, :, held])):
        np.testing.assert_array_equal(reference[key], values, err_msg=key)
    teacher = native_forecasts(clips, split, views, metadata, device)
    banks = np.full((len(views), len(clips.ids), 1), -1, dtype=np.int32)
    forecasts = {arm: [] for arm in ARMS}
    choices = []
    for row, (run, record) in enumerate(zip(original_runs, metadata, strict=True)):
        args = argparse.Namespace(**record['config'])
        saved = anchor_for(run, source, split, record['config'], reference, row)
        targets = fitting_targets(clips, fit, teacher, row, args.seed)
        selected = {}
        for arm, (head, axes) in ARMS.items():
            objective = gap_objective(source, fit, targets[arm], axes)
            model, choice = fit_fusion(args, head, source, split, saved, clips.label, banks,
                                        class_objective=objective)
            posterior, score, gain = calibrated_head_forecasts(model, source, clips, split, saved, args.batch)
            if choice['parameters'] != record['selection'][head]['parameters']:
                raise ValueError('The transfer loss must leave the deployed head size unchanged.')
            forecasts[arm].append((posterior, score))
            selected[arm] = dict(choice, score_gain=gain)
            print(f'{cohort} seed {args.seed}: {arm} complete', flush=True)
        choices.append(selected)

    for arm, pairs in forecasts.items():
        probabilities, scores = zip(*pairs, strict=True)
        reference[arm], reference[arm + '_score'] = np.stack(probabilities), np.stack(scores)
    return dict(reference, **teacher, seeds=np.asarray(seeds), cohort=np.asarray(cohort),
                selections=np.asarray(json.dumps(choices)), n_fit_groups=np.asarray(len(np.unique(clips.groups[fit]))),
                sizes=np.asarray([len(split[key]) for key in ('fit', 'tune', 'eval')]))


def summarize(evidence, resamples):
    y, group, intensity = (evidence[key] for key in ('y', 'groups', 'sentiment'))
    _, membership = np.unique(group, return_inverse=True)
    methods, rps, accuracy, mae = {}, {}, {}, {}
    for name in (*ARMS, *REFERENCES, *BASELINES):
        posterior = np.ascontiguousarray(evidence[name], dtype=np.float64)
        np.testing.assert_allclose(posterior.sum(-1), 1., rtol=0, atol=2e-7)
        rps[name] = ranked_probability_score(posterior, y).mean(axis=1)
        accuracy[name] = (posterior.argmax(-1) == y).mean(axis=1)
        mae[name] = np.abs(evidence[name + '_score'] - intensity).mean(axis=1)
        squared = np.square(coordinates(posterior) - outcomes(y)) / 4.
        signed, neutral = squared.mean(axis=(0, 1, 2))
        np.testing.assert_allclose(signed + neutral, rps[name].mean(), atol=5e-8)
        by_polarity = [float(rps[name][:, y == label].mean()) for label in range(3)]
        methods[name] = dict(rps=float(rps[name].mean()), rps_by_polarity=by_polarity,
                             balanced_rps=float(np.mean(by_polarity)), signed_rps=float(signed),
                             neutral_rps=float(neutral), accuracy=float(accuracy[name].mean()),
                             mae=float(mae[name].mean()), macro_f1=float(f1_by_run(y, posterior).mean()))
    teacher = {}
    fit_y = evidence['fit_y']
    for name in ('teacher_crossfit', 'teacher_anchor', 'teacher_included'):
        posterior = np.ascontiguousarray(evidence[name], dtype=np.float64)
        np.testing.assert_allclose(posterior.sum(-1), 1., rtol=0, atol=2e-7)
        loss = np.square(coordinates(posterior) - outcomes(fit_y)) / 4.
        teacher[name] = dict(rps=float(loss.sum(-1).mean()), signed_rps=float(loss[..., 0].mean()),
                             neutral_rps=float(loss[..., 1].mean()))
    preferred = 'signed_crossfit'
    contrasts = {name: paired_reduction(rps[name] - rps[preferred], membership, resamples, 2 * len(family))
                 for family in CONTRAST_FAMILIES for name in family}
    return dict(cohort=evidence['cohort'].item(), seeds=evidence['seeds'].tolist(), resamples=resamples,
                fitting_tuning_evaluation=evidence['sizes'].tolist(), n_videos=len(np.unique(group)),
                n_fit_groups=int(evidence['n_fit_groups'].item()), methods=methods, teacher=teacher,
                rps_reductions=contrasts, accuracy_gain=paired_reduction(accuracy[preferred] - accuracy['shared'],
                                                                        membership, resamples, 2),
                mae_reduction=paired_reduction(mae['shared'] - mae[preferred], membership, resamples, 2),
                teacher_folds=json.loads(evidence['teacher_records'].item()),
                selections=json.loads(evidence['selections'].item()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', choices=RUN_NAME)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--first-seed', action='store_true')
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and provide --cohort with --runs.')
    if args.runs:
        evidence = collect(args.runs, args.cohort, SEEDS[:1] if args.first_seed else SEEDS, torch.device(args.device))
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    report = summarize(evidence, args.resamples)
    if args.write_evidence:
        partial = args.write_evidence.with_suffix('.npz.part')
        with partial.open('wb') as pending:
            np.savez_compressed(pending, **evidence)
        partial.replace(args.write_evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, result in report['rps_reductions'].items():
        low, high = result['ci95_familywise']
        print(f'{name}: {result["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
