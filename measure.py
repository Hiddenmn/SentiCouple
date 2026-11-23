"""Summarize matched held-out predictions and resample at the original-video level."""

import argparse
import json
from pathlib import Path

import numpy as np


def macro_f1_from_confusion(matrix):
    denom = matrix.sum(-1) + matrix.sum(-2)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    return (2 * diagonal / np.maximum(denom, 1)).mean(-1)


def score_view(label, prediction, anchor, reference, reference_log, eligible):
    predicted = prediction.argmax(-1)
    right = predicted == label
    anchor_right = anchor.argmax(-1) == label
    true_probability = prediction[np.arange(len(label)), label].clip(1e-8, 1)
    confusion = np.bincount(3 * label + predicted, minlength=9).reshape(3, 3)
    stats = dict(accuracy=float(right.mean()), macro_f1=float(macro_f1_from_confusion(confusion)),
                 nll=float(-np.log(true_probability).mean()),
                 brier=float(((prediction - np.eye(3)[label]) ** 2).sum(-1).mean()),
                 corrected=int((right & ~anchor_right).sum()),
                 spoiled=int((~right & anchor_right).sum()),
                 anchor_errors=int((~anchor_right).sum()), anchor_correct=int(anchor_right.sum()),
                 recall=(confusion.diagonal() / confusion.sum(-1).clip(1)).tolist(),
                 confusion=confusion.tolist())
    if reference is not None and eligible.any():
        replacement_p = reference[np.arange(len(label)), label][eligible].clip(1e-8, 1)
        log_ratio = np.log(true_probability[eligible]) - np.log(replacement_p)
        stats.update(reference_n=int(eligible.sum()), genuine_vs_mixture_nll=float(log_ratio.mean()),
                     reference_drift=float(np.square(reference[eligible] - anchor[eligible]).sum(-1).mean()))
        if reference_log is not None:
            stats['genuine_vs_single_nll'] = float((np.log(true_probability[eligible]) - reference_log[eligible]).mean())
    return stats


def group_resampled_delta(label, groups, prediction, baseline, resamples, seed):
    _, membership = np.unique(groups, return_inverse=True)
    n_groups = membership.max() + 1
    # Cache each original video's confusion matrix for both block conditions.
    by_video = np.zeros((2, 2, n_groups, 9), np.int64)
    for m, posterior in enumerate((prediction, baseline)):
        for condition in range(2):
            pair = 3 * label + posterior[condition + 1].argmax(-1)
            np.add.at(by_video[m, condition], (membership, pair), 1)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=resamples)
    sampled_confusion = np.einsum('bg,mcgk->bmck', weights, by_video).reshape(-1, 2, 2, 3, 3)
    scores = macro_f1_from_confusion(sampled_confusion).mean(-1)
    delta = scores[:, 0] - scores[:, 1]
    return np.quantile(delta, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=23)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    meta = json.loads((args.run / 'run.json').read_text(encoding='utf-8'))
    arms = ['anchor'] + meta['config']['arms']
    views = meta['views']
    with np.load(args.run / 'predictions.npz', allow_pickle=False) as outputs:
        label, anchor, eligible = outputs['y'], outputs['anchor'], outputs['eligible']
        summary = {}
        for arm in arms:
            p = outputs[arm]
            reference = outputs.get(f'{arm}_reference', [None] * len(views))
            reference_log = outputs.get(f'{arm}_reference_log', [None] * len(views))
            stats = {view: score_view(label, p[i], anchor[i], reference[i], reference_log[i], eligible[i])
                     for i, view in enumerate(views)}
            primary = float(np.mean([stats[name]['macro_f1'] for name in ['block35', 'block65']]))
            record = dict(view=stats, block_macro_f1=primary)
            if arm != 'anchor':
                record['delta_vs_anchor_ci95'] = group_resampled_delta(
                    label, outputs['groups'], p, anchor, args.resamples, args.seed)
            if arm != 'fusion' and arm != 'anchor' and 'fusion' in arms:
                record['delta_vs_fusion_ci95'] = group_resampled_delta(
                    label, outputs['groups'], p, outputs['fusion'], args.resamples, args.seed)
            summary[arm] = record
            print(f'{arm:14s} native={stats["native"]["macro_f1"]:.4f} '
                  f'block35={stats["block35"]["macro_f1"]:.4f} block65={stats["block65"]["macro_f1"]:.4f} '
                  f'block-mean={primary:.4f} '
                  f'corrected/spoiled (65%)={stats["block65"]["corrected"]}/{stats["block65"]["spoiled"]}',
                  flush=True)
    (args.run / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
