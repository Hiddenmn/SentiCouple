"""Combine repeated training seeds or disjoint held-out folds without mixing their sampling units."""

import argparse
import json
from pathlib import Path

import numpy as np

from measure import macro_f1_from_confusion


def combine(packets, key, mode, clip_axis):
    values = [packet[key] for packet in packets]
    if mode == 'restarts':
        return np.stack(values)
    return np.expand_dims(np.concatenate(values, axis=clip_axis), 0)


def bootstrap_f1(label, predictions, baseline, memberships, weights):
    matrices = []
    for posterior in (predictions, baseline):
        by_video = np.zeros((*posterior.shape[:2], weights.shape[1], 9), np.int64)
        for seed in range(posterior.shape[0]):
            for condition in range(posterior.shape[1]):
                encoded = 3 * label + posterior[seed, condition].argmax(-1)
                np.add.at(by_video[seed, condition], (memberships, encoded), 1)
        confusion = np.einsum('bg,scgi->bsci', weights, by_video).reshape(
            len(weights), *posterior.shape[:2], 3, 3)
        matrices.append(macro_f1_from_confusion(confusion).mean(axis=(1, 2)))
    return np.quantile(matrices[0] - matrices[1], [.025, .975]).tolist()


def bootstrap_mean(values, membership, weights):
    total, count = np.zeros(weights.shape[1]), np.zeros(weights.shape[1])
    for block in values.reshape(-1, len(membership)):
        finite = np.isfinite(block)
        np.add.at(total, membership[finite], block[finite])
        np.add.at(count, membership[finite], 1)
    totals, counts = weights @ total, weights @ count
    valid = counts > 0
    return np.quantile(totals[valid] / counts[valid], [.025, .975]).tolist() if valid.any() else None


def annotation_agreement(source, ids, label):
    import pickle

    # This source is the checksum-verified MMSA distribution used for training.
    with source.open('rb') as stream:
        packets = pickle.load(stream)
    text_label = {}
    for packet in packets.values():
        annotations = np.sign(np.asarray(packet['regression_labels_T']).reshape(-1)).astype(int) + 1
        text_label.update(zip(np.asarray(packet['id']).astype(str).reshape(-1), annotations))
    return np.asarray([text_label[sample_id] for sample_id in ids]) == label


def subgroup_counts(agree, correct, anchor_correct):
    result = {}
    for name, membership in [('agreement', agree), ('disagreement', ~agree)]:
        corrected = (correct[..., membership] & ~anchor_correct[..., membership]).sum(-1)
        spoiled = (~correct[..., membership] & anchor_correct[..., membership]).sum(-1)
        result[name] = dict(n=int(membership.sum()), corrected=corrected.tolist(), spoiled=spoiled.tolist())
    return result


def f1_by_run(label, posterior):
    expected = 3 * label
    confusion = np.stack([np.bincount(expected + prediction, minlength=9).reshape(3, 3)
                          for prediction in posterior.argmax(-1).reshape(-1, len(label))])
    return macro_f1_from_confusion(confusion).reshape(posterior.shape[:2])


def aggregate(args):
    packets = []
    for run in args.runs:
        with np.load(run / 'predictions.npz', allow_pickle=False) as packet:
            packets.append(dict(packet))
    if args.mode == 'restarts':
        for packet in packets[1:]:
            for key in ('ids', 'groups', 'y', 'masks'):
                np.testing.assert_array_equal(packet[key], packets[0][key])
    ids, groups, label = (combine(packets, key, args.mode, 0)[0] for key in ('ids', 'groups', 'y'))
    if len(ids) != len(np.unique(ids)):
        raise ValueError('Held-out folds must contain disjoint clips.')
    if args.mode == 'folds':
        group_count = sum(len(np.unique(packet['groups'])) for packet in packets)
        if len(np.unique(groups)) != group_count:
            raise ValueError('Held-out folds must contain disjoint original videos.')
    _, membership = np.unique(groups, return_inverse=True)
    n_groups = membership.max() + 1
    rng = np.random.default_rng(23)
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=args.resamples)
    states = {arm: combine(packets, arm, args.mode, 1)[:, 1:3] for arm in args.arms}
    anchor_right = states['anchor'].argmax(-1) == label[None, None, :]
    fusion_f1 = f1_by_run(label, states['fusion']).mean()
    agree = annotation_agreement(args.sims, ids, label) if args.sims else None
    summary = {}
    for arm, probs in states.items():
        selected = probs[..., np.arange(len(label)), label].clip(1e-8, 1)
        correct = probs.argmax(-1) == label[None, None, :]
        scores = f1_by_run(label, probs)
        result = dict(f1=float(scores.mean()), f1_by_seed_and_condition=scores.tolist(),
                      nll=float(-np.log(selected).mean()),
                      brier=float(np.square(probs - np.eye(3)[label]).sum(-1).mean()),
                      corrected=(correct & ~anchor_right).sum(-1).tolist(),
                      spoiled=(~correct & anchor_right).sum(-1).tolist())
        if arm != 'fusion':
            result['f1_minus_fusion'] = float(scores.mean() - fusion_f1)
            result['f1_minus_fusion_ci'] = bootstrap_f1(
                label, probs, states['fusion'], membership, weights)
        if all(f'{arm}_reference' in packet for packet in packets):
            reference = combine(packets, f'{arm}_reference', args.mode, 1)[:, 1:3]
            difference = np.log(selected) - np.log(reference[..., np.arange(len(label)), label].clip(1e-8, 1))
            result['reference_coverage'] = float(np.isfinite(difference).mean())
            result['reference_drift'] = float(np.nanmean(np.square(reference - states['anchor']).sum(-1)))
            result['genuine_vs_mixture_nll'] = float(np.nanmean(difference))
            result['genuine_vs_mixture_ci'] = bootstrap_mean(difference, membership, weights)
            log_key = f'{arm}_reference_log'
            supported = [i for i, packet in enumerate(packets) if log_key in packet]
            if supported and (args.mode == 'restarts' or len(supported) == len(packets)):
                log_reference = combine([packets[i] for i in supported], log_key, args.mode, 1)[:, 1:3]
                genuine = selected[supported] if args.mode == 'restarts' else selected
                difference = np.log(genuine) - log_reference
                result['genuine_vs_single_nll'] = float(np.nanmean(difference))
                result['genuine_vs_single_ci'] = bootstrap_mean(difference, membership, weights)
                result['genuine_vs_single_runs'] = len(supported)
        if agree is not None:
            result['subgroups'] = subgroup_counts(agree, correct, anchor_right)
        summary[arm] = result
        print(f'{arm:14s} macro-F1={result["f1"]:.4f} NLL={result["nll"]:.4f} '
              f'Brier={result["brier"]:.4f}', flush=True)
    record = dict(n_clips=len(ids), n_videos=int(n_groups), mode=args.mode, runs=len(packets),
                  bootstrap_replicates=args.resamples, arms=summary)
    args.output.write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--mode', required=True, choices=['restarts', 'folds'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arms', nargs='+', default=['anchor', 'fusion', 'shrinkage', 'mean', 'moment', 'random_moment'])
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--sims', type=Path, help='Use the published text annotation for subgroup analysis.')
    aggregate(parser.parse_args())
