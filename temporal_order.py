"""Probe observed-frame ordering under the matched uniform-class-weight benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score
from benchmark import model_inputs
from compare_benchmarks import load_matched, load_metadata
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_inference import predictor
from published_seed_effects import run_paths, verify_training
from score_anchor_effects import SEEDS
from text_weighted_loss import RUN_NAME, source_for


CONDITIONS = {'chronological': (0,), 'reverse': (1,), 'rotate': (2,), 'random': (3, 4, 5)}
MODELS = ('compact', 'rmnet')


def permutations(visible):
    gap_count, n_clips, bins = visible.shape
    identity = np.arange(bins, dtype=np.int64)
    order = np.broadcast_to(identity, (6, gap_count, n_clips, bins)).copy()
    rng = np.random.default_rng(4153)
    for gap, masks in enumerate(visible):
        for clip, mask in enumerate(masks):
            rank = np.flatnonzero(mask)
            order[1, gap, clip, rank] = rank[::-1]
            order[2, gap, clip, rank] = np.roll(rank, len(rank) // 2)
            for draw in range(3, 6):
                order[draw, gap, clip, rank] = rng.permutation(rank)
    np.testing.assert_array_equal(np.sort(order, axis=-1), np.broadcast_to(identity, order.shape))
    np.testing.assert_array_equal(order[:, ~visible], np.broadcast_to(identity, visible.shape)[~visible][None, :]
                                  .repeat(len(order), axis=0))
    return order


@torch.inference_mode()
def infer(forecast, source, held, order, batch_size):
    probabilities = np.empty((len(order), 2, len(held), 3), np.float32)
    scores = np.empty(probabilities.shape[:-1], np.float32)
    for gap in range(2):
        for start in range(0, len(held), batch_size):
            end = min(start + batch_size, len(held))
            clips = torch.as_tensor(held[start:end], device=source.device)
            view = torch.full_like(clips, gap + 1)
            observed, mask = model_inputs(source, view, clips)
            vision = observed[-1]
            permutations_batch = order[:, gap, start:end]
            for index, order_batch in enumerate(permutations_batch):
                shuffled = vision.gather(1, order_batch[..., None].expand_as(vision))
                p, s = forecast(((*observed[:-1], shuffled), mask))
                probabilities[index, gap, start:end] = p.cpu().numpy()
                scores[index, gap, start:end] = s.cpu().numpy()
    return probabilities, scores


def collect(runs, cohort, seeds, device, upstream):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    candidate = RUN_NAME[cohort]
    choices = run_paths(runs, seeds, cohort)
    paths = {'compact': choices[candidate], 'rmnet': [runs / f'{cohort}_rmnet_uniform_seed{s}' for s in seeds]}
    metadata = {name: load_metadata(sequence) for name, sequence in paths.items()}
    verify_training(candidate, metadata['compact'], seeds, cohort)
    verify_training('rmnet_core', metadata['rmnet'], seeds, cohort)
    if any(row['config'].get('rmnet_class_weighting') != 'uniform' for row in metadata['rmnet']):
        raise ValueError('The order comparison uses the uniform-weight RM-Net refits.')

    clips, split, views, source = source_for(metadata['compact'][0]['config'], device)
    held = split['eval']
    visible = views[1:3, 1, held]
    order = permutations(visible)
    torch_order = torch.as_tensor(order, device=device)
    result = dict(ids=clips.ids[held], groups=clips.groups[held], y=clips.label[held],
                  sentiment=clips.sentiment[held], masks=views[:, :, held], permutations=order.astype(np.uint8),
                  cohort=np.asarray(cohort), seeds=np.asarray(seeds))
    for name, runs_for_head in paths.items():
        archived = load_matched(runs_for_head, result)
        scores, posterior = [], []
        uniform = name == 'rmnet'
        for run, row, reference in zip(runs_for_head, metadata[name], archived, strict=True):
            cfg = row['config']
            repo = upstream if upstream is not None else Path(cfg.get('upstream', '.'))
            forecast = predictor('rmnet_core' if uniform else candidate, run, row, repo, source)
            probabilities, intensity = infer(forecast, source, held, torch_order, cfg['batch'])
            p_key, s_key = ('posterior', 'score') if uniform else ('intensity', 'intensity_score')
            np.testing.assert_allclose(probabilities[0], reference[p_key][1:3], rtol=3e-5, atol=5e-6,
                                       err_msg=run.name)
            np.testing.assert_allclose(intensity[0], reference[s_key][1:3], rtol=3e-5, atol=5e-6,
                                       err_msg=run.name)
            if name == 'compact' and cohort == 'omg':
                np.testing.assert_allclose(probabilities, np.broadcast_to(probabilities[0], probabilities.shape),
                                           rtol=2e-5, atol=2e-6, err_msg='pointwise permutation invariance')
                np.testing.assert_allclose(intensity, np.broadcast_to(intensity[0], intensity.shape),
                                           rtol=2e-5, atol=2e-6, err_msg='pointwise score invariance')
            posterior.append(probabilities)
            scores.append(intensity)
            print(f'{cohort} {run.name}: 6 orders over {len(held)} clips per gap', flush=True)
        result[name], result[name + '_score'] = np.stack(posterior), np.stack(scores)
    return result


def summarize(evidence, resamples):
    y, group, sentiment = (evidence[key] for key in ('y', 'groups', 'sentiment'))
    _, membership = np.unique(group, return_inverse=True)
    risks, methods = {}, {}
    for name in MODELS:
        posterior = evidence[name].astype(np.float64)
        np.testing.assert_allclose(posterior.sum(-1), 1., rtol=0, atol=2e-7)
        rps = ranked_probability_score(posterior, y)
        correct = posterior.argmax(-1) == y
        errors = np.abs(evidence[name + '_score'] - sentiment)
        square = np.square(coordinates(posterior) - outcomes(y)) / 4.
        risks[name], methods[name] = {}, {}
        for condition, indices in CONDITIONS.items():
            take = list(indices)
            losses = rps[:, take]
            signed, neutral = square[:, take].mean(axis=(0, 1, 2, 3))
            np.testing.assert_allclose(signed + neutral, losses.mean(), atol=5e-8)
            risks[name][condition] = losses.mean(axis=(1, 2))
            methods[name][condition] = dict(rps=float(losses.mean()), signed_rps=float(signed),
                                             neutral_rps=float(neutral), accuracy=float(correct[:, take].mean()),
                                             mae=float(errors[:, take].mean()))
    penalty = {name: risks[name]['random'] - risks[name]['chronological'] for name in MODELS}
    interaction = penalty['rmnet'] - penalty['compact']
    altered_gap = {condition: risks['compact'][condition] - risks['rmnet'][condition]
                   for condition in CONDITIONS}
    np.testing.assert_allclose(altered_gap['chronological'] - altered_gap['random'], interaction, atol=1e-13)
    other_orders = {condition: altered_gap['chronological'] - altered_gap[condition]
                    for condition in ('reverse', 'rotate')}
    visible_per_gap = evidence['masks'][1:3, 1].sum(-1)
    return dict(cohort=evidence['cohort'].item(), n_clips=len(y), n_videos=len(np.unique(group)),
                seeds=evidence['seeds'].tolist(), resamples=resamples, random_draws=3,
                mean_visible_bins=visible_per_gap.mean(axis=1).tolist(), methods=methods,
                random_penalties={name: paired_reduction(values, membership, resamples, 6)
                                  for name, values in dict(penalty, interaction=interaction).items()},
                other_interactions={name: paired_reduction(values, membership, resamples, 4)
                                    for name, values in other_orders.items()},
                matched_gaps={name: paired_reduction(altered_gap[name], membership, resamples, 4)
                              for name in ('chronological', 'random')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--runs', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--cohort', choices=RUN_NAME)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--first-seed', action='store_true')
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.runs and not args.cohort) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples and provide --cohort with --runs.')
    if args.runs:
        seeds = SEEDS[:1] if args.first_seed else SEEDS
        evidence = collect(args.runs, args.cohort, seeds, torch.device(args.device), args.upstream)
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    result = summarize(evidence, args.resamples)
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **evidence)
        temporary.replace(args.write_evidence)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, effect in result['random_penalties'].items():
        lower, upper = effect['ci95_familywise']
        print(f'{name}: {effect["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
