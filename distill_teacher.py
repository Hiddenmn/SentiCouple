"""Replay the paired RM-Net models and export logits on fitting clips only."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ablate import read_predictions
from benchmark import predict
from compare_benchmarks import load_metadata
from frozen_lnln_heads import prepare as load_cohort
from published_baselines import build_published
from published_seed_effects import run_paths, verify_training
from score_anchor_effects import SEEDS


def export(args):
    clips, split, views, source = load_cohort(args, args.cohort)
    fit, heldout = split['fit'], split['eval']
    match = dict(ids=clips.ids[heldout], groups=clips.groups[heldout], y=clips.label[heldout],
                 sentiment=clips.sentiment[heldout], masks=views[:, :, heldout])
    runs = run_paths(args.runs, args.seeds, args.cohort)['rmnet_core']
    metadata = load_metadata(runs)
    verify_training('rmnet_core', metadata, args.seeds, args.cohort)
    dimensions = (source.text.shape[-1], 1, source.vision.shape[-1])
    lengths = (source.text.shape[1], source.audio.shape[1], source.vision.shape[1])
    corpus = 'omg' if args.cohort == 'omg' else 'sims'
    args.output.mkdir(parents=True, exist_ok=True)

    for seed, run, meta in zip(args.seeds, runs, metadata, strict=True):
        output = args.output / f'{args.cohort}_teacher_seed{seed}.npz'
        if output.exists():
            raise FileExistsError(output)
        start = time.perf_counter()
        archived = read_predictions(run)
        for key, expected in match.items():
            np.testing.assert_array_equal(archived[key], expected, err_msg=key)
        model = build_published('rm-net', args.upstream, corpus, dimensions, lengths, 'joint', True)
        model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
        model = model.to(source.device)
        replay = predict(model, source, heldout, meta['config']['batch'], range(len(views)))
        np.testing.assert_allclose(replay, archived['raw'], atol=1e-5, rtol=3e-5)
        logits = predict(model, source, fit, meta['config']['batch'], range(len(views)))[..., :3]
        with output.with_suffix('.npz.part').open('wb') as stream:
            np.savez_compressed(stream, seed=seed, ids=clips.ids[fit], masks=views[:, :, fit], logits=logits,
                                revision=meta['upstream_revision'], teacher_run=run.name)
        output.with_suffix('.npz.part').replace(output)
        print(f'{output.name}: {len(fit)} fitting clips, four views, {time.perf_counter()-start:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, choices=['sims_v2', 'omg'])
    parser.add_argument('--sims-data', type=Path)
    parser.add_argument('--group-reference', type=Path)
    parser.add_argument('--omg-data', type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.cohort == 'sims_v2' and not (args.sims_data and args.group_reference):
        parser.error('CH-SIMS v2 requires its feature packet and original video groups.')
    if args.cohort == 'omg' and not args.omg_data:
        parser.error('OMG-Emotion requires its feature packet.')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    export(args)


if __name__ == '__main__':
    main()
