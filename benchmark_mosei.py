"""Extend the matched CMU-MOSEI benchmark with published baselines."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark import run
from data import load_clips


ROOT = Path(__file__).resolve().parent
SEEDS = (97, 17, 521)
SETTINGS = {
    'api': ('regression', .0001, 0.),
    'ucmr-net': ('regression', .001, .0001),
    'tfr-net': ('joint', .001, .01),
    'rm-net': ('joint', .0001, .001),
}


def verify_development_split(reference, clips, split, seed):
    settings = json.loads((reference / 'run.json').read_text(encoding='utf-8'))
    config = settings['config']
    expected = dict(seed=seed, split='official', mask_seed=1777, split_seed=1729, bins=24, inputs='visual')
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'The reference protocol differs at {reference}.')
    for name in ('fit', 'tune', 'eval'):
        ids = split[name]
        actual = dict(n=len(ids), videos=len(np.unique(clips.groups[ids])),
                      labels=np.bincount(clips.label[ids], minlength=3).tolist())
        if actual != settings['splits'][name]:
            raise ValueError(f'The {name} split differs at {reference}.')
    with np.load(reference / 'tuning.npz', allow_pickle=False) as archived:
        np.testing.assert_array_equal(clips.label[split['tune']], archived['y'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=SETTINGS, required=True)
    parser.add_argument('--upstream', nargs='+', type=Path, required=True,
                        help='One pinned source checkout for each requested model, in the same order.')
    parser.add_argument('--seeds', nargs='+', choices=SEEDS, type=int, default=SEEDS)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/mosei_aligned_50.pkl')
    parser.add_argument('--reference-runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--eval-groups', type=Path, default=ROOT / 'splits/mosei_eval_videos.txt')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/mosei_matched')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--patience', type=int, default=6)
    options = parser.parse_args()
    if len(options.models) != len(options.upstream) or len(set(options.models)) != len(options.models):
        parser.error('Supply a distinct model and upstream checkout for each baseline.')
    if len(set(options.seeds)) != len(options.seeds):
        parser.error('Each seed must appear only once.')
    if min(options.epochs, options.patience) < 1 or options.batch < 2:
        parser.error('Use positive epoch and patience limits, with a batch size of at least two.')
    pending = []
    for name, upstream in zip(options.models, options.upstream, strict=True):
        head, lr, weight_decay = SETTINGS[name]
        for seed in options.seeds:
            output = options.output.resolve() / f'{name.replace("-", "_")}_seed{seed}'
            args = SimpleNamespace(
                model=name, upstream=upstream.resolve(), head=head, corpus='mosei', data=options.data.resolve(),
                output=output, match_run=(options.reference_runs / f'mosei_fresh_visual_seed{seed}').resolve(),
                split='official', group_reference=None, eval_groups=options.eval_groups.resolve(), neutral_margin=0.,
                inputs='visual', seed=seed, epochs=options.epochs, patience=options.patience, batch=options.batch,
                lr=lr, weight_decay=weight_decay, rmnet_class_weighting='inverse-frequency', device=options.device,
                threads=4, resume=output.is_dir(),
            )
            metadata = output / 'run.json'
            if metadata.is_file():
                config = json.loads(metadata.read_text(encoding='utf-8'))['config']
                fields = ('model', 'head', 'corpus', 'seed', 'epochs', 'patience', 'batch', 'lr', 'weight_decay',
                          'device')
                if any(config.get(key) != getattr(args, key) for key in fields):
                    parser.error(f'{output} contains a completed fit with different settings. Use --output.')
                if not all((output / filename).is_file() for filename in ('predictions.npz', 'tuning.npz', 'model.pt')):
                    parser.error(f'{output} contains a partial copy of a completed fit. Use --output.')
                print(f'{name} seed {seed}: completed results at {output}', flush=True)
            else:
                pending.append(args)
    if not pending:
        return

    dataset = load_clips(options.data, 24, 'official', 1729, eval_groups=options.eval_groups)
    for seed in options.seeds:
        verify_development_split(options.reference_runs / f'mosei_fresh_visual_seed{seed}', *dataset, seed)
    for args in pending:
        print(f'{args.model} seed {args.seed}', flush=True)
        run(args, dataset)
        if torch.device(options.device).type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
