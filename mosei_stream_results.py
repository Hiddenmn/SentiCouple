"""Paired head and audio-availability comparisons on held-out CMU-MOSEI videos."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions
from benchmark_mosei import ROOT, SEEDS
from compare_benchmarks import load_matched, load_metadata, summarize
from mosei_results import paired_summary


def evaluate(runs, resamples):
    three_runs = [runs / f'mosei_three_stream_seed{seed}' for seed in SEEDS]
    visual_runs = [runs / f'mosei_fresh_visual_seed{seed}' for seed in SEEDS]
    reference = read_predictions(three_runs[0])
    three = load_matched(three_runs, reference)
    visual = [read_predictions(run) for run in visual_runs]
    labels, intensity, groups = (reference[key] for key in ('y', 'sentiment', 'groups'))
    three_configs = load_metadata(three_runs)
    visual_configs = load_metadata(visual_runs)
    for seed, packet, primary, control in zip(SEEDS, visual, three_configs, visual_configs, strict=True):
        for key in ('ids', 'groups', 'y', 'sentiment'):
            np.testing.assert_array_equal(packet[key], reference[key], err_msg=key)
        np.testing.assert_array_equal(packet['masks'][:, 1], reference['masks'][:, 1], err_msg='Visual masks')
        fields = ('split', 'split_seed', 'mask_seed', 'bins', 'backbone', 'selection', 'width', 'batch',
                  'anchor_epochs', 'fusion_epochs', 'lr', 'pairing_weight', 'intensity_weight')
        a, b = primary['config'], control['config']
        if a['seed'] != seed or b['seed'] != seed or any(a[f] != b[f] for f in fields):
            raise ValueError(f'Training settings differ at seed {seed}.')
        if primary['splits'] != control['splits'] or (a['inputs'], b['inputs']) != ('audiovisual', 'visual'):
            raise ValueError(f'Expected the original splits with three or two streams at seed {seed}.')
    if not reference['masks'][1:3, 0].any():
        raise ValueError('The three-stream run must receive observed audio.')

    methods, losses, errors = {}, {}, {}
    for name, forecasts, arm in (
        ('three_shared', three, 'intensity'),
        ('three_independent', three, 'independent_intensity'),
        ('three_class_only', three, 'fusion'),
        ('text_vision_shared', visual, 'intensity'),
    ):
        methods[name], losses[name], errors[name] = summarize(forecasts, arm, f'{arm}_score', labels, intensity)
    comparisons = ('three_independent', 'three_class_only', 'text_vision_shared')
    _, membership = np.unique(groups, return_inverse=True)
    rps = {key: losses[key] - losses['three_shared'] for key in comparisons}
    mae = {key: errors[key] - errors['three_shared'] for key in comparisons}
    return dict(clips=len(labels), videos=int(len(np.unique(groups))), seeds=SEEDS, resamples=resamples,
                methods=methods, rps_gain=paired_summary(rps, membership, resamples, len(comparisons)),
                mae_gain=paired_summary(mae, membership, resamples, len(comparisons)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    result = evaluate(args.runs, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
