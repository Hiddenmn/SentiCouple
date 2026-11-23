"""Compare visual dispersion, fusion width, and shared intensity under identical gaps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_seed_effects import count_anchor, run_paths, verify_training
from score_anchor_effects import SEEDS
from text_weighted_loss import REFERENCES, RUN_NAME, anchor_for, load_references, source_for
from train import calibrated_head_forecasts, fit_fusion


WIDTHS = {'original': 1, 'wide': 2}
POOLING = {'max': 'mean_max', 'std': 'mean_std'}
HEADS = {name: REFERENCES[name] for name in ('shared', 'independent')}
CELLS = {(width, pool, head): f'{width}_{pool}_{head}'
         for width in WIDTHS for pool in POOLING for head in HEADS}
BASELINES = ('tfr_joint', 'lnln_joint', 'rmnet_uniform')


def collect(runs, cohort, seeds, device):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    candidate = RUN_NAME[cohort]
    paths = run_paths(runs, seeds, cohort)
    original_runs = paths[candidate]
    metadata = load_metadata(original_runs)
    verify_training(candidate, metadata, seeds, cohort)
    settings = metadata[0]['config']
    archived = load_references(runs, cohort, seeds, original_runs)
    evidence = {key: archived[key] for key in ('ids', 'groups', 'y', 'sentiment', 'masks')}
    for head in HEADS:
        key = CELLS['original', 'max', head]
        evidence[key] = archived[head]
        evidence[key + '_score'] = archived[head + '_score']
    for name in BASELINES[:2]:
        verify_training(name, load_metadata(paths[name]), seeds, cohort)
        for suffix in ('', '_score'):
            evidence[name + suffix] = archived[name + suffix]
    rm_runs = [runs / f'{cohort}_rmnet_uniform_seed{seed}' for seed in seeds]
    rm_metadata = load_metadata(rm_runs)
    verify_training('rmnet_core', rm_metadata, seeds, cohort)
    if any(row['config'].get('rmnet_class_weighting') != 'uniform' for row in rm_metadata):
        raise ValueError('The RM-Net reference must use uniform class weights.')
    rm = load_matched(rm_runs, evidence)
    evidence['rmnet_uniform'] = np.stack([packet['posterior'][1:3] for packet in rm])
    evidence['rmnet_uniform_score'] = np.stack([packet['score'][1:3] for packet in rm])

    clips, split, views, source = source_for(settings, device)
    held = split['eval']
    for key, values in (('ids', clips.ids[held]), ('groups', clips.groups[held]),
                        ('y', clips.label[held]), ('sentiment', clips.sentiment[held]),
                        ('masks', views[:, :, held])):
        np.testing.assert_array_equal(evidence[key], values, err_msg=key)
    banks = np.full((len(views), len(clips.ids), 1), -1, dtype=np.int32)
    forecasts = {cell: [] for cell in CELLS.values() if cell not in evidence}
    selections = []
    for row, (seed, run, record) in enumerate(zip(seeds, original_runs, metadata, strict=True)):
        config = record['config']
        if any(config[key] != value for key, value in settings.items() if key not in ('seed', 'output')):
            raise ValueError(f'{run.name} has different training settings.')
        if config['score_anchor'] != 'class-mean' or config['selection'] != 'rps':
            raise ValueError('Use the original heads selected by tuning RPS with class-mean score anchors.')
        args = argparse.Namespace(**config)
        saved = anchor_for(run, source, split, config, archived, row)
        selected = {}
        for (width, pool, head), cell in CELLS.items():
            arm = HEADS[head]
            if cell not in forecasts:
                selected[cell] = record['selection'][arm]
                continue
            model, choice = fit_fusion(args, arm, source, split, saved, clips.label, banks,
                                        fusion_width=args.width * WIDTHS[width],
                                        visual_pooling=POOLING[pool])
            posterior, score, gain = calibrated_head_forecasts(model, source, clips, split, saved, args.batch)
            forecasts[cell].append((posterior, score))
            selected[cell] = dict(choice, score_gain=gain)
            print(f'{cohort} seed {seed}: {cell}', flush=True)
        for width in WIDTHS:
            for head in HEADS:
                maximum = selected[CELLS[width, 'max', head]]['parameters']
                dispersion = selected[CELLS[width, 'std', head]]['parameters']
                if maximum != dispersion:
                    raise ValueError('The pooling comparison must match parameter counts.')
        selections.append(selected)

    for name, pairs in forecasts.items():
        posterior, scores = zip(*pairs, strict=True)
        evidence[name], evidence[name + '_score'] = np.stack(posterior), np.stack(scores)
    anchor_parameters = [count_anchor(path) for path in original_runs]
    if len(set(anchor_parameters)) != 1:
        raise ValueError('All paired seeds must have the same anchor architecture.')
    parameters = {name: [row[name]['parameters'] + anchor_parameters[i] for i, row in enumerate(selections)]
                  for name in CELLS.values()}
    for name, records in [('rmnet_uniform', rm_metadata),
                          *((baseline, load_metadata(paths[baseline])) for baseline in BASELINES[:2])]:
        parameters[name] = [row['trainable_parameters'] for row in records]
    return dict(evidence, cohort=np.asarray(cohort), seeds=np.asarray(seeds),
                original_width=np.asarray(settings['width']), parameters=np.asarray(json.dumps(parameters)),
                selections=np.asarray(json.dumps(selections)))


def summarize(evidence, resamples):
    label, groups = evidence['y'], evidence['groups']
    n_videos = len(np.unique(groups))
    _, membership = np.unique(groups, return_inverse=True)
    selections = json.loads(evidence['selections'].item())
    parameters = json.loads(evidence['parameters'].item())
    tune_rps = {name: float(np.mean([-row[name]['tune_score'] for row in selections]))
                for name in CELLS.values()}
    shared = [name for (width, pool, head), name in CELLS.items() if head == 'shared']
    chosen = min(shared, key=tune_rps.__getitem__)

    methods, risks = {}, {}
    target = outcomes(label)
    for name in (*CELLS.values(), *BASELINES):
        posterior = np.ascontiguousarray(evidence[name], dtype=np.float64)
        np.testing.assert_allclose(posterior.sum(-1), 1., rtol=0, atol=2e-7)
        risks[name] = ranked_probability_score(posterior, label)
        signed, neutral = (np.square(coordinates(posterior) - target) / 4.).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(signed + neutral, risks[name].mean(), rtol=0, atol=5e-8)
        error = np.abs(evidence[name + '_score'] - evidence['sentiment'])
        methods[name] = dict(**rps_summary(posterior, label), signed_rps=float(signed),
                             neutral_rps=float(neutral), mae=float(error.mean()),
                             parameters=parameters[name][0])
        if len(set(parameters[name])) != 1:
            raise ValueError(f'{name} parameter count differs across seeds.')

    def contrast(difference, family):
        return paired_reduction(difference.mean(1), membership, resamples, family)

    def risk(width, pool, head='shared'):
        return risks[CELLS[width, pool, head]]

    pooling = {width: risk(width, 'max') - risk(width, 'std') for width in WIDTHS}
    selected_refs = ('original_max_shared', *BASELINES)
    return dict(cohort=evidence['cohort'].item(), original_width=evidence['original_width'].item(),
                n_clips=len(label), n_videos=n_videos, seeds=evidence['seeds'].tolist(), resamples=resamples,
                tune_rps=tune_rps, tuning_selected=chosen, methods=methods,
                pooling={width: contrast(delta, 4) for width, delta in pooling.items()},
                pooling_by_gap={width: delta.mean(axis=(0, 2)).tolist() for width, delta in pooling.items()},
                widening={pool: contrast(risk('original', pool) - risk('wide', pool), 4) for pool in POOLING},
                width_pooling_interaction=contrast(pooling['wide'] - pooling['original'], 2),
                sharing={f'{width}_{pool}': contrast(risk(width, pool, 'independent') - risk(width, pool), 8)
                         for width in WIDTHS for pool in POOLING},
                selected_vs_reference={name: contrast(risks[name] - risks[chosen], 8) for name in selected_refs},
                selections=selections)


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
        seeds = SEEDS[:1] if args.first_seed else SEEDS
        evidence = collect(args.runs, args.cohort, seeds, torch.device(args.device))
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
    print(f'Tuning selected: {result["tuning_selected"]}', flush=True)
    for family in ('pooling', 'widening', 'sharing', 'selected_vs_reference'):
        for name, contrast in result[family].items():
            lower, upper = contrast['ci95_familywise']
            print(f'{family} / {name}: {contrast["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
