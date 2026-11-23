"""Compare matched sentiment heads and fusion cores under equal-scale visual crops."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_seed_effects import count_anchor, run_paths, verify_training
from score_anchor_effects import HEADS, SEEDS
from train import run as fit_compact


REPRESENTATIONS = ('whole', 'face', 'center', 'face_scene', 'center_scene')
CORES = {'lnln': 'lnln_joint', 'rmnet': 'rmnet_core'}
OBSERVATIONS = ('ids', 'groups', 'y', 'sentiment', 'masks')


def fit_core(config, output):
    if (output / 'run.json').is_file():
        return
    command = [sys.executable, '-B', str(Path(__file__).with_name('benchmark.py'))]
    for key, value in config.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend((option, str(value)))
    if (output / 'resume.pt').is_file():
        command.append('--resume')
    subprocess.run(command, check=True)


def unchanged_settings(original, variant, allowed):
    before, after = original['config'], variant['config']
    for key in (before.keys() | after.keys()) - set(allowed):
        if before.get(key) != after.get(key):
            raise ValueError(f'Visual representation trial changed {key}.')


def check_anchors(original, changed, original_packets, changed_packets):
    for first, second, old, new in zip(original, changed, original_packets, changed_packets, strict=True):
        np.testing.assert_allclose(old['anchor'], new['anchor'], rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(old['anchor_score'], new['anchor_score'], rtol=2e-5, atol=4e-6)
        with (np.load(first / 'tuning.npz', allow_pickle=False) as left,
              np.load(second / 'tuning.npz', allow_pickle=False) as right):
            np.testing.assert_array_equal(left['y'], right['y'])
            np.testing.assert_allclose(left['anchor'], right['anchor'], rtol=2e-5, atol=3e-6)


def collect(runs, features, work, seeds):
    work.mkdir(parents=True, exist_ok=True)
    paths = run_paths(runs, seeds, 'omg')
    originals = paths['pointwise']
    old_metadata = load_metadata(originals)
    verify_training('pointwise', old_metadata, seeds, 'omg')
    reference = read_predictions(originals[0])
    old_packets = load_matched(originals, reference)
    evidence = {key: reference[key] for key in OBSERVATIONS}
    count = count_anchor(originals[0])
    coverage = json.loads((features / 'crop_summary.json').read_text(encoding='utf-8'))
    with np.load(features / 'crop_coverage.npz', allow_pickle=False) as packet:
        indices = {clip: i for i, clip in enumerate(packet['ids'])}
        used = [indices[clip] for clip in reference['ids']]
        np.testing.assert_array_equal(packet['native'][used], reference['masks'][0, 1])
        evidence['localized'] = packet['localized'][used]
        evidence['crop_overlap'] = packet['overlap'][used]
    roster = dict(representations=REPRESENTATIONS, crop_metadata=coverage, seeds=list(seeds),
                  parameters={}, selections={})

    for part in REPRESENTATIONS:
        name = features / f'omg_{part}.pkl'
        folders = originals if part == 'whole' else [work / part / run.name for run in originals]
        if part != 'whole':
            for seed, output, base in zip(seeds, folders, old_metadata, strict=True):
                print(f'{part}: compact seed {seed}.', flush=True)
                config = dict(base['config'], output=output, data=name, arms=list(HEADS.values()))
                config.setdefault('teacher', None)
                if not (output / 'run.json').is_file():
                    fit_compact(argparse.Namespace(**config))
        metadata = load_metadata(folders)
        verify_training('pointwise', metadata, seeds, 'omg')
        packets = load_matched(folders, reference)
        if part != 'whole':
            for old, new in zip(old_metadata, metadata, strict=True):
                unchanged_settings(old, new, ('data', 'output', 'arms'))
                if old['splits'] != new['splits'] or Path(new['config']['data']).resolve() != name.resolve():
                    raise ValueError('Expected the original source-video split with the specified crops.')
            check_anchors(originals, folders, old_packets, packets)
        for head, arm in HEADS.items():
            method = f'{part}_{head}'
            evidence[method] = np.stack([packet[arm][1:3] for packet in packets])
            evidence[method + '_score'] = np.stack([packet[arm + '_score'][1:3] for packet in packets])
            selections = [row['selection'][arm] for row in metadata]
            if {row['parameters'] for row in selections} != {old_metadata[0]['selection'][arm]['parameters']}:
                raise ValueError('Crop representations must retain the original parameter count.')
            roster['parameters'][method] = selections[0]['parameters'] + count
            roster['selections'][method] = selections

    for core, runner_name in CORES.items():
        base_folders = (paths[runner_name] if core == 'lnln' else
                        [runs / f'omg_rmnet_uniform_seed{seed}' for seed in seeds])
        full = load_metadata(base_folders)
        verify_training(runner_name, full, seeds, 'omg')
        folders = [work / 'published' / folder.name for folder in base_folders]
        for seed, folder, old, compact in zip(seeds, folders, full, originals, strict=True):
            print(f'{core}: localized representation seed {seed}.', flush=True)
            cfg = dict(old['config'], output=folder, data=features / 'omg_face_scene.pkl',
                       match_run=work / 'face_scene' / compact.name)
            if core == 'rmnet' and cfg['rmnet_class_weighting'] != 'uniform':
                raise ValueError('Use the uniform-weight RM-Net baseline.')
            fit_core(cfg, folder)
        face_metadata = load_metadata(folders)
        verify_training(runner_name, face_metadata, seeds, 'omg')
        for old, new in zip(full, face_metadata, strict=True):
            unchanged_settings(old, new, ('data', 'output', 'match_run'))
            if Path(new['config']['data']).resolve() != (features / 'omg_face_scene.pkl').resolve():
                raise ValueError('The matched baseline must use the face-plus-whole feature packet.')
        for part, selected, meta in (('whole', base_folders, full), ('face_scene', folders, face_metadata)):
            method = f'{part}_{core}'
            packets = load_matched(selected, reference)
            evidence[method] = np.stack([packet['posterior'][1:3] for packet in packets])
            evidence[method + '_score'] = np.stack([packet['score'][1:3] for packet in packets])
            params = {row['trainable_parameters'] for row in meta + full}
            if len(params) != 1:
                raise ValueError('The published core must retain its architecture.')
            roster['parameters'][method] = params.pop()
            roster['selections'][method] = [dict(row['selection'], seconds=row.get('training_seconds')) for row in meta]
    return dict(evidence, roster=np.asarray(json.dumps(roster, allow_nan=False)))


def summarize(evidence, resamples):
    roster = json.loads(evidence['roster'].item())
    label, sentiment = evidence['y'], evidence['sentiment']
    videos, membership = np.unique(evidence['groups'], return_inverse=True)
    methods, risk = {}, {}
    for method, parameters in roster['parameters'].items():
        p = evidence[method].astype(np.float64)
        np.testing.assert_allclose(p.sum(-1), 1., rtol=0, atol=2e-7)
        risk[method] = ranked_probability_score(p, label)
        signed, neutral = (np.square(coordinates(p) - outcomes(label)) / 4).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(signed + neutral, risk[method].mean(), rtol=0, atol=5e-8)
        methods[method] = dict(**rps_summary(p, label), signed_rps=float(signed), neutral_rps=float(neutral),
                               mae=float(np.abs(evidence[method + '_score'] - sentiment).mean()),
                               parameters=parameters, rps_by_gap=risk[method].mean(axis=(0, 2)).tolist())

    def compare(difference, family_size):
        return paired_reduction(difference.mean(1), membership, resamples, family_size)

    def shared(name):
        return risk[name + '_shared']

    benefit = {name: risk[name + '_independent'] - shared(name) for name in REPRESENTATIONS}
    sharing = {name: compare(values, 6) for name, values in benefit.items()}
    sharing['face_scene_change'] = compare(benefit['face_scene'] - benefit['whole'], 6)
    return dict(cohort='omg', n_clips=len(label), n_videos=len(videos), seeds=roster['seeds'],
                resamples=resamples, crop_metadata=roster['crop_metadata'],
                mean_cropped_bins=evidence['localized'].mean(),
                median_crop_iou=float(np.nanmedian(evidence['crop_overlap'])),
                primary={name: compare(shared(name) - shared('face_scene'), 2)
                         for name in ('whole', 'center_scene')},
                localized_only={name: compare(shared(name) - shared('face'), 2)
                                for name in ('whole', 'center')},
                sharing=sharing,
                benchmark_representation={core: compare(risk['whole_' + core] - risk['face_scene_' + core], 2)
                                          for core in CORES},
                benchmark_matched={core: compare(risk['face_scene_' + core] - shared('face_scene'), 2)
                                   for core in CORES},
                methods=methods, selections=roster['selections'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--features', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--first-seed', action='store_true')
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.features and not (args.runs and args.work)) or (args.evidence and args.write_evidence):
        parser.error('Use positive resamples, and supply both --runs and --work when fitting.')
    if args.features:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        data = collect(args.runs, args.features, args.work, SEEDS[:1] if args.first_seed else SEEDS)
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            data = dict(packet)
    report = summarize(data, args.resamples)
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(args.write_evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for family in ('primary', 'localized_only', 'sharing', 'benchmark_representation', 'benchmark_matched'):
        for name, row in report[family].items():
            lower, upper = row['ci95_familywise']
            print(f'{family} / {name} {row["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
