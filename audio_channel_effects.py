"""Refit and compare matched OMG-Emotion models with and without acoustic features."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from benchmark import model_inputs
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from data import load_clips, observation_views
from face_crop_effects import fit_core, unchanged_settings
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from published_inference import predictor
from published_seed_effects import count_anchor, run_paths, verify_training
from score_anchor_effects import HEADS, SEEDS
from train import BatchSource, index_batches, run as train_compact


ARMS = dict(HEADS, class_only='fusion')
CORES = {'lnln': ('lnln_joint', 'omg_lnln_joint'),
         'rmnet': ('rmnet_core', 'omg_rmnet_uniform')}
LABEL_FIELDS = ('ids', 'groups', 'y', 'sentiment')


def compact_paths(runs, work):
    previous = run_paths(runs, SEEDS, 'omg')['pointwise']
    return previous, [work / 'compact' / path.name for path in previous]


def fit_compact(runs, work, data):
    previous, current = compact_paths(runs, work)
    originals = load_metadata(previous)
    verify_training('pointwise', originals, SEEDS, 'omg')
    for seed, old, output in zip(SEEDS, originals, current, strict=True):
        if (output / 'run.json').is_file():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f'Acoustic input, compact seed {seed}.', flush=True)
        with tempfile.TemporaryDirectory(prefix='.train_', dir=output.parent) as tmp:
            staged = Path(tmp).resolve() / output.name
            if not staged.is_relative_to(output.parent.resolve()):
                raise ValueError('Keep compact checkpoints inside their experiment directory.')
            cfg = dict(old['config'], data=data, output=staged, inputs='audiovisual',
                       preserve_vision_gaps=True, arms=list(ARMS.values()))
            cfg.setdefault('teacher', None)
            train_compact(argparse.Namespace(**cfg))
            metadata = staged / 'run.json'
            result = json.loads(metadata.read_text(encoding='utf-8'))
            result['config']['output'] = str(output)
            metadata.write_text(json.dumps(result, indent=2), encoding='utf-8')
            staged.rename(output)


def fit_published(runs, work, data, model):
    name, stem = CORES[model]
    _, compact = compact_paths(runs, work)
    previous = [runs / f'{stem}_seed{seed}' for seed in SEEDS]
    originals = load_metadata(previous)
    verify_training(name, originals, SEEDS, 'omg')
    for seed, old, base, match in zip(SEEDS, originals, previous, compact, strict=True):
        target = work / 'published' / base.name
        if model == 'rmnet' and old['config'].get('rmnet_class_weighting') != 'uniform':
            raise ValueError('The acoustic comparison uses uniform classification weights in RM-Net.')
        print(f'Acoustic input, {model} seed {seed}.', flush=True)
        cfg = dict(old['config'], data=data, output=target, inputs='audiovisual',
                   preserve_vision_gaps=True, match_run=match)
        fit_core(cfg, target)


def verify_settings(name, before, after, data, allowed):
    verify_training(name, before, SEEDS, 'omg')
    verify_training(name, after, SEEDS, 'omg', inputs='audiovisual')
    for old, new in zip(before, after, strict=True):
        unchanged_settings(old, new, ('data', 'output', 'inputs', 'preserve_vision_gaps', *allowed))
        config = new['config']
        if (Path(config['data']).resolve() != data.resolve()
                or config.get('preserve_vision_gaps') is not True):
            raise ValueError('A forecast used a different audio packet or visual mask rule.')


def within_video_donors(ids, groups):
    donors = np.full(len(ids), -1, dtype=np.int64)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        if len(members) > 1:
            ordered = members[np.argsort(ids[members], kind='stable')]
            donors[ordered] = np.roll(ordered, len(ordered) // 2)
    return donors


@torch.inference_mode()
def predict_audio_swap(model, source, held, recipients, donors, batch):
    original, swapped, values = [], [], []
    for gap in (1, 2):
        original_gap, swapped_gap, score_gap = [], [], []
        for view, positions in index_batches(recipients, batch, source.device, gap):
            rows, matches = held[positions], held[donors[positions]]
            observed, masks = model_inputs(source, view, rows)
            probability, score = model((observed, masks))
            audio = source.audio[matches] * masks[1][..., None]
            replacement, _ = model(((observed[0], audio, observed[2]), masks))
            original_gap.append(probability.cpu().numpy())
            swapped_gap.append(replacement.cpu().numpy())
            score_gap.append(score.cpu().numpy())
        original.append(np.concatenate(original_gap))
        swapped.append(np.concatenate(swapped_gap))
        values.append(np.concatenate(score_gap))
    return np.stack(original), np.stack(swapped), np.stack(values)


def swap_audio(evidence, paths, data):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    clips, split = load_clips(data, 24, 'official', 1729, neutral_margin=.10)
    views = observation_views(clips, [.35, .65], 1777, preserve_vision_gaps=True)
    held = split['eval']
    np.testing.assert_array_equal(clips.ids[held], evidence['ids'])
    np.testing.assert_array_equal(views[:, :, held], evidence['masks'])
    donors = within_video_donors(clips.ids[held], clips.groups[held])
    eligible = np.flatnonzero(donors >= 0)
    np.testing.assert_array_equal(clips.groups[held[eligible]], clips.groups[held[donors[eligible]]])
    if (donors[eligible] == eligible).any() or not len(eligible):
        raise ValueError('Expected another clip from the same video for every audio replacement.')
    if (views[1:3, 0, held[eligible]] & ~clips.audio_mask[held[donors[eligible]]]).any():
        raise ValueError('Replacement audio must cover the original observed bins.')

    source = BatchSource(clips, views, split['fit'], device, stream_text=True)
    evidence['donor_rows'] = donors
    held = torch.as_tensor(held, device=device)
    recipients = torch.as_tensor(eligible, device=device)
    donor_index = torch.as_tensor(donors, device=device)
    for method, folders in paths.items():
        name = CORES[method][0] if method in CORES else 'pointwise'
        predictions = []
        for seed, folder, row in zip(SEEDS, folders, load_metadata(folders), strict=True):
            upstream = row['config'].get('upstream')
            model = predictor(name, folder, row, Path(upstream) if upstream else None, source)
            replay, replacement, score = predict_audio_swap(model, source, held, recipients, donor_index,
                                                            row['config']['batch'])
            base = f'joint_{method}'
            np.testing.assert_allclose(replay, evidence[base][len(predictions)][:, eligible],
                                       rtol=2e-5, atol=1e-5, err_msg=f'{method} seed {seed}')
            np.testing.assert_allclose(score, evidence[base + '_score'][len(predictions)][:, eligible],
                                       rtol=2e-5, atol=1e-5, err_msg=f'{method} seed {seed}')
            predictions.append(replacement)
        evidence['acoustic_swap_' + method] = np.stack(predictions)
        print(f'Replayed and replaced the acoustic input for {method} in all ten fits.', flush=True)


def collect(runs, work, data):
    previous, current = compact_paths(runs, work)
    before, after = load_metadata(previous), load_metadata(current)
    verify_settings('pointwise', before, after, data, ('arms',))
    if any(x['splits'] != y['splits'] for x, y in zip(before, after, strict=True)):
        raise ValueError('Acoustic input must retain all source-video splits.')

    visual, joint = read_predictions(previous[0]), read_predictions(current[0])
    for key in LABEL_FIELDS:
        np.testing.assert_array_equal(visual[key], joint[key])
    np.testing.assert_array_equal(visual['masks'][:, 1], joint['masks'][:, 1])
    if visual['masks'][:, 0].any() or joint['masks'][-1, 0].any():
        raise ValueError('Audio must be absent from the visual model and both models on the anchor view.')

    evidence = dict(**{key: joint[key] for key in LABEL_FIELDS}, masks=joint['masks'])
    roster = dict(seeds=SEEDS, parameters={}, selection={},
                  audio_summary=json.loads(data.with_name('audio_summary.json').read_text('utf-8')))
    with np.load(data.with_name('audio_coverage.npz'), allow_pickle=False) as packet:
        indices = {clip: row for row, clip in enumerate(packet['ids'])}
        used = [indices[clip] for clip in joint['ids']]
        np.testing.assert_array_equal(packet['available'][used], joint['masks'][0, 0])
        evidence['audio_coverage'] = packet['coverage'][used]

    def save(method, packets, probability, score, parameters, selections):
        evidence[method] = np.stack([row[probability][1:3] for row in packets])
        evidence[method + '_score'] = np.stack([row[score][1:3] for row in packets])
        roster['parameters'][method] = parameters
        roster['selection'][method] = selections

    for view, paths, metadata, reference in (('visual', previous, before, visual), ('joint', current, after, joint)):
        packets = load_matched(paths, reference)
        anchor_parameters = count_anchor(paths[0])
        for arm, key in ARMS.items():
            selection = [run['selection'][key] for run in metadata]
            save(f'{view}_{arm}', packets, key, key + '_score',
                 selection[0]['parameters'] + anchor_parameters, selection)

    swap_paths = {'shared': current}
    for model, (name, stem) in CORES.items():
        previous = [runs / f'{stem}_seed{seed}' for seed in SEEDS]
        current = [work / 'published' / folder.name for folder in previous]
        before, after = load_metadata(previous), load_metadata(current)
        verify_settings(name, before, after, data, ('match_run',))
        for new, reference in zip(after, compact_paths(runs, work)[1], strict=True):
            if Path(new['config']['match_run']).resolve() != reference.resolve():
                raise ValueError('Published cores must match the compact audiovisual inputs.')
        for view, paths, metadata, reference in (('visual', previous, before, visual), ('joint', current, after, joint)):
            packets = load_matched(paths, reference)
            save(f'{view}_{model}', packets, 'posterior', 'score', metadata[0]['trainable_parameters'],
                 [run['selection'] for run in metadata])
        swap_paths[model] = current

    swap_audio(evidence, swap_paths, data)
    return dict(evidence, roster=np.asarray(json.dumps(roster, allow_nan=False)))


def summarize(evidence, resamples):
    roster = json.loads(evidence['roster'].item())
    label, sentiment = evidence['y'], evidence['sentiment']
    groups, membership = np.unique(evidence['groups'], return_inverse=True)
    methods, risk = {}, {}
    for method, parameters in roster['parameters'].items():
        probabilities = evidence[method].astype(np.float64)
        np.testing.assert_allclose(probabilities.sum(-1), 1, atol=2e-7, rtol=0)
        risk[method] = ranked_probability_score(probabilities, label)
        signed, neutral = (np.square(coordinates(probabilities) - outcomes(label)) / 4).mean(axis=(0, 1, 2))
        np.testing.assert_allclose(signed + neutral, risk[method].mean(), rtol=0, atol=5e-8)
        methods[method] = dict(**rps_summary(probabilities, label), signed_rps=float(signed),
                               neutral_rps=float(neutral), parameters=parameters,
                               mae=float(np.abs(evidence[method + '_score'] - sentiment).mean()),
                               rps_by_gap=risk[method].mean(axis=(0, 2)).tolist())

    def compare(difference, family_size):
        return paired_reduction(difference.mean(1), membership, resamples, family_size)

    visual = risk['visual_shared']
    joint = risk['joint_shared']
    sharing = risk['joint_independent'] - joint
    old_sharing = risk['visual_independent'] - visual
    contrast = dict(adding_audio=compare(visual - joint, 4),
                    sharing=compare(sharing, 4),
                    class_only=compare(risk['joint_class_only'] - joint, 4),
                    sharing_change=compare(sharing - old_sharing, 4))
    eligible = evidence['donor_rows'] >= 0
    swapped_groups, swapped_membership = np.unique(evidence['groups'][eligible], return_inverse=True)
    replacement = {}
    for method in ('shared', *CORES):
        predictions = evidence['acoustic_swap_' + method].astype(np.float64)
        replaced = ranked_probability_score(predictions, label[eligible])
        rise = replaced - risk['joint_' + method][:, :, eligible]
        replacement[method] = paired_reduction(rise.mean(1), swapped_membership, resamples, 3)
    return dict(n_clips=len(label), n_videos=len(groups), resamples=resamples, seeds=roster['seeds'],
                audio_summary=roster['audio_summary'], methods=methods, primary=contrast,
                benchmark_matched={core: compare(risk['joint_' + core] - joint, 2) for core in CORES},
                benchmark_audio={core: compare(risk['visual_' + core] - risk['joint_' + core], 2)
                                 for core in CORES},
                acoustic_swap=dict(n_clips=int(eligible.sum()), n_videos=len(swapped_groups), effects=replacement),
                selection=roster['selection'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data', type=Path)
    inputs.add_argument('--evidence', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--phase', choices=('compact', 'lnln', 'rmnet', 'report'), default='report')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1 or (args.data and not (args.runs and args.work)):
        parser.error('Use positive resamples and provide --runs and --work when using --data.')
    if args.phase != 'report':
        if not args.data:
            parser.error('Fitting requires --data.')
        if args.phase == 'compact':
            fit_compact(args.runs, args.work, args.data)
        else:
            fit_published(args.runs, args.work, args.data, args.phase)
        return
    if not args.output:
        parser.error('Reporting requires --output.')
    if args.evidence:
        with np.load(args.evidence, allow_pickle=False) as packet:
            data = dict(packet)
    else:
        data = collect(args.runs, args.work, args.data)
    report = summarize(data, args.resamples)
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(args.write_evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for key in ('primary', 'benchmark_matched', 'benchmark_audio'):
        for name, value in report[key].items():
            lower, upper = value['ci95_familywise']
            print(f'{key} / {name}: {value["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]')
    for name, value in report['acoustic_swap']['effects'].items():
        lower, upper = value['ci95_familywise']
        print(f'acoustic_swap / {name} RPS increase: {value["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]')


if __name__ == '__main__':
    main()
