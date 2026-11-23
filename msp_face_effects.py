"""Evaluate matched MSP-Face fits or frozen OMG-Emotion transfer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from benchmark import forecast, predict
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from data import source_groups
from encode_omg import NEUTRAL_MARGIN, sha256
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from prepare_msp_face import write_json
from published_baselines import build_published
from published_inference import load_weights
from score_anchor_effects import SEEDS
from transfer_moud import check_configuration, matched_input, replay_seed


PRIMARY = 'independent_intensity'
SECONDARY = ('anchor', 'fusion', 'intensity_shuffled', 'intensity_projection', 'lnln', 'rmnet')
HEADS = ('intensity', PRIMARY, *SECONDARY[:-2])
FROZEN_SECONDARY = ('anchor', 'fusion', 'lnln', 'rmnet')
FROZEN = ('intensity', PRIMARY, *FROZEN_SECONDARY)
BASELINES = {'lnln': 'msp_face_lnln_seed', 'rmnet': 'msp_face_rmnet_seed'}
SOURCE_BASELINES = {'lnln': 'omg_lnln_joint_seed', 'rmnet': 'omg_rmnet_uniform_seed'}
OPTIMIZATION = {'lnln': dict(epochs=32, patience=8, batch=32, lr=.001, weight_decay=.01),
                'rmnet': dict(epochs=48, patience=10, batch=64, lr=.0001, weight_decay=.001)}


def shared_splits(cohort):
    return {role: dict(n=summary['clips'], videos=summary['videos'], labels=summary['classes'])
            for role, summary in zip(('fit', 'tune', 'eval'),
                                     (cohort['splits'][part] for part in ('train', 'valid', 'test')),
                                     strict=True)}


def checked_target(target, cohort):
    if sha256(target) != cohort['test_sha256'] or cohort['neutral_margin'] != NEUTRAL_MARGIN:
        raise ValueError('The test packet must match the recorded MSP-Face cohort and neutral interval.')
    with np.load(target, allow_pickle=False) as packet:
        ids, score, participants = (packet[key].copy() for key in ('id', 'regression_labels', 'participant_group'))
    np.testing.assert_array_equal(ids, cohort['ids']['test'])
    groups = np.array([clip.split('$_$', 1)[0] for clip in ids])
    labels = np.where(np.abs(score) <= NEUTRAL_MARGIN, 0, np.sign(score)).astype(int) + 1
    summary = cohort['splits']['test']
    np.testing.assert_array_equal(np.bincount(labels, minlength=3), summary['classes'])
    if (len(ids) != summary['clips'] or len(np.unique(groups)) != summary['videos']
            or len(np.unique(participants)) != summary['participant_groups']):
        raise ValueError('The test packet does not contain the reported clips or independent units.')
    return dict(ids=ids, sentiment=score, y=labels, groups=groups, participant_group=participants)


def refits(runs, target, cohort):
    paths = [runs / f'msp_face_heads_seed{seed}' for seed in SEEDS]
    metadata = load_metadata(paths)
    predictions = read_predictions(paths[0])
    for field in ('ids', 'y', 'sentiment', 'groups'):
        np.testing.assert_array_equal(predictions[field], target[field])
    compact = load_matched(paths, predictions)
    splits = shared_splits(cohort)
    config = metadata[0]['config']
    for seed, packet, record in zip(SEEDS, compact, metadata, strict=True):
        if (record['config']['seed'] != seed or record['splits'] != splits
                or any(record['config'].get(field) != value for field, value in config.items()
                       if field not in ('seed', 'output'))):
            raise ValueError('The compact fits must share the same input and selection settings.')
        for head in HEADS:
            np.testing.assert_array_equal(packet[head][-1], packet['anchor'][-1])
    data = dict(study=np.asarray('refit'), masks=predictions['masks'], **target)
    for head in HEADS:
        data[head] = np.stack([packet[head] for packet in compact])
        data[f'{head}_score'] = np.stack([packet[f'{head}_score'] for packet in compact])
    sizes = {head: metadata[0]['selection'][head]['parameters'] for head in HEADS if head != 'anchor'}
    for name, prefix in BASELINES.items():
        baseline_paths = [runs / f'{prefix}{seed}' for seed in SEEDS]
        baseline_metadata = load_metadata(baseline_paths)
        baseline = load_matched(baseline_paths, predictions)
        for seed, match_path, record in zip(SEEDS, paths, baseline_metadata, strict=True):
            settings = record['config']
            if (settings['seed'] != seed or settings['corpus'] != 'msp-face' or settings['head'] != 'joint'
                    or settings['neutral_margin'] != NEUTRAL_MARGIN or settings['split'] != 'official'
                    or settings['inputs'] != 'visual' or Path(settings['match_run']).resolve() != match_path.resolve()
                    or any(settings[key] != value for key, value in OPTIMIZATION[name].items())
                    or (name == 'rmnet' and settings['rmnet_class_weighting'] != 'uniform')):
                raise ValueError(f'The {name} fits must follow the MSP-Face reference protocol.')
        data[name] = np.stack([packet['posterior'] for packet in baseline])
        data[f'{name}_score'] = np.stack([packet['score'] for packet in baseline])
        sizes[name] = baseline_metadata[0]['trainable_parameters']
    data['parameters'] = np.asarray(json.dumps(sizes))
    return data


def frozen_core(name, run, upstream, seed, batches, source_ids, source_index, target_index):
    metadata, = load_metadata([run])
    config = metadata['config']
    if (config['seed'] != seed or config['corpus'] != 'omg' or config['head'] != 'joint'
            or config['inputs'] != 'visual' or config['neutral_margin'] != NEUTRAL_MARGIN
            or config['split'] != 'official'
            or (name == 'rmnet' and config['rmnet_class_weighting'] != 'uniform')):
        raise ValueError('Expected the same-seed joint OMG-Emotion fusion reference.')
    reference = read_predictions(run)
    np.testing.assert_array_equal(source_ids, reference['ids'])
    np.testing.assert_array_equal(batches.views[:, :, source_index].cpu().numpy(), reference['masks'])
    dimensions = (batches.text.shape[-1], max(1, batches.audio.shape[-1]), batches.vision.shape[-1])
    lengths = (batches.text.shape[1], batches.audio.shape[1], batches.vision.shape[1])
    root = upstream or Path(config['upstream'])
    model = build_published(config['model'], root, 'omg', dimensions, lengths, 'joint', True)
    model = load_weights(model, run / 'model.pt', batches.device)
    archived = predict(model, batches, source_index, config['batch'], range(4))
    probability, score = forecast(archived, 'joint', batches.intensity_scale, None)
    np.testing.assert_allclose(probability, reference['posterior'], rtol=3e-5, atol=6e-6)
    np.testing.assert_allclose(score, reference['score'], rtol=3e-5, atol=6e-6)
    raw = predict(model, batches, target_index, config['batch'], range(4))
    return forecast(raw, 'joint', batches.intensity_scale, None)[0]


def reuse_refit_masks(refit_run, batches, target_index, target):
    """Evaluate the frozen models at the gaps used by the target-trained fits."""
    metadata, = load_metadata([refit_run])
    config = metadata['config']
    if (config['split'] != 'official' or config['inputs'] != 'visual'
            or config['neutral_margin'] != NEUTRAL_MARGIN):
        raise ValueError('Expected a compact MSP-Face fit with the official visual-only split.')
    with np.load(refit_run / 'predictions.npz', allow_pickle=False) as reference:
        for field in ('ids', 'sentiment', 'y', 'groups'):
            np.testing.assert_array_equal(reference[field], target[field], err_msg=field)
        matched = reference['masks']
    previous = batches.views[:, :, target_index].cpu().numpy()
    if matched.shape != previous.shape or matched.dtype != np.bool_:
        raise ValueError('The recorded target views must be Boolean and have identical shape.')
    np.testing.assert_array_equal(matched[[0, 3]], previous[[0, 3]])
    np.testing.assert_array_equal(matched.sum(axis=-1), previous.sum(axis=-1))
    batches.views[:, :, target_index] = torch.from_numpy(matched).to(batches.device)


def frozen(runs, source, target_path, target, device, upstream, match_refit=None):
    paths = [runs / f'omg_pointwise_heads_seed{seed}' for seed in SEEDS]
    config = check_configuration(load_metadata(paths))
    batches, source_ids, source_index, clips, target_index = matched_input(
        source, target_path, config, device, target_neutral_margin=NEUTRAL_MARGIN)
    np.testing.assert_array_equal(clips.ids, target['ids'])
    np.testing.assert_array_equal(clips.label, target['y'])
    if source_groups(source).intersection(target['groups']):
        raise ValueError('Frozen source and target clips share a video.')
    if match_refit is not None:
        reuse_refit_masks(match_refit, batches, target_index, target)
    predictions = []
    for seed, path in zip(SEEDS, paths, strict=True):
        fitted = replay_seed(path, [], config, batches, source_ids, source_index, target_index)
        for name, prefix in SOURCE_BASELINES.items():
            fitted[name] = frozen_core(name, runs / f'{prefix}{seed}', upstream[name], seed,
                                       batches, source_ids, source_index, target_index)
        predictions.append(fitted)
        print(f'Replayed all OMG-Emotion models for seed {seed} and predicted MSP-Face.', flush=True)
    data = dict(study=np.asarray('frozen'), masks=batches.views[:, :, target_index].cpu().numpy(), **target)
    for name in FROZEN:
        data[name] = np.stack([packet[name] for packet in predictions])
    if match_refit is not None:
        data['gap_protocol'] = np.asarray('refit_test_masks')
    return data


def summarize(evidence, resamples):
    seeds = evidence['seeds']
    np.testing.assert_array_equal(seeds, SEEDS)
    study = evidence['study'].item()
    y, scores = evidence['y'], evidence['sentiment']
    groups, membership = np.unique(evidence['participant_group'], return_inverse=True)
    methods, risk, axes = {}, {}, {}
    score_errors = {}
    names = (*HEADS, *BASELINES) if study == 'refit' else FROZEN
    for name in names:
        probability = evidence[name]
        if probability.shape != (len(seeds), 4, len(y), 3) or not np.isfinite(probability).all():
            raise ValueError(f'Unexpected probability forecasts for {name}.')
        np.testing.assert_allclose(probability.sum(-1), 1, rtol=0, atol=2e-6)
        loss = ranked_probability_score(probability, y)
        risk[name] = loss[:, 1:3].mean(1)
        axes[name] = ((coordinates(probability[:, 1:3]) - outcomes(y)) ** 2 / 4).mean(1)
        np.testing.assert_allclose(axes[name].sum(-1), risk[name], rtol=0, atol=2e-7)
        methods[name] = dict(rps_summary(probability[:, 1:3], y),
                             native_rps=float(loss[:, 0].mean()), no_vision_rps=float(loss[:, 3].mean()),
                             rps_by_gap=loss[:, 1:3].mean(axis=(0, 2)).tolist(),
                             signed_error=float(axes[name][..., 0].mean()),
                             neutral_error=float(axes[name][..., 1].mean()))
        if f'{name}_score' in evidence:
            score_errors[name] = np.abs(evidence[f'{name}_score'][:, 1:3] - scores).mean(1)
            methods[name]['mae'] = float(score_errors[name].mean())

    primary_gain = risk[PRIMARY] - risk['intensity']
    primary = paired_reduction(primary_gain, membership, resamples, 1)
    secondary_names = SECONDARY if study == 'refit' else FROZEN_SECONDARY
    secondary = {name: paired_reduction(risk[name] - risk['intensity'], membership, resamples, len(secondary_names))
                 for name in secondary_names}
    primary_decomposition = (axes[PRIMARY] - axes['intensity']).mean(axis=(0, 1))
    np.testing.assert_allclose(primary_decomposition.sum(), primary['reduction'], rtol=0, atol=2e-8)
    result = dict(study=study, seeds=seeds.tolist(), resamples=resamples, n_clips=len(y),
                  n_videos=len(np.unique(evidence['groups'])), n_participant_groups=len(groups),
                  cohort=json.loads(evidence['cohort'].item()), methods=methods,
                  primary=primary, secondary=secondary,
                  primary_signed_reduction=float(primary_decomposition[0]),
                  primary_neutral_reduction=float(primary_decomposition[1]))
    if score_errors:
        result['primary_mae_reduction'] = paired_reduction(score_errors[PRIMARY] - score_errors['intensity'],
                                                            membership, resamples, 1)
        result['parameters'] = json.loads(evidence['parameters'].item())
        result['primary_reduction_by_polarity'] = {
            name: float(primary_gain[:, y == polarity].mean())
            for polarity, name in enumerate(('negative', 'neutral', 'positive'))
            if np.any(y == polarity)
        }
    if 'gap_protocol' in evidence:
        result['gap_protocol'] = evidence['gap_protocol'].item()
    if study == 'refit' or 'gap_protocol' in evidence:
        ensemble_risk = {
            name: float(ranked_probability_score(evidence[name][:, 1:3].mean(0, dtype=np.float64), y).mean())
            for name in ('intensity', PRIMARY)
        }
        ensemble_gain = ensemble_risk[PRIMARY] - ensemble_risk['intensity']
        result['sharing_seed_decomposition'] = dict(mean_forecast_reduction=ensemble_gain,
                                                    dispersion_reduction=primary['reduction'] - ensemble_gain,
                                                    mean_forecast_rps=ensemble_risk)
    return result


def adaptation_comparison(refit, transfer, resamples):
    """Measure how the paired sharing and visual-fusion gains change after refitting."""
    if refit['study'].item() != 'refit' or transfer['study'].item() != 'frozen':
        raise ValueError('Compare a refit with a frozen transfer forecast.')
    for field in ('seeds', 'ids', 'y', 'sentiment', 'groups', 'participant_group', 'masks'):
        np.testing.assert_array_equal(refit[field], transfer[field], err_msg=field)
    _, membership = np.unique(refit['participant_group'], return_inverse=True)
    y = refit['y']
    shared_refit = ranked_probability_score(refit['intensity'][:, 1:3], y).mean(1)
    shared_transfer = ranked_probability_score(transfer['intensity'][:, 1:3], y).mean(1)
    comparisons = {'sharing_gain_change': PRIMARY, 'visual_gain_change': 'anchor'}
    results = {}
    for name, comparator in comparisons.items():
        refit_gain = ranked_probability_score(refit[comparator][:, 1:3], y).mean(1) - shared_refit
        transfer_gain = ranked_probability_score(transfer[comparator][:, 1:3], y).mean(1) - shared_transfer
        results[name] = paired_reduction(refit_gain - transfer_gain, membership, resamples, len(comparisons))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--source-data', type=Path, help='Use frozen OMG-Emotion fits.')
    parser.add_argument('--match-refit', type=Path, help='Use test masks from a completed compact MSP-Face run.')
    parser.add_argument('--lnln-upstream', type=Path, help='Override the source checkpoint’s LNLN source path.')
    parser.add_argument('--rmnet-upstream', type=Path, help='Override the source checkpoint’s RM-Net source path.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--evidence', type=Path, help='Recalculate the intervals from paired forecasts.')
    parser.add_argument('--frozen-evidence', type=Path, help='Compare refits with frozen forecasts at identical masks.')
    parser.add_argument('--save-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Use at least one paired bootstrap resample.')
    if args.match_refit and (args.evidence or args.source_data is None):
        parser.error('--match-refit applies when computing new frozen forecasts.')
    if args.evidence:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    else:
        if not (args.target and args.cohort and args.runs):
            parser.error('Provide the test packet, cohort and run directory.')
        cohort = json.loads(args.cohort.read_text(encoding='utf-8'))
        target = checked_target(args.target, cohort)
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        if args.source_data:
            upstream = dict(lnln=args.lnln_upstream, rmnet=args.rmnet_upstream)
            evidence = frozen(args.runs, args.source_data, args.target, target, torch.device(args.device), upstream,
                              match_refit=args.match_refit)
        else:
            evidence = refits(args.runs, target, cohort)
        evidence['seeds'] = np.asarray(SEEDS)
        evidence['cohort'] = np.asarray(json.dumps({key: value for key, value in cohort.items() if key != 'ids'}))
        if args.save_evidence:
            np.savez_compressed(args.save_evidence, **evidence)
    report = summarize(evidence, args.resamples)
    if args.frozen_evidence:
        with np.load(args.frozen_evidence, allow_pickle=False) as packet:
            transfer = dict(packet)
        report['adaptation'] = adaptation_comparison(evidence, transfer, args.resamples)
    write_json(args.output, report)
    print(dict(study=report['study'], clips=report['n_clips'], primary=report['primary']), flush=True)


if __name__ == '__main__':
    main()
