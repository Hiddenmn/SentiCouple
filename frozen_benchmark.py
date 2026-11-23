"""Compare the six OMG-Emotion models on unseen speech and translated reviews."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import read_predictions
from benchmark import model_inputs
from compare_benchmarks import load_metadata, summarize_forecasts
from data import source_groups
from mosei_results import earlier_corpus_runs, paired_summary, predictor_for, profile_cached
from msp_face_effects import NEUTRAL_MARGIN, checked_target, reuse_refit_masks
from score_anchor_effects import SEEDS
from transfer_moud import check_configuration, matched_input


BASELINES = ('tfr-net', 'lnln', 'rm-net', 'api', 'ucmr-net')


def forecasts(predict, batches, indices, batch_size):
    probabilities, scores = [], []
    for gap in (1, 2):
        probability_rows, score_rows = [], []
        for start in range(0, len(indices), batch_size):
            samples = torch.as_tensor(indices[start:start + batch_size], device=batches.device)
            view = torch.full_like(samples, gap)
            probability, score = predict(model_inputs(batches, view, samples))
            probability_rows.append(probability.cpu().numpy())
            score_rows.append(score.cpu().numpy())
        probabilities.append(np.concatenate(probability_rows))
        scores.append(np.concatenate(score_rows))
    return np.stack(probabilities), np.stack(scores)


def target_batches(args, paths):
    config = check_configuration(load_metadata(paths['SentiCouple']))
    msp_face = args.cohort == 'msp_face'
    margin = NEUTRAL_MARGIN if msp_face else 0.
    batches, source_ids, source_index, target, indices = matched_input(
        args.source_data, args.target, config, torch.device(args.device), target_neutral_margin=margin)
    if source_groups(args.source_data).intersection(target.groups):
        raise ValueError('The source and target cohorts share a video.')
    if msp_face:
        cohort = json.loads(args.msp_manifest.read_text(encoding='utf-8'))
        checked = checked_target(args.target, cohort)
        np.testing.assert_array_equal(target.ids, checked['ids'])
        np.testing.assert_array_equal(target.label, checked['y'])
        reuse_refit_masks(args.msp_match, batches, indices, checked)
        groups = checked['participant_group']
    else:
        groups = target.groups
    return batches, source_ids, source_index, target, indices, groups


@torch.inference_mode()
def transfer(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    paths = earlier_corpus_runs(args.runs, args.runs, SEEDS, BASELINES, 'omg')
    batches, source_ids, source_index, target, indices, groups = target_batches(args, paths)
    evidence = dict(cohort=np.asarray(args.cohort), ids=target.ids, videos=target.groups,
                    groups=groups, y=target.label, intensity=target.sentiment,
                    masks=batches.views[:, :, indices].cpu().numpy(), seeds=np.asarray(SEEDS),
                    methods=np.asarray(list(paths)))
    upstream = dict(zip(BASELINES, args.upstream, strict=True))
    for name, runs in paths.items():
        target_predictions, target_scores = [], []
        for seed, run, metadata in zip(SEEDS, runs, load_metadata(runs), strict=True):
            data_config = metadata['config']
            if data_config['seed'] != seed or data_config['inputs'] != 'visual':
                raise ValueError(f'{run.name} does not match the source protocol.')
            archive = read_predictions(run)
            np.testing.assert_array_equal(archive['ids'], source_ids)
            np.testing.assert_array_equal(archive['masks'], batches.views[:, :, source_index].cpu().numpy())
            make_prediction, _, (probability_key, score_key) = predictor_for(run, name, batches, upstream.get(name))
            batch_size = data_config['batch']
            source_probability, source_score = forecasts(make_prediction, batches, source_index, batch_size)
            np.testing.assert_allclose(source_probability, archive[probability_key][1:3], atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(source_score, archive[score_key][1:3], atol=1e-4, rtol=1e-4)
            probability, score = forecasts(make_prediction, batches, indices, batch_size)
            target_predictions.append(probability)
            target_scores.append(score)
        evidence[f'{name}_probability'] = np.stack(target_predictions)
        evidence[f'{name}_score'] = np.stack(target_scores)
        print(f'{args.cohort}: matched ten source replays and transferred {name}.', flush=True)
    return evidence


@torch.inference_mode()
def profile_transfer(args, evidence):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    paths = earlier_corpus_runs(args.runs, args.runs, SEEDS, BASELINES, 'omg')
    source, _, _, target, indices, groups = target_batches(args, paths)
    np.testing.assert_array_equal(evidence['ids'], target.ids)
    np.testing.assert_array_equal(evidence['groups'], groups)
    np.testing.assert_array_equal(evidence['masks'], source.views[:, :, indices].cpu().numpy())
    upstream = dict(zip(BASELINES, args.upstream, strict=True))
    predictors, archived, timings = {}, {}, {}
    for name, (run, *_) in paths.items():
        metadata, = load_metadata([run])
        predict, parameters, _ = predictor_for(run, name, source, upstream.get(name))
        expected = [evidence[f'{name}_{output}'][0] for output in ('probability', 'score')]
        actual = forecasts(predict, source, indices, metadata['config']['batch'])
        for estimate, stored in zip(actual, expected, strict=True):
            np.testing.assert_allclose(estimate, stored, atol=1e-4, rtol=1e-4, err_msg=name)
        predictors[name], archived[name] = predict, expected
        timings[name] = dict(trainable_parameters=parameters, batches={}, max_batch_difference={})
        print(f'{args.cohort}: replayed {name} at its original batch size.', flush=True)
    return profile_cached(source, indices, predictors, archived, timings)


def summarize(evidence, resamples):
    labels = evidence['y']
    methods, risk, error = {}, {}, {}
    name = evidence['cohort'].item()
    _, groups = np.unique(evidence['groups'], return_inverse=True)
    np.testing.assert_array_equal(evidence['seeds'], SEEDS)
    for model in evidence['methods']:
        result, risk[model], error[model] = summarize_forecasts(
            evidence[f'{model}_probability'], evidence[f'{model}_score'], labels, evidence['intensity'])
        if name == 'moud':
            result.pop('mae')
        methods[model] = result
    risk_differences = {k: risk[k] - risk['SentiCouple'] for k in BASELINES}
    result = dict(cohort=name, clips=len(labels), videos=int(len(np.unique(evidence['videos']))),
                  groups=int(len(np.unique(groups))), seeds=list(SEEDS), resamples=resamples,
                  methods=methods, rps_gain=paired_summary(risk_differences, groups, resamples, len(BASELINES)))
    if name == 'msp_face':
        differences = {k: error[k] - error['SentiCouple'] for k in BASELINES}
        result['mae_gain'] = paired_summary(differences, groups, resamples, len(BASELINES))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path(__file__).resolve().parent / 'runs')
    parser.add_argument('--cohort', choices=('msp_face', 'moud'))
    parser.add_argument('--source-data', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--msp-manifest', type=Path)
    parser.add_argument('--msp-match', type=Path)
    parser.add_argument('--upstream', nargs=5, type=Path, help='TFR-Net, LNLN, RM-Net, API and UCMR-Net checkouts.')
    parser.add_argument('--evidence', type=Path, help='Summarize the archived forecasts instead of running inference.')
    parser.add_argument('--profile', action='store_true', help='Time the frozen seed-97 models on the target features.')
    parser.add_argument('--save-evidence', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    if args.profile or not args.evidence:
        if not all((args.cohort, args.source_data, args.target, args.upstream)):
            parser.error('Supply the cohort, source data, target data and five reference checkouts.')
        if args.cohort == 'msp_face' and not all((args.msp_manifest, args.msp_match)):
            parser.error('Supply the MSP-Face cohort manifest and reference masks.')
    if args.profile and (not args.evidence or not args.device.startswith('cuda')):
        parser.error('Profile cached features on CUDA using the archived forecasts.')
    if args.evidence:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    else:
        evidence = transfer(args)
        if args.save_evidence:
            args.save_evidence.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_evidence, **evidence)
    report = summarize(evidence, args.resamples)
    if args.profile:
        if args.cohort != report['cohort']:
            parser.error('The archived forecasts belong to a different cohort.')
        report['timing'] = profile_transfer(args, evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
