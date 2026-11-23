"""Fit matched corrections on the visual feature change of a frozen LNLN model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ablate import ranked_probability_score, read_predictions
from benchmark import model_inputs
from compare_benchmarks import load_metadata, summarize_forecasts
from data import load_clips, observation_views
from ordinal_controls import paired_reduction
from published_baselines import build_published
from published_seed_effects import run_paths, verify_training
from score_anchor_effects import SEEDS
from train import BatchSource, index_batches


COHORTS = {'sims_v2': (1956, 325, 2122, 85), 'omg': (1731, 138, 834, 65)}
ARMS = ('shared', 'independent', 'class_only')
METHODS = ('frozen', *ARMS)


@torch.no_grad()
def encode_views(model, source, indices, batch):
    features, outputs = [], []
    captured = []

    def record_feature(layer, inputs):
        captured.append(inputs[0])

    hook = model.model.dmml[3].register_forward_pre_hook(record_feature)
    try:
        for condition in range(len(source.views)):
            captured.clear()
            rows = [model(*model_inputs(source, view, clips))[0]
                    for view, clips in index_batches(indices, batch, source.device, condition)]
            if len(rows) != len(captured):
                raise ValueError('Expected one penultimate feature per LNLN forward.')
            features.append(torch.cat(captured))
            outputs.append(torch.cat(rows))
    finally:
        hook.remove()
    embeddings = torch.stack(features)
    visible = source.views[:, 1, indices].any(-1)[..., None]
    visual_change = (embeddings - embeddings[-1]) * visible
    return visual_change, torch.stack(outputs)


def forward_correction(head, features, original, arm):
    change = head(features)
    logits = original[..., :3] + change[..., :3]
    if arm == 'independent':
        shift = change[..., 3]
    elif arm == 'shared':
        shift = (change[..., 2] - change[..., 0]) / 2
    else:
        shift = 0.
    return logits, (original[..., 3] + shift).tanh()


@torch.no_grad()
def tuning_rps(head, features, original, labels, arm):
    logits, _ = forward_correction(head, features[1:3], original[1:3], arm)
    return float(ranked_probability_score(logits.softmax(-1).cpu().numpy(), labels).mean())


def fit_head(arm, features, original, labels, intensity, split, seed):
    width = features.shape[-1]
    head = nn.Linear(width, 4 if arm == 'independent' else 3, bias=False).to(features.device)
    nn.init.zeros_(head.weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    rng = np.random.default_rng(seed + 1)
    fit, tune = split['fit'], split['tune']
    tune_features, tune_original = features[:, tune], original[:, tune]
    tune_label = labels[tune].cpu().numpy()
    best_weights = head.weight.detach().clone()
    best_rps = tuning_rps(head, tune_features, tune_original, tune_label, arm)
    selected_epoch = 0
    for epoch in range(1, 19):
        order = rng.permutation(fit)
        for start in range(0, len(fit), 96):
            indices = order[start:start + 96]
            view = rng.integers(len(features), size=len(indices))
            logits, score = forward_correction(head, features[view, indices], original[view, indices], arm)
            objective = F.cross_entropy(logits, labels[indices]) + .005 * head.weight.square().sum()
            if arm != 'class_only':
                objective = objective + .5 * F.smooth_l1_loss(score, intensity[indices], beta=.15)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        rps = tuning_rps(head, tune_features, tune_original, tune_label, arm)
        if rps < best_rps:
            best_weights, best_rps, selected_epoch = head.weight.detach().clone(), rps, epoch
    with torch.no_grad():
        head.weight.copy_(best_weights)
    return head, dict(epoch=selected_epoch, tuning_rps=best_rps, parameters=head.weight.numel())


def prepare(args, cohort):
    if cohort == 'sims_v2':
        path, layout, group_reference, margin = args.sims_data, 'new-videos', args.group_reference, 0.
    else:
        path, layout, group_reference, margin = args.omg_data, 'official', None, .10
    clips, split = load_clips(path, 24, layout, 1729, group_reference=group_reference, neutral_margin=margin)
    views = observation_views(clips, [.35, .65], 1777)
    views[:, 0] = False
    counts = tuple(len(split[key]) for key in ('fit', 'tune', 'eval')) + (len(np.unique(clips.groups[split['eval']])),)
    if counts != COHORTS[cohort]:
        raise ValueError(f'Unexpected fitting, tuning or evaluation counts for {cohort}.')
    source = BatchSource(clips, views, split['fit'], torch.device(args.device),
                         visual_only=True, stream_text=True)
    return clips, split, views, source


def collect_corpus(args, cohort, seeds=SEEDS):
    clips, split, views, source = prepare(args, cohort)
    heldout = split['eval']
    runs = run_paths(args.runs, seeds, cohort)['lnln_joint']
    metadata = load_metadata(runs)
    verify_training('lnln_joint', metadata, seeds, cohort)
    obs = dict(ids=clips.ids[heldout], groups=clips.groups[heldout], y=clips.label[heldout],
               sentiment=clips.sentiment[heldout], masks=views[:, :, heldout])
    probabilities, intensities = {arm: [] for arm in METHODS}, {arm: [] for arm in METHODS}
    selections = {arm: [] for arm in ARMS}
    dimensions = (source.text.shape[-1], 1, source.vision.shape[-1])
    lengths = (source.text.shape[1], source.audio.shape[1], source.vision.shape[1])
    benchmark_corpus = 'omg' if cohort == 'omg' else 'sims'
    for seed, run, meta in zip(seeds, runs, metadata, strict=True):
        archived = read_predictions(run)
        for key, value in obs.items():
            np.testing.assert_array_equal(value, archived[key], err_msg=key)
        model = build_published('lnln', args.upstream, benchmark_corpus, dimensions, lengths, 'joint', True)
        model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
        model = model.to(source.device).requires_grad_(False).eval()
        indices = np.concatenate([split[role] for role in ('fit', 'tune', 'eval')])
        features, original = encode_views(model, source, indices, meta['config']['batch'])
        del model
        fit_size, tune_size = len(split['fit']), len(split['tune'])
        local = dict(fit=np.arange(fit_size), tune=np.arange(fit_size, fit_size + tune_size))
        eval_start = fit_size + tune_size
        np.testing.assert_allclose(original[:, eval_start:].cpu().numpy(), archived['raw'], rtol=3e-5, atol=1e-5)
        with np.load(run / 'tuning.npz', allow_pickle=False) as tune:
            np.testing.assert_array_equal(tune['y'], clips.label[split['tune']])
            np.testing.assert_allclose(original[1:3, local['tune']].cpu().numpy(), tune['raw'][1:3],
                                       rtol=3e-5, atol=1e-5)
        rms = features[1:3, :fit_size].square().mean(dim=(0, 1)).sqrt().clamp_min(1e-3)
        features = features / rms
        label, intensity = source.label[indices], source.intensity[indices]
        eval_features, eval_original = features[:, eval_start:], original[:, eval_start:]
        probabilities['frozen'].append(archived['posterior'][1:3])
        intensities['frozen'].append(archived['score'][1:3])
        for arm in ARMS:
            head, selection = fit_head(arm, features, original, label, intensity, local, seed)
            with torch.no_grad():
                logits, scores = forward_correction(head, eval_features, eval_original, arm)
                torch.testing.assert_close(logits[-1], eval_original[-1, :, :3], atol=0, rtol=0)
                torch.testing.assert_close(scores[-1], eval_original[-1, :, 3].tanh(), atol=0, rtol=0)
                probabilities[arm].append(logits[1:3].softmax(-1).cpu().numpy())
                intensities[arm].append(scores[1:3].cpu().numpy() * source.intensity_scale)
            selections[arm].append(selection)
        print(f'{cohort} seed {seed}: selected correction epochs '
              f'{[selections[arm][-1]["epoch"] for arm in ARMS]}', flush=True)
    return dict(y=obs['y'], groups=obs['groups'], ids=obs['ids'], sentiment=obs['sentiment'],
                seeds=np.asarray(seeds), selection=np.asarray(json.dumps(selections)),
                forecasts=np.stack([probabilities[arm] for arm in METHODS]),
                scores=np.stack([intensities[arm] for arm in METHODS]))


def describe(cohort, evidence, draws):
    labels, intensity = evidence['y'], evidence['sentiment']
    groups, membership = np.unique(evidence['groups'], return_inverse=True)
    if (len(labels), len(groups)) != COHORTS[cohort][2:] or evidence['seeds'].tolist() != list(SEEDS):
        raise ValueError('Reproduce the declared held-out groups and all ten paired seeds.')
    if len(np.unique(evidence['ids'])) != len(labels):
        raise ValueError('Each clip must have one forecast per seed and gap.')
    selection = json.loads(evidence['selection'].item())
    methods, rps, mae = {}, {}, {}
    for name, forecasts, scores in zip(METHODS, evidence['forecasts'], evidence['scores'], strict=True):
        summary, rps[name], mae[name] = summarize_forecasts(forecasts, scores, labels, intensity)
        methods[name] = dict(**summary, selection=selection.get(name, []))
    rps_effects = {name: paired_reduction(rps[name] - rps['shared'], membership, draws, 6)
                   for name in ('frozen', 'independent', 'class_only')}
    mae_effects = {name: paired_reduction(mae[name] - mae['shared'], membership, draws, 4)
                   for name in ('frozen', 'independent')}
    return dict(clips=len(labels), videos=len(groups), methods=methods, rps_reductions=rps_effects,
                mae_reductions=mae_effects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', type=Path)
    source.add_argument('--evidence', type=Path, nargs='+')
    parser.add_argument('--cohort', choices=(*COHORTS, 'both'), default='both')
    parser.add_argument('--sims-data', type=Path)
    parser.add_argument('--group-reference', type=Path)
    parser.add_argument('--omg-data', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write-evidence', type=Path)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose a positive number of bootstrap draws.')
    names = tuple(COHORTS) if args.cohort == 'both' else (args.cohort,)
    if args.runs:
        if not args.upstream or ('sims_v2' in names and not all((args.sims_data, args.group_reference))) or (
                'omg' in names and not args.omg_data):
            parser.error('Supply the selected feature packets, CH-SIMS group reference, and pinned LNLN source.')
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        data = {f'{cohort}_{key}': value for cohort in names
                for key, value in collect_corpus(args, cohort).items()}
    else:
        data = {}
        for path in args.evidence:
            with np.load(path, allow_pickle=False) as archive:
                if data.keys() & set(archive.files):
                    parser.error('Each case must be supplied only once.')
                data.update(dict(archive))
    cases = {name: describe(name, {key.removeprefix(f'{name}_'): value for key, value in data.items()
                                   if key.startswith(f'{name}_')}, args.resamples) for name in names}
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(args.write_evidence)
    report = dict(seeds=SEEDS, resamples=args.resamples, cases=cases)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, case in cases.items():
        for comparison, contrast in case['rps_reductions'].items():
            print(name, comparison, f'{contrast["reduction"]:+.6f}', contrast['ci95_familywise'])


if __name__ == '__main__':
    main()
