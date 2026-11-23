"""Re-evaluate frozen classifiers under new visual block placements."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions, seed_video_interval
from data import load_clips, observation_views
from networks import AudiovisualResidual, TextAnchor
from train import BatchSource, index_batches


RATES = (.20, .35, .65, .80)
MASK_SEEDS = (1801, 1811, 1823, 1831, 1847)
ARMS = ('anchor', 'fusion', 'independent_intensity', 'intensity')


def matching_config(runs):
    configs = [json.loads((run / 'run.json').read_text(encoding='utf-8'))['config'] for run in runs]
    first = configs[0]
    for config in configs:
        if config['inputs'] != 'visual' or config['backbone'] not in ('attention', 'stream-query'):
            raise ValueError('Use the text-and-vision attention runs.')
        defaults = {'visual_encoding': 'normalized', 'neutral_margin': 0.}
        for key in ('split', 'split_seed', 'mask_seed', 'bins', 'width', 'data', 'backbone', *defaults):
            if config.get(key, defaults.get(key)) != first.get(key, defaults.get(key)):
                raise ValueError(f'Expected the same {key} across runs.')
    return first, [config['seed'] for config in configs]


@torch.inference_mode()
def forecast(run, source, heldout, config):
    width, bins, batch = (config[key] for key in ('width', 'bins', 'batch'))
    anchor = TextAnchor(source.text.shape[-1], bins, width, visual_only=True).to(source.device)
    anchor.load_state_dict(torch.load(run / 'anchor.pt', map_location=source.device, weights_only=True))
    anchor.eval()
    fusion = {}
    for arm in ARMS[1:]:
        model = AudiovisualResidual(0, source.vision.shape[-1], width, attention_bins=bins,
                                    independent_intensity=arm == 'independent_intensity',
                                    text_query=config['backbone'] == 'attention',
                                    visual_encoding=config.get('visual_encoding', 'normalized')).to(source.device)
        model.load_state_dict(torch.load(run / f'{arm}.pt', map_location=source.device, weights_only=True))
        fusion[arm] = model.eval()
    forecasts = {arm: [] for arm in ARMS}
    for condition in range(len(source.views)):
        rows = {arm: [] for arm in ARMS}
        for view, clips in index_batches(heldout, batch, source.device, condition):
            context, prior = source.anchor(anchor, view, clips)
            rows['anchor'].append(prior.softmax(-1).cpu().numpy())
            for arm, model in fusion.items():
                logits = source.fusion(model, context, prior, view, clips, clips)
                rows[arm].append(logits.softmax(-1).cpu().numpy())
        for arm in ARMS:
            forecasts[arm].append(np.concatenate(rows[arm]))
    return {arm: np.stack(rows) for arm, rows in forecasts.items()}


def widths_report(all_losses, groups, resamples):
    _, membership = np.unique(groups, return_inverse=True)
    reports = {}
    for index, name in enumerate(('native', *(f'block{int(rate * 100)}' for rate in RATES), 'no_vision')):
        mean_loss = {arm: loss[:, index, :].mean() for arm, loss in all_losses.items()}
        controls = {}
        for arm in ARMS[:-1]:
            effect = all_losses[arm][:, index, :] - all_losses['intensity'][:, index, :]
            controls[arm] = dict(rps_reduction=float(effect.mean()),
                                 ci95_seeds_and_videos=seed_video_interval(effect, membership, resamples))
        reports[name] = dict(rps={arm: float(value) for arm, value in mean_loss.items()}, versus_shared=controls)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--group-reference', type=Path)
    parser.add_argument('--eval-groups', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if min(args.threads, args.resamples) < 1:
        parser.error('Threads and resamples must be positive.')
    torch.set_num_threads(args.threads)
    config, seeds = matching_config(args.runs)
    if len(seeds) != len(set(seeds)):
        raise ValueError('Use distinct training seed checkpoints.')
    parameters = dict(group_reference=args.group_reference, eval_groups=args.eval_groups,
                      neutral_margin=config.get('neutral_margin', 0.))
    clips, split = load_clips(args.data, config['bins'], config['split'], config['split_seed'], **parameters)
    old_views = observation_views(clips, [.35, .65], config['mask_seed'])
    new_views = [observation_views(clips, RATES, seed)[1:-1] for seed in MASK_SEEDS]
    views = np.concatenate([old_views, *new_views])
    views[:, 0] = False
    test = split['eval']
    y = clips.label[test]
    source = BatchSource(clips, views, split['fit'], torch.device(args.device), visual_only=True, stream_text=True)
    losses = {arm: [] for arm in ARMS}
    errors = {}
    for run in args.runs:
        archive = read_predictions(run)
        np.testing.assert_array_equal(clips.ids[test], archive['ids'])
        np.testing.assert_array_equal(views[:4, :, test], archive['masks'])
        np.testing.assert_array_equal(y, archive['y'])
        model_forecasts = forecast(run, source, test, config)
        errors[run.name] = {}
        for arm, probability in model_forecasts.items():
            np.testing.assert_allclose(probability[:4], archive[arm], atol=4e-6, rtol=2e-5, err_msg=arm)
            np.testing.assert_array_equal(probability[3], model_forecasts['anchor'][3], err_msg=arm)
            errors[run.name][arm] = float(np.max(np.abs(probability[:4] - archive[arm])))
            score = ranked_probability_score(probability, y)
            draws = score[4:].reshape(len(MASK_SEEDS), len(RATES), len(y))
            losses[arm].append(np.concatenate([score[:1], draws.mean(axis=0), score[3:4]]))
        print(f'{run.name}: reconciled four original views and scored {len(new_views) * len(RATES)} new gap views',
              flush=True)
    all_losses = {arm: np.stack(rows) for arm, rows in losses.items()}
    result = dict(training_seeds=seeds, gap_seeds=list(MASK_SEEDS), gap_widths=list(RATES),
                  n_clips=len(y), n_videos=int(len(np.unique(clips.groups[test]))),
                  max_abs_reconstruction_error=errors,
                  comparisons=widths_report(all_losses, clips.groups[test], args.resamples))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
