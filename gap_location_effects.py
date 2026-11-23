"""Train with resampled visual gaps and compare forecasts over every allowed start."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate import grouped_seed_video_interval, ranked_probability_score
from benchmark import model_inputs
from compare_benchmarks import load_matched, load_metadata
from networks import AudiovisualResidual, intensity_anchor, sentiment_shift
from polarity_pool import coordinates, outcomes
from published_inference import load_weights, predictor
from published_seed_effects import run_paths, verify_training
from score_anchor_effects import SEEDS
from text_weighted_loss import RUN_NAME, anchor_for, load_references, source_for
from train import anchor_outputs, calibrated_head_forecasts, fit_anchor, fit_fusion


RATES = (.35, .65)
GAPS = (1, 2)
VARIANTS = (
    ('fixed_shared', 'intensity', False, False),
    ('anchor_resampled', 'intensity', True, False),
    ('fusion_resampled', 'intensity', False, True),
    ('both_shared', 'intensity', True, True),
    ('fixed_independent', 'independent_intensity', False, False),
    ('both_independent', 'independent_intensity', True, True),
)
BASELINES = ('rmnet_uniform', 'lnln_joint')
FIELDS = ('signed_rps', 'neutral_rps', 'accuracy', 'mae', 'worst_rps')


def placement_views(original):
    """Keep the first four views and append every visual-gap start per clip."""
    native = original[0, 1]
    bins = np.arange(native.shape[-1])
    first = np.where(native, bins, len(bins)).min(axis=1)
    last = np.where(native, bins, -1).max(axis=1)
    span = (last - first + 1).clip(0)
    counts = np.ones((len(original), len(native)), dtype=np.int64)
    offsets = np.arange(len(original))
    appended = [original]
    next_view = len(original)
    for gap, rate in zip(GAPS, RATES, strict=True):
        width = np.maximum(1, np.rint(span * rate).astype(np.int64))
        counts[gap] = np.maximum(1, span - width + 1)
        ranks = np.arange(counts[gap].max())[:, None]
        starts = first + ranks
        removed = (bins >= starts[..., None]) & (bins < (starts + width)[..., None])
        masks = native & ~removed
        valid = ranks < counts[gap]
        contains = ((masks == original[gap, 1]).all(-1) & valid).any(0)
        if not contains.all():
            raise ValueError('Every original gap must occur in the corresponding set of start positions.')
        views = np.zeros((len(masks), *original.shape[1:]), dtype=bool)
        views[:, 1] = masks
        appended.append(views)
        offsets[gap] = next_view
        next_view += len(views)
    return np.concatenate(appended), offsets, counts


def sampler(offsets, counts, seed, resample):
    """The original RNG still determines severity and clip order."""
    placement_rng = np.random.default_rng(seed)
    device = offsets.device

    def draw(severity_rng, clips):
        category = torch.as_tensor(severity_rng.integers(len(offsets), size=len(clips)), device=device)
        if not resample:
            return category
        uniform = torch.as_tensor(placement_rng.random(len(clips)), device=device)
        start = (uniform * counts[category, clips]).long()
        return offsets[category] + start

    return draw


def metrics(posterior, score, label, sentiment):
    posterior = posterior.astype(np.float64)
    np.testing.assert_allclose(posterior.sum(-1), 1., atol=2e-7, rtol=0)
    terms = np.square(coordinates(posterior) - outcomes(label)) / 4
    correct = posterior.argmax(-1) == label
    return np.concatenate((terms, correct[..., None], np.abs(score - sentiment)[..., None]), axis=-1)


@torch.inference_mode()
def all_location_metrics(forecast, source, held, label, sentiment, offsets, counts, batch_size):
    results = []
    for gap in GAPS:
        ranks, local_clip = np.nonzero(np.arange(counts[gap, held].max())[:, None] < counts[gap, held])
        indices = torch.as_tensor(held[local_clip], device=source.device)
        views = torch.as_tensor(offsets[gap] + ranks, device=source.device)
        sums = np.zeros((len(held), 4), dtype=np.float64)
        worst = np.zeros(len(held), dtype=np.float64)
        for start in range(0, len(indices), batch_size):
            batch = slice(start, start + batch_size)
            posterior, score = forecast(views[batch], indices[batch])
            clip = local_clip[batch]
            values = metrics(posterior.cpu().numpy(), score.cpu().numpy(), label[clip], sentiment[clip])
            np.add.at(sums, clip, values)
            np.maximum.at(worst, clip, values[:, :2].sum(-1))
        means = sums / counts[gap, held, None]
        results.append(np.column_stack((means, worst)))
    return np.stack(results)


def compact_forecast(model, source, saved_anchor, gain):
    contexts, priors = saved_anchor
    independent = model.intensity_head is not None

    def forecast(view, clips):
        prior = priors[view, clips]
        result = source.fusion(model, contexts[view, clips], prior, view, clips, clips,
                               return_intensity=independent)
        logits, shift = result if independent else (result, sentiment_shift(result, prior))
        score = (intensity_anchor(prior, source.class_centers) + gain * shift).tanh() * source.intensity_scale
        return logits.softmax(-1), score

    return forecast


def by_video(original, score, exact, label, sentiment, membership, n_videos):
    fixed = metrics(original, score, label, sentiment)
    fixed_loss = fixed[..., :2].sum(-1)
    # Float32 softmax rows can differ from unit mass at their last digit.
    np.testing.assert_allclose(fixed_loss, ranked_probability_score(original.astype(np.float64), label),
                               atol=2e-7, rtol=0)
    if np.any(exact[..., -1] + 2e-5 < fixed_loss):
        raise ValueError('Every-location worst RPS must bound the loss at the original start.')
    fixed = np.concatenate((fixed, fixed_loss[..., None]), axis=-1)
    return np.stack([np.stack([np.stack([np.bincount(membership, weights=values, minlength=n_videos)
                                         for values in gap.T])
                               for gap in condition])
                     for condition in (fixed, exact)])


def collect(runs, cohort, seeds, device):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    candidate = RUN_NAME[cohort]
    paths = run_paths(runs, seeds, cohort)
    originals = paths[candidate]
    metadata = load_metadata(originals)
    verify_training(candidate, metadata, seeds, cohort)
    settings = metadata[0]['config']
    archive = load_references(runs, cohort, seeds, originals)
    clips, split, original, source = source_for(settings, device)
    held = split['eval']
    label, sentiment = clips.label[held], clips.sentiment[held]
    for key, values in (('ids', clips.ids[held]), ('groups', clips.groups[held]), ('y', label),
                        ('sentiment', sentiment), ('masks', original[:, :, held])):
        np.testing.assert_array_equal(archive[key], values, err_msg=key)
    videos, membership = np.unique(clips.groups[held], return_inverse=True)
    count_by_video = np.bincount(membership)
    expanded, offsets, counts = placement_views(original)
    source.views = torch.as_tensor(expanded, device=device)
    gpu_offsets, gpu_counts = (torch.as_tensor(x, device=device) for x in (offsets, counts))
    banks = np.full((len(expanded), len(clips.ids), 1), -1, dtype=np.int32)
    stats = {name: [] for name in (*[row[0] for row in VARIANTS], *BASELINES)}
    selections = []

    def record(name, probability, score, forecast, batch):
        exact = all_location_metrics(forecast, source, held, label, sentiment, offsets, counts, batch)
        stats[name].append(by_video(probability, score, exact, label, sentiment, membership, len(videos)))

    for row, (seed, run, meta) in enumerate(zip(seeds, originals, metadata, strict=True)):
        cfg = meta['config']
        if any(cfg[key] != value for key, value in settings.items() if key not in ('output', 'seed')):
            raise ValueError(f'{run.name} changed its fitting setup.')
        if cfg['score_anchor'] != 'class-mean' or not cfg['calibrate_score_gain']:
            raise ValueError('Use the original calibrated compact heads.')
        args = argparse.Namespace(**cfg)
        fixed_anchor = anchor_for(run, source, split, cfg, archive, row)
        draw_anchor = sampler(gpu_offsets, gpu_counts, seed + 4801, True)
        new_anchor, anchor_choice = fit_anchor(args, source, split, clips.label,
                                               training_view_sampler=draw_anchor)
        varied_anchor = anchor_outputs(new_anchor, source, cfg['batch'])
        run_choices = {'anchor': anchor_choice}
        for name, arm, resampled_anchor, resampled_fusion in VARIANTS:
            saved = varied_anchor if resampled_anchor else fixed_anchor
            if resampled_anchor or resampled_fusion:
                draw = sampler(gpu_offsets, gpu_counts, seed + 4802, resampled_fusion)
                model, selection = fit_fusion(args, arm, source, split, saved, clips.label, banks,
                                               training_view_sampler=draw)
            else:
                model = AudiovisualResidual(0, source.vision.shape[-1], cfg['width'],
                                            visual_encoding=cfg['visual_encoding'],
                                            independent_intensity=arm == 'independent_intensity')
                model = load_weights(model, run / f'{arm}.pt', device)
                selection = meta['selection'][arm]
            if selection['parameters'] != meta['selection'][arm]['parameters']:
                raise ValueError('Resampling must preserve the fitted head size.')
            probability, score, gain = calibrated_head_forecasts(model, source, clips, split, saved, cfg['batch'])
            if not resampled_anchor and not resampled_fusion:
                ref = 'shared' if arm == 'intensity' else 'independent'
                np.testing.assert_allclose(probability, archive[ref][row], rtol=3e-5, atol=5e-6)
                np.testing.assert_allclose(score, archive[ref + '_score'][row], rtol=3e-5, atol=5e-6)
            record(name, probability, score, compact_forecast(model, source, saved, gain), cfg['batch'] * 3)
            run_choices[name] = dict(selection, score_gain=gain)
            print(f'{cohort} seed {seed}: {name} fitted or replayed and evaluated at every start', flush=True)
        selections.append(run_choices)

    references = {'rmnet_uniform': ('rmnet_core', [runs / f'{cohort}_rmnet_uniform_seed{s}' for s in seeds]),
                  'lnln_joint': ('lnln_joint', paths['lnln_joint'])}
    for name, (reference, ref_paths) in references.items():
        meta = load_metadata(ref_paths)
        verify_training(reference, meta, seeds, cohort)
        archived = load_matched(ref_paths, archive)
        for seed, run, row, packet in zip(seeds, ref_paths, meta, archived, strict=True):
            cfg = row['config']
            if name == 'rmnet_uniform' and cfg.get('rmnet_class_weighting') != 'uniform':
                raise ValueError('Use the uniform-class-weight RM-Net refits.')
            model_forecast = predictor(reference, run, row, Path(cfg['upstream']), source)

            def forecast(view, indices):
                return model_forecast(model_inputs(source, view, indices))

            with torch.inference_mode():
                for gap in GAPS:
                    batches = [torch.as_tensor(held[start:start+cfg['batch']], device=device)
                               for start in range(0, len(held), cfg['batch'])]
                    outputs = [forecast(torch.full_like(batch, gap), batch) for batch in batches]
                    p, s = (torch.cat(values).cpu().numpy() for values in zip(*outputs, strict=True))
                    np.testing.assert_allclose(p, packet['posterior'][gap], rtol=3e-5, atol=5e-6, err_msg=run.name)
                    np.testing.assert_allclose(s, packet['score'][gap], rtol=3e-5, atol=5e-6, err_msg=run.name)
            record(name, packet['posterior'][1:3], packet['score'][1:3], forecast, cfg['batch'] * 2)
            print(f'{cohort} seed {seed}: {name} replayed and evaluated at every start', flush=True)
    return dict(**{name: np.stack(total) for name, total in stats.items()}, n_clips_by_video=count_by_video,
                videos=videos, seeds=np.asarray(seeds), cohort=np.asarray(cohort),
                mean_starts=counts[list(GAPS)][:, held].mean(1),
                starts_total=counts[list(GAPS)][:, held].sum(1),
                selections=np.asarray(json.dumps(selections)))


def summarize(data, resamples):
    counts = data['n_clips_by_video']
    seeds = data['seeds'].tolist()
    methods, rps, worst = {}, {}, {}
    for name in (*[row[0] for row in VARIANTS], *BASELINES):
        total = data[name]
        loss = total[:, :, :, :2, :].sum(axis=3)
        rps[name] = loss.mean(2)
        worst[name] = total[:, :, :, -1, :].mean(2)
        descriptions = {}
        for view, protocol in enumerate(('fixed_start', 'all_starts')):
            by_gap = total[:, view].sum(-1).mean(0) / counts.sum()
            fields = {key: float(value) for key, value in zip(FIELDS, by_gap.mean(0), strict=True)}
            descriptions[protocol] = dict(**fields, rps=fields['signed_rps'] + fields['neutral_rps'],
                                          rps_by_width=by_gap[:, :2].sum(1).tolist())
        methods[name] = descriptions

    def effect(diff, family):
        tail = .025 / family
        by_seed = diff.sum(1) / counts.sum()
        return dict(reduction=float(by_seed.mean()), by_seed=by_seed.tolist(),
                    ci95_familywise=grouped_seed_video_interval(diff, counts, resamples,
                                                               quantiles=(tail, 1-tail)))

    static, anchor, fusion, both = [rps[name][:, 1] for name in
                                    ('fixed_shared', 'anchor_resampled', 'fusion_resampled', 'both_shared')]
    training_gain = static - both
    fixed_start_gain = rps['fixed_shared'][:, 0] - rps['both_shared'][:, 0]
    share_fixed = rps['fixed_independent'][:, 1] - static
    share_both = rps['both_independent'][:, 1] - both
    return dict(
        cohort=data['cohort'].item(), seeds=seeds, n_clips=int(counts.sum()), n_videos=len(counts),
        mean_starts=data['mean_starts'].tolist(), starts_total=data['starts_total'].tolist(),
        resamples=resamples, methods=methods,
        resampling=effect(training_gain, 2),
        resampling_fixed_start=effect(fixed_start_gain, 2),
        start_protocol_interaction=effect(training_gain-fixed_start_gain, 2),
        worst_start=effect(worst['fixed_shared'][:, 1]-worst['both_shared'][:, 1], 2),
        components=dict(anchor=effect(((static-anchor)+(fusion-both))/2, 6),
                        fusion=effect(((static-fusion)+(anchor-both))/2, 6),
                        interaction=effect((anchor-both)-(static-fusion), 6)),
        sharing=dict(fixed=effect(share_fixed, 6), resampled=effect(share_both, 6),
                     change=effect(share_both-share_fixed, 6)),
        benchmarks={f'{scheme}_vs_{reference}': effect(rps[reference][:, 1]-rps[scheme][:, 1], 8)
                    for scheme in ('fixed_shared', 'both_shared') for reference in BASELINES},
        selections=json.loads(data['selections'].item()))


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
        parser.error('Use positive resamples and add --cohort when reading checkpoints.')
    if args.runs:
        seeds = SEEDS[:1] if args.first_seed else SEEDS
        evidence = collect(args.runs, args.cohort, seeds, torch.device(args.device))
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            evidence = dict(packet)
    result = summarize(evidence, args.resamples)
    if args.write_evidence:
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as pending:
            np.savez_compressed(pending, **evidence)
        temporary.replace(args.write_evidence)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name in ('resampling', 'resampling_fixed_start', 'worst_start'):
        comparison = result[name]
        lower, upper = comparison['ci95_familywise']
        print(f'{name}: {comparison["reduction"]:+.6f} [{lower:+.6f}, {upper:+.6f}]', flush=True)


if __name__ == '__main__':
    main()
