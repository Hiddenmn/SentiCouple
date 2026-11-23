"""Replay all visual-gap starts on two held-out CREMA-D examples."""

import argparse
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from gap_location_effects import GAPS, RATES, placement_views
from networks import AudiovisualResidual, TextAnchor, intensity_anchor, sentiment_shift
from prepare_cremad import frames_at_centers
from published_inference import load_weights
from score_anchor_effects import SEEDS
from text_weighted_loss import source_for


def representative_clips(forecasts):
    reference = forecasts[0]
    masks = reference['masks'][:, 1]
    eligible = masks[0].all(1) & masks[1, :, :4].all(1) & masks[1, :, -4:].all(1)
    y = reference['y']
    gains = np.mean([ranked_probability_score(run['anchor'][1:3], y)
                     - ranked_probability_score(run['intensity'][1:3], y)
                     for run in forecasts], axis=(0, 1))
    chosen = []
    for sign in (1, -1):
        candidates = np.flatnonzero(eligible & (sign * gains > 0))
        if not len(candidates):
            raise ValueError('The original gap evaluation needs an eligible example on each side of zero.')
        chosen.append(candidates[np.abs(gains[candidates] - np.median(gains[candidates])).argmin()])
        eligible &= reference['groups'] != reference['groups'][chosen[-1]]
    return np.asarray(chosen), gains[chosen]


@torch.inference_mode()
def replay(source, config, path, selection, view, cases):
    anchor = TextAnchor(source.text.shape[-1], config['bins'], config['width'], visual_only=True)
    model = AudiovisualResidual(0, source.vision.shape[-1], config['width'],
                                visual_encoding=config['visual_encoding'])
    load_weights(anchor, path / 'anchor.pt', source.device)
    load_weights(model, path / 'intensity.pt', source.device)
    context, prior = source.anchor(anchor, view, cases)
    logits = source.fusion(model, context, prior, view, cases, cases)
    correction = sentiment_shift(logits, prior)
    base = intensity_anchor(prior, source.class_centers)
    gain = selection['intensity']['score_gain']
    results = (prior.softmax(-1), logits.softmax(-1), base.tanh() * source.intensity_scale,
               (base + gain * correction).tanh() * source.intensity_scale)
    return {name: value.cpu().numpy().reshape((len(source.views), 2, *value.shape[1:]))
            for name, value in zip(('anchor', 'intensity', 'anchor_score', 'intensity_score'),
                                   results, strict=True)}


def collect(runs, features, release, device):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    paths = [runs / f'cremad_heads_seed{seed}' for seed in SEEDS]
    meta = load_metadata(paths)
    settings = meta[0]['config']
    for seed, config in zip(SEEDS, (row['config'] for row in meta), strict=True):
        if config['seed'] != seed or any(config[key] != value for key, value in settings.items()
                                         if key not in ('seed', 'output')):
            raise ValueError('Expected the same CREMA-D configuration across the ten seeds.')
    reference = read_predictions(paths[0])
    packets = load_matched(paths, reference)
    selected, signed_gain = representative_clips(packets)
    clips, split, original, source = source_for(dict(settings, data=str(features)), device)
    held = split['eval']
    cases = held[selected]
    for key, data in [('ids', clips.ids[held]), ('y', clips.label[held]),
                      ('sentiment', clips.sentiment[held]), ('masks', original[:, :, held])]:
        np.testing.assert_array_equal(reference[key], data, err_msg=key)
    views, offsets, counts = placement_views(original[:, :, cases])
    full_views = np.concatenate((original, np.zeros((len(views) - len(original), *original.shape[1:]),
                                                    dtype=bool)))
    full_views[len(original):, :, cases] = views[len(original):]
    source.views = torch.as_tensor(full_views, device=device)
    flat_views = torch.arange(len(views), device=device).repeat_interleave(len(cases))
    flat_cases = torch.as_tensor(cases, device=device).repeat(len(views))
    results = []
    for path, metadata, packet in zip(paths, meta, packets, strict=True):
        prediction = replay(source, settings, path, metadata['selection'], flat_views, flat_cases)
        for key, values in prediction.items():
            np.testing.assert_allclose(values[:4], packet[key][:, selected], atol=6e-6, rtol=3e-5,
                                       err_msg=f'{path.name} {key}')
        results.append(prediction)
    widths = np.rint(24 * np.asarray(RATES)).astype(int)
    starts = counts[list(GAPS), 0]
    np.testing.assert_array_equal(counts[list(GAPS)], np.broadcast_to(starts[:, None], (2, 2)))
    frame_slots = np.arange(2, 24, 6)
    frames = []
    for clip_id in clips.ids[cases]:
        name = clip_id.split('$_$', 1)[1]
        native_frames = frames_at_centers(release / 'VideoFlash' / f'{name}.flv')
        if len(native_frames) != 24:
            raise ValueError(f'Could not read the original frames for {name}.')
        frames.append([native_frames[slot] for slot in frame_slots])
    return dict(ids=clips.ids[cases], y=clips.label[cases], sentiment=clips.sentiment[cases],
                seeds=np.asarray(SEEDS), original_mean_gain=signed_gain, masks=views[:, 1],
                offsets=offsets[list(GAPS)], counts=starts, widths=widths,
                frames=np.asarray(frames), frame_slots=frame_slots, visual=clips.vision[cases, :, :6],
                **{key: np.stack([result[key] for result in results]) for key in results[0]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    evidence = collect(args.runs, args.features, args.release, torch.device(args.device))
    np.savez_compressed(args.output, **evidence)
    print(evidence['ids'].tolist(), evidence['original_mean_gain'].tolist(), flush=True)


if __name__ == '__main__':
    main()
