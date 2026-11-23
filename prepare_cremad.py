"""Build a CREMA-D packet with audience-derived polarity and disjoint actors."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import cv2
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from encode_omg import FRAME_BINS, VISION


REVISION = '1658cd342dff90010aa843eaeebd53610a08b1dc'
NEUTRAL_MARGIN = .10
POLARITY = {'A': -1, 'D': -1, 'F': -1, 'H': 1, 'N': 0, 'S': -1}
DOCUMENTED_INVALID = {'1076_MTI_NEU_XX', '1076_MTI_SAD_XX', '1064_TIE_SAD_XX', '1064_IEO_DIS_MD'}


def audience_ratings(path):
    ratings = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['queryType'] != '3':
                continue
            level = float(row['respLevel'])
            if not 0 <= level <= 100:
                raise ValueError(f'Invalid rating for {row["clipName"]}.')
            ratings[row['clipName']].append(POLARITY[row['respEmo']] * level / 100)
    if len(ratings) != 7_442 or not DOCUMENTED_INVALID.issubset(ratings):
        raise ValueError('Unexpected coverage of the released audiovisual responses.')
    return {clip: values for clip, values in ratings.items() if clip not in DOCUMENTED_INVALID}


def audience_scores(path):
    return {clip: (np.mean(values), len(values)) for clip, values in audience_ratings(path).items()}


def actor_partition(actors):
    distinct = np.unique(actors)
    if len(distinct) != 91:
        raise ValueError('Expected all 91 actors.')
    ordered = np.random.default_rng(1729).permutation(distinct)
    return {'train': np.isin(actors, ordered[:48]), 'valid': np.isin(actors, ordered[48:60]),
            'test': np.isin(actors, ordered[60:])}


def frames_at_centers(path):
    video = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        video.release()
    if not frames:
        return []
    indices = (2 * np.arange(FRAME_BINS) + 1) * len(frames) // (2 * FRAME_BINS)
    return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in indices]


@torch.inference_mode()
def encode_frames(decoded, model, processor, batch, device):
    result = np.zeros((len(decoded), FRAME_BINS, model.config.hidden_size), dtype=np.float16)
    images = [frame for clip in decoded for frame in clip]
    rows = [i for i, clip in enumerate(decoded) if clip]
    if not images:
        return result
    embeds = []
    for i in range(0, len(images), batch):
        pixels = processor(images=images[i:i + batch], return_tensors='pt')['pixel_values']
        features = model(pixel_values=pixels.to(device=device, dtype=model.dtype)).pooler_output
        embeds.append(features.cpu().numpy().astype(np.float16, copy=False))
    result[rows] = np.concatenate(embeds).reshape(len(rows), FRAME_BINS, -1)
    if not np.isfinite(result).all():
        raise ValueError('The vision encoder produced a non-finite feature.')
    return result


def encode_videos(media, clips, args):
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    kwargs = dict(revision=VISION[1], cache_dir=args.cache_dir)
    processor = CLIPImageProcessor.from_pretrained(VISION[0], **kwargs)
    model = CLIPVisionModel.from_pretrained(VISION[0], weights_only=True, dtype=dtype, **kwargs).to(device).eval()
    parts = args.output.parent / f'{args.output.stem}_parts'
    parts.mkdir(parents=True, exist_ok=True)
    features = np.empty((len(clips), FRAME_BINS, model.config.hidden_size), dtype=np.float16)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(clips), 32):
            chunk = clips[start:start + 32]
            stop = start + len(chunk)
            path = parts / f'{start:05d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['clips'], chunk)
                    features[start:stop] = saved['vision']
            else:
                decoded = list(pool.map(frames_at_centers, [media / f'{clip}.flv' for clip in chunk]))
                features[start:stop] = encode_frames(decoded, model, processor, args.batch, device)
                temporary = path.with_suffix('.npz.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, clips=chunk, vision=features[start:stop])
                temporary.replace(path)
            if stop % 256 == 0 or stop == len(clips):
                print(f'Encoded {stop:,}/{len(clips):,} clips.', flush=True)
    return features


def prepare(args):
    revision = subprocess.check_output(['git', '-C', str(args.release), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected the CREMA-D revision {REVISION}.')
    scores = audience_scores(args.release / 'finishedResponses.csv')
    clips = sorted(scores)
    media = args.release / 'VideoFlash'
    unavailable = [clip for clip in clips if not (media / f'{clip}.flv').exists()
                   or (media / f'{clip}.flv').stat().st_size <= 256]
    if unavailable:
        raise ValueError(f'{len(unavailable)} original videos have not been downloaded.')
    actors = np.asarray([clip.split('_')[0] for clip in clips])
    roles = actor_partition(actors)
    statements = [clip.split('_')[1] for clip in clips]
    vocabulary, statement_index = np.unique(statements, return_inverse=True)
    if len(vocabulary) != 12:
        raise ValueError('Expected the 12 repeated statements.')
    sentiments = np.asarray([scores[clip][0] for clip in clips], dtype=np.float32)
    votes = np.asarray([scores[clip][1] for clip in clips])
    vision = encode_videos(media, clips, args)
    empty = ~np.any(vision != 0, axis=(1, 2))
    tokens = np.zeros((len(clips), 3, 1), dtype=np.int64)
    tokens[:, 0, 0] = statement_index + 1
    tokens[:, 1, 0] = 1
    packet = dict(id=np.asarray([f'actor{actor}$_${clip}' for actor, clip in zip(actors, clips, strict=True)]),
                  regression_labels=sentiments, text=np.eye(len(vocabulary), dtype=np.float32)[statement_index, None],
                  text_bert=tokens, audio=np.zeros((len(clips), 1, 1), np.float32), vision=vision)
    temporary = args.output.with_suffix('.pkl.part')
    with temporary.open('wb') as stream:
        pickle.dump({role: {key: values[mask] for key, values in packet.items()}
                     for role, mask in roles.items()}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)

    label = np.where(np.abs(sentiments) <= NEUTRAL_MARGIN, 1, np.sign(sentiments) + 1).astype(int)
    by_role = {role: dict(clips=int(mask.sum()), actors=np.unique(actors[mask]).tolist(),
                          audiovisual_ratings=int(votes[mask].sum()), no_decoded_frames=int((empty & mask).sum()),
                          class_counts=np.bincount(label[mask], minlength=3).tolist())
               for role, mask in roles.items()}
    with args.output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    summary = dict(revision=revision, vision_revision=VISION[1], neutral_margin=NEUTRAL_MARGIN,
                   documented_invalid=sorted(DOCUMENTED_INVALID), decode_missing=np.asarray(clips)[empty].tolist(),
                   sha256=digest, splits=by_role)
    args.summary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for name, role in by_role.items():
        print(f'{name}: {role["clips"]} clips, {len(role["actors"])} actors, class counts {role["class_counts"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=96)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if min(args.batch, args.workers) < 1:
        parser.error('Use positive batch and worker counts.')
    if args.output.exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    prepare(args)


if __name__ == '__main__':
    main()
