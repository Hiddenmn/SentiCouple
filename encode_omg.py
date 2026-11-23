"""Extract fixed text and frame features from the downloaded OMG-Emotion sections."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel, CLIPImageProcessor, CLIPVisionModel


TEXT = ('google-bert/bert-base-uncased', '86b5e0934494bd15c9632b12f734a8a67f723594')
VISION = ('openai/clip-vit-base-patch32', '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
FRAME_BINS = 24
TEXT_TOKENS = 64
NEUTRAL_MARGIN = .10


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_npz(path, **values):
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('wb') as stream:
        np.savez(stream, **values)
    temporary.replace(path)


def matched_sections(manifest, ids):
    rows = {clip: index for index, clip in enumerate(ids)}
    if len(rows) != len(ids):
        raise ValueError('Original clip IDs must be unique.')
    matched_rows = set()
    sections = []
    for section in manifest['windows']:
        clips = [clip for clip in section['clips'] if clip['id'] in rows]
        if not clips:
            continue
        indices = [rows[clip['id']] for clip in clips]
        if len(set(indices)) != len(indices) or matched_rows.intersection(indices):
            raise ValueError('Each matched clip must occur exactly once in the media manifest.')
        matched_rows.update(indices)
        sections.append((dict(section, clips=clips), indices))
    if len(matched_rows) != len(ids):
        raise ValueError('The media manifest must cover every original clip.')
    return sections


def frame_requests(clips, start, fps, frames, allow_partial=False):
    requested = defaultdict(list)
    for index, clip in enumerate(clips):
        centers = clip['start'] - start + (
            (np.arange(FRAME_BINS) + .5) / FRAME_BINS * (clip['end'] - clip['start']))
        for slot, frame in enumerate(np.rint(centers * fps).astype(int).clip(0, frames - 1)):
            if allow_partial and not 0 <= centers[slot] < frames / fps:
                continue
            requested[int(frame)].append((index, slot))
    return requested


def read_requested_frames(video, requested, fps):
    """Decode the same frame numbers for all downstream image representations."""
    position = 0
    for frame in sorted(requested):
        if frame - position > 2 * fps:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            position = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        while position <= frame and video.grab():
            position += 1
        if position == frame + 1:
            ok, image = video.retrieve()
            if ok:
                yield frame, image


@torch.inference_mode()
def encode_window(window, media, vision, image_processor, text, tokenizer, batch, device,
                  allow_partial=False):
    full_clip = window.get('full_clip', False)
    clips = window['clips']
    if full_clip and (len(clips) != 1 or window['start'] != 0):
        raise ValueError('Whole-video encoding requires one utterance starting at zero.')
    video = cv2.VideoCapture(str(media / window['media']))
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frames < 1:
            if full_clip:
                visual = np.zeros((1, FRAME_BINS, vision.config.hidden_size), np.float16)
                return encode_utterances(clips, visual, text, tokenizer, batch, device)
            return None
        duration = frames / fps
        if full_clip:
            clips = [{**clips[0], 'start': 0., 'end': duration}]
        elif not allow_partial:
            clips = [clip for clip in clips if clip['end'] - window['start'] <= duration + .04]
        if not clips:
            return None
        requested = frame_requests(clips, window['start'], fps, frames, allow_partial=allow_partial)

        visual = np.zeros((len(clips), FRAME_BINS, vision.config.hidden_size), np.float16)
        images, frame_indices = [], []
        for frame, image in read_requested_frames(video, requested, fps):
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame_indices.append(frame)
            if len(images) == batch:
                save_frames(vision, image_processor, images, frame_indices, requested, visual, device)
                images.clear()
                frame_indices.clear()
        if images:
            save_frames(vision, image_processor, images, frame_indices, requested, visual, device)
    finally:
        video.release()

    if full_clip:
        visual[~np.isfinite(visual).all(axis=-1)] = 0
        return encode_utterances(clips, visual, text, tokenizer, batch, device)
    valid = np.isfinite(visual).all(axis=(1, 2)) & np.any(visual != 0, axis=(1, 2))
    clips = [clip for clip, keep in zip(clips, valid) if keep]
    if not clips:
        return None
    return encode_utterances(clips, visual[valid], text, tokenizer, batch, device)


def encode_utterances(clips, visual, text, tokenizer, batch, device):
    embeddings, tokens = [], []
    for start in range(0, len(clips), batch):
        transcripts = [clip['transcript'] for clip in clips[start:start + batch]]
        inputs = tokenizer(transcripts, padding='max_length', truncation=True, max_length=TEXT_TOKENS,
                           return_tensors='pt').to(device)
        embeddings.append(text(**inputs).last_hidden_state.cpu().numpy().astype(np.float16, copy=False))
        tokens.append(torch.stack([inputs[key] for key in ('input_ids', 'attention_mask', 'token_type_ids')], dim=1)
                      .cpu().numpy())
    return dict(id=np.asarray([clip['id'] for clip in clips]),
                regression_labels=np.asarray([clip['valence'] for clip in clips], np.float32),
                text=np.concatenate(embeddings), text_bert=np.concatenate(tokens),
                audio=np.zeros((len(clips), 1, 1), np.float32), vision=visual)


def save_frames(model, processor, images, frame_indices, requested, visual, device):
    pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device=device, dtype=model.dtype)
    embedding = model(pixel_values=pixels).pooler_output.cpu().numpy().astype(np.float16, copy=False)
    for frame, feature in zip(frame_indices, embedding):
        for row, slot in requested[frame]:
            visual[row, slot] = feature


def load_text_and_vision(cache_dir, device):
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    text_args = dict(revision=TEXT[1], cache_dir=cache_dir)
    vision_args = dict(revision=VISION[1], cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(TEXT[0], **text_args)
    text = BertModel.from_pretrained(TEXT[0], use_safetensors=True, dtype=dtype, **text_args).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(VISION[0], **vision_args)
    vision = CLIPVisionModel.from_pretrained(VISION[0], weights_only=True, dtype=dtype, **vision_args).to(device).eval()
    return vision, processor, text, tokenizer


def encode_sections(pending, args, device, allow_partial=False):
    torch.set_num_threads(4)
    vision, processor, text, tokenizer = load_text_and_vision(args.cache_dir, device)
    hardware = torch.cuda.get_device_name(device) if device.type == 'cuda' else device
    print(f'Encoding {len(pending)} sections on {hardware}.', flush=True)
    for index, (window, path) in enumerate(pending, 1):
        result = encode_window(window, args.prepared / 'media', vision, processor, text, tokenizer, args.batch,
                               device, allow_partial=allow_partial)
        if result is not None:
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream, **result)
            temporary.replace(path)
        if index % 20 == 0 or index == len(pending):
            print(f'Processed {index}/{len(pending)} sections.', flush=True)


def cohort_summary(manifest, availability, data, dataset):
    roles = {}
    used_sources = set()
    for role, packet in data.items():
        windows = [window for window in manifest['windows'] if window['role'] == role]
        planned = {clip['id']: clip['valence'] for window in windows for clip in window['clips']}
        ids = packet['id'].tolist()
        values = packet['regression_labels']
        if len(set(ids)) != len(ids) or not set(ids).issubset(planned):
            raise ValueError(f'Unexpected or duplicate utterance in {role}.')
        np.testing.assert_array_equal(values, np.array([planned[key] for key in ids], dtype=np.float32))
        sources = {key.split('$_$', 1)[0] for key in ids}
        if sources & used_sources:
            raise ValueError('A source video occurs in more than one partition.')
        used_sources.update(sources)
        roles[role] = dict(
            candidate_clips=len(planned), candidate_videos=len({window['source'] for window in windows}),
            clips=len(ids), videos=len(sources),
            class_counts=[int((values < -NEUTRAL_MARGIN).sum()), int((np.abs(values) <= NEUTRAL_MARGIN).sum()),
                          int((values > NEUTRAL_MARGIN).sum())],
            section_status=dict(Counter(availability.get(window['media'], 'unattempted') for window in windows)))
    with dataset.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(revision=manifest['revision'], sha256=digest, neutral_margin=NEUTRAL_MARGIN,
                class_order=['negative', 'neutral', 'positive'], splits=roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path)
    destination.add_argument('--cache-only', action='store_true')
    parser.add_argument('--summary', type=Path, help='Write coverage and label counts for the extracted dataset.')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, help='Encode a small media sample to verify frame extraction.')
    args = parser.parse_args()
    if args.batch < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size and sample size must be positive.')
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    if args.summary is not None and args.output is None:
        parser.error('--summary requires --output.')
    if args.summary is not None and args.summary.resolve() == args.output.resolve():
        parser.error('Use different paths for the dataset and its summary.')
    manifest = json.loads((args.prepared / 'manifest.json').read_text(encoding='utf-8'))
    availability = json.loads((args.prepared / 'availability.json').read_text(encoding='utf-8'))
    windows = [window for window in manifest['windows']
               if availability.get(window['media']) in ('available', 'partial_section')][:args.limit]
    if not windows:
        raise ValueError('No complete media is available for extraction.')
    roles = ('train', 'valid', 'test')
    if not args.cache_only and args.limit is None:
        if any(window['media'] not in availability for window in manifest['windows']):
            raise ValueError('Complete the media acquisition before assembling the full dataset.')
        if {window['role'] for window in windows} != set(roles):
            raise ValueError('The full dataset requires available sections from all three splits.')
    device = torch.device(args.device)
    cache = args.prepared / f'frame_features_{device.type}_batch{args.batch}'
    cache.mkdir(exist_ok=True)
    sections = [(window, cache / (Path(window['media']).stem + '.npz')) for window in windows]
    pending = [(window, path) for window, path in sections if not path.is_file()]
    if pending:
        encode_sections(pending, args, device)
    if args.cache_only:
        print(f'Cached {sum(1 for _ in cache.glob("*.npz"))} sections.', flush=True)
        return
    data = {}
    for role in roles:
        parts = []
        for window, path in sections:
            if window['role'] == role and path.is_file():
                with np.load(path, allow_pickle=False) as packet:
                    parts.append(dict(packet))
        if parts:
            data[role] = {key: np.concatenate([packet[key] for packet in parts]) for key in parts[0]}
    if not data or (args.limit is None and set(data) != set(roles)):
        raise ValueError('Feature extraction did not retain clips in all requested splits.')
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if args.summary is not None:
        report = cohort_summary(manifest, availability, data, temporary)
    temporary.replace(args.output)
    if args.summary is not None:
        args.summary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({role: len(packet['id']) for role, packet in data.items()}, flush=True)


if __name__ == '__main__':
    main()
