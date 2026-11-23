"""Encode disjoint audio bins on the timestamps of the archived OMG-Emotion clips."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle

import av
import numpy as np
import torch
from transformers import WavLMModel

from encode_omg import FRAME_BINS, atomic_npz, matched_sections, sha256


MODEL = ('microsoft/wavlm-base-plus', '4c66d4806a428f2e922ccfa1a962776e232d487b')
MODEL_HASH = '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'
RATE = 16_000
STRIDE = 320
MIN_SAMPLES = 400
MIN_COVERAGE = .90


def decode_section(path, length):
    waveform = np.zeros(length, dtype=np.float32)
    decoded = np.zeros(length, dtype=bool)
    with av.open(str(path)) as container:
        video = next(container.decode(video=0), None)
        if video is None or video.pts is None:
            raise ValueError(f'No video presentation origin in {path.name}.')
        video_origin = video.pts * video.time_base
        if not container.streams.audio:
            return waveform, decoded, None

    resampler = av.AudioResampler(format='flt', layout='mono', rate=RATE)
    first_audio = None

    def store(frames):
        for frame in frames:
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f'Missing presentation timestamp in {path.name}.')
            offset = round((frame.pts * frame.time_base - video_origin) * RATE)
            values = frame.to_ndarray().reshape(-1)
            start, stop = max(offset, 0), min(offset + len(values), length)
            if stop > start:
                waveform[start:stop] = values[start - offset:stop - offset]
                decoded[start:stop] = True

    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            if first_audio is None and frame.pts is not None:
                first_audio = float(frame.pts * frame.time_base - video_origin)
            store(resampler.resample(frame))
        store(resampler.resample(None))
    return waveform, decoded, first_audio


def prepare_section(section, media):
    offset = section['start']
    edges = [np.rint((np.linspace(clip['start'], clip['end'], FRAME_BINS + 1) - offset)
                     * RATE).astype(int) for clip in section['clips']]
    count = max(1, max(boundaries[-1] for boundaries in edges))
    waveform, decoded, audio_offset = decode_section(media / section['media'], count)
    coverage = np.zeros((len(edges), FRAME_BINS), dtype=np.float32)
    slices = []
    for row, boundaries in enumerate(edges):
        for slot, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            length = stop - start
            if length < MIN_SAMPLES or stop < 1:
                continue
            left = max(start, 0)
            observed = decoded[left:stop].sum()
            if observed < MIN_SAMPLES:
                continue
            coverage[row, slot] = observed / length
            if coverage[row, slot] >= MIN_COVERAGE:
                samples = waveform[left:stop]
                if left > start:
                    samples = np.pad(samples, (left - start, 0))
                slices.append((row, slot, samples))
    return coverage, slices, audio_offset


@torch.inference_mode()
def encode_chunk(decoded, model, device, batch):
    features = [np.zeros((*coverage.shape, model.config.hidden_size), dtype=np.float16)
                for coverage, _, _ in decoded]
    buckets = defaultdict(list)
    for section, (_, slices, _) in enumerate(decoded):
        for row, slot, samples in slices:
            length = ((len(samples) + STRIDE - 1) // STRIDE) * STRIDE
            buckets[length].append((section, row, slot, samples))
    for length, items in sorted(buckets.items()):
        step = min(batch, max(1, 640_000 // length))
        for start in range(0, len(items), step):
            examples = items[start:start + step]
            waveform = np.zeros((len(examples), length), dtype=np.float32)
            attention = np.zeros_like(waveform, dtype=bool)
            for i, (_, _, _, samples) in enumerate(examples):
                waveform[i, :len(samples)] = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
                attention[i, :len(samples)] = True
            audio = torch.as_tensor(waveform, device=device, dtype=model.dtype)
            mask = torch.as_tensor(attention, device=device)
            tokens = model(input_values=audio, attention_mask=mask.long()).last_hidden_state.float()
            valid = model._get_feature_vector_attention_mask(tokens.shape[1], mask.long())
            pooled = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            pooled = pooled.cpu().numpy().astype(np.float16)
            if not np.isfinite(pooled).all() or not np.any(pooled != 0, axis=1).all():
                raise ValueError('Expected finite, nonzero features for each observed audio interval.')
            for (section, row, slot, _), vector in zip(examples, pooled, strict=True):
                features[section][row, slot] = vector
    return features


def run(args):
    digest = sha256(args.model / 'pytorch_model.bin')
    if digest != MODEL_HASH:
        raise ValueError('Use the pinned WavLM Base Plus weights.')
    settings = dict(encoder=MODEL, encoder_sha256=digest, config_sha256=sha256(args.model / 'config.json'),
                    original_sha256=sha256(args.original), sample_rate=RATE, bins=FRAME_BINS,
                    minimum_samples=MIN_SAMPLES, minimum_coverage=MIN_COVERAGE,
                    av_version=av.__version__, torch_version=str(torch.__version__),
                    encoding_device=torch.device(args.device).type, batch=args.batch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / 'audio_sections'
    cache.mkdir(exist_ok=True)
    settings_path = args.output_dir / 'audio_settings.json'
    canonical = json.dumps(settings, indent=2)
    if settings_path.exists() and settings_path.read_text(encoding='utf-8') != canonical:
        raise ValueError('Use a separate directory for different audio encoding settings.')
    settings_path.write_text(canonical, encoding='utf-8')
    with args.original.open('rb') as stream:
        packets = pickle.load(stream)
    ids = np.concatenate([packet['id'] for packet in packets.values()])
    manifest = json.loads((args.prepared / 'manifest.json').read_text(encoding='utf-8'))
    sections = matched_sections(manifest, ids)[:args.limit]
    missing = [(section, rows) for section, rows in sections
               if not (cache / Path(section['media']).with_suffix('.npz').name).is_file()]
    device = torch.device(args.device)
    if missing:
        torch.set_num_threads(4)
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        model = WavLMModel.from_pretrained(args.model, local_files_only=True, weights_only=True,
                                         use_safetensors=False, dtype=dtype).to(device).eval()
        with ThreadPoolExecutor(max_workers=4) as decoder:
            for start in range(0, len(missing), 12):
                chunk = missing[start:start + 12]
                jobs = [decoder.submit(prepare_section, section, args.prepared / 'media')
                        for section, _ in chunk]
                decoded = [job.result() for job in jobs]
                vectors = encode_chunk(decoded, model, device, args.batch)
                for (section, rows), (coverage, _, offset), audio in zip(chunk, decoded, vectors, strict=True):
                    atomic_npz(cache / Path(section['media']).with_suffix('.npz').name,
                               ids=ids[rows], audio=audio, coverage=coverage,
                               audio_start=np.nan if offset is None else offset)
                print(f'Encoded {min(start + len(chunk), len(missing))}/{len(missing)} audio sections.', flush=True)

    coverage = np.zeros((len(ids), FRAME_BINS), dtype=np.float32)
    audio = None
    offsets = []
    for section, rows in sections:
        with np.load(cache / Path(section['media']).with_suffix('.npz').name, allow_pickle=False) as record:
            np.testing.assert_array_equal(record['ids'], ids[rows])
            if audio is None:
                audio = np.zeros((len(ids), FRAME_BINS, record['audio'].shape[-1]), dtype=np.float16)
            audio[rows], coverage[rows] = record['audio'], record['coverage']
            offsets.append(record['audio_start'].item())
    available = coverage >= MIN_COVERAGE
    np.testing.assert_array_equal(available, np.any(audio != 0, axis=-1))
    print(f'Observed {int(available.sum()):,} audio bins in {len(sections)} sections.', flush=True)
    if args.limit is not None:
        return

    rows = {clip: index for index, clip in enumerate(ids)}
    report = dict(settings, clips=len(ids), sections=len(sections), observed_bins=int(available.sum()),
                  sections_with_audio=int(np.isfinite(offsets).sum()),
                  audio_origin_offset_seconds=np.quantile(np.asarray(offsets)[np.isfinite(offsets)],
                                                          [0, .5, 1]).tolist(), splits={})
    for role, packet in packets.items():
        indices = [rows[clip] for clip in packet['id']]
        packet['audio'], packet['audio_bin_mask'] = audio[indices], available[indices]
        report['splits'][role] = dict(clips=len(indices), observed_bins=int(available[indices].sum()),
                                      possible_bins=len(indices) * FRAME_BINS)
    destination = args.output_dir / 'omg_audiovisual.pkl'
    temporary = destination.with_suffix('.pkl.part')
    with temporary.open('wb') as stream:
        pickle.dump(packets, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    atomic_npz(args.output_dir / 'audio_coverage.npz', ids=ids, coverage=coverage, available=available)
    (args.output_dir / 'audio_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False),
                                                       encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=96)
    parser.add_argument('--limit', type=int, help='Verify audio extraction on the first media sections.')
    args = parser.parse_args()
    if args.batch < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size and sample size must be positive.')
    run(args)
