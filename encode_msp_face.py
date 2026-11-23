"""Build matched visual and transcribed text inputs for the MSP-Face study."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle

import av
import cv2
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from encode_omg import NEUTRAL_MARGIN, atomic_npz, encode_window, load_text_and_vision, sha256
from encode_omg_audio import RATE, decode_section
from prepare_msp_face import PARTITIONS, write_json


ASR = ('openai/whisper-small.en', 'e8727524f962ee844a7319d92be39ac1bd25655a')
MIN_AUDIO_COVERAGE = .90
AVAILABLE = ('available', 'partial_section')
FEATURE_ROLES = tuple(PARTITIONS.values())


def read_availability(prepared):
    status = json.loads((prepared / 'availability.json').read_text(encoding='utf-8'))
    retry = prepared / 'availability_retry.json'
    if retry.is_file():
        recovered = json.loads(retry.read_text(encoding='utf-8'))
        if any(status.get(name) not in ('download_error', 'timeout') for name in recovered):
            raise ValueError('The separate retry must cover only failed first attempts.')
        status.update(recovered)
    return status


@torch.inference_mode()
def transcripts(waveforms, processor, model, device, batch_size):
    results = []
    for start in range(0, len(waveforms), batch_size):
        inputs = processor(waveforms[start:start + batch_size], sampling_rate=RATE,
                           return_attention_mask=True, return_tensors='pt')
        inputs = inputs.to(device=device, dtype=model.dtype)
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=96)
        results.extend(processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return results


def encode_section(section, media, processor, asr, visual_models, device, batch_size):
    clips = section['clips']
    offset = section['start']
    edges = np.rint(np.array([(clip['start'] - offset, clip['end'] - offset)
                              for clip in clips]) * RATE).astype(int)
    try:
        audio, decoded, _ = decode_section(media / section['media'], int(edges[:, 1].max()))
    except (ValueError, av.FFmpegError):
        return None, dict(audio_decode=len(clips))

    waveforms = []
    candidates = []
    dropped = Counter()
    for clip, (left, right) in zip(clips, edges):
        if decoded[left:right].mean() < MIN_AUDIO_COVERAGE:
            dropped['short_audio'] += 1
            continue
        waveforms.append(audio[left:right])
        candidates.append(clip)
    text = transcripts(waveforms, processor, asr, device, batch_size)
    spoken_clips = []
    for clip, hypothesis in zip(candidates, text, strict=True):
        if any(character.isalnum() for character in hypothesis):
            spoken_clips.append({**clip, 'transcript': hypothesis.strip()})
        else:
            dropped['empty_transcript'] += 1
    if not spoken_clips:
        return None, dict(dropped)

    packet = encode_window(dict(section, clips=spoken_clips), media, *visual_models, batch_size, device,
                           allow_partial=True)
    if packet is None:
        dropped['incomplete_video'] += len(spoken_clips)
        return None, dict(dropped)
    complete = np.all(np.any(packet['vision'] != 0, axis=-1), axis=1)
    dropped['incomplete_video'] += len(spoken_clips) - int(complete.sum())
    return {key: value[complete] for key, value in packet.items()} if complete.any() else None, dict(dropped)


def encode_pending(windows, availability, prepared, cache_dir, batch_size, device):
    cache = prepared / 'encoded'
    cache.mkdir(exist_ok=True)
    pending = [window for window in windows if availability.get(window['media']) in AVAILABLE
               and not (cache / window['media']).with_suffix('.json').is_file()]
    if not pending:
        return
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    checkpoint = dict(revision=ASR[1], cache_dir=cache_dir, token=False, local_files_only=True)
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(ASR[0], **checkpoint,
                                                 dtype=dtype, use_safetensors=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(ASR[0], **checkpoint)
    visual_models = load_text_and_vision(cache_dir, device)
    hardware = torch.cuda.get_device_name(device) if device.type == 'cuda' else device
    print(f'Transcribing and encoding {len(pending)} sections on {hardware}.', flush=True)
    for index, window in enumerate(pending, 1):
        packet, dropped = encode_section(window, prepared / 'media', processor, asr, visual_models,
                                         device, batch_size)
        name = cache / window['media']
        if packet is not None:
            atomic_npz(name.with_suffix('.npz'), **packet)
        write_json(name.with_suffix('.json'), dict(clips=0 if packet is None else len(packet['id']),
                                                  dropped=dropped))
        if index % 10 == 0 or index == len(pending):
            print(f'Encoded {index}/{len(pending)} sections.', flush=True)


def participant_groups(rows):
    """Keep speakers connected by a shared source in the same resampling group."""
    by_video = defaultdict(set)
    for row in rows:
        by_video[row['source']].add(row['speaker'])
    adjacent = defaultdict(set)
    for speakers in by_video.values():
        for speaker in speakers:
            adjacent[speaker].update(speakers)
    component = {}
    for speaker in sorted(adjacent):
        pending = {speaker} - component.keys()
        while pending:
            peer = pending.pop()
            component[peer] = speaker
            pending.update(adjacent[peer] - component.keys())
    return np.asarray([component[row['speaker']] for row in rows])


def assemble_role(windows, availability, cache):
    packets = []
    dropped = Counter()
    statuses = Counter()
    original = {row['id']: row for window in windows for row in window['clips']}
    for window in windows:
        status = availability.get(window['media'], 'unattempted')
        statuses[status] += 1
        if status not in AVAILABLE:
            dropped[status] += len(window['clips'])
            continue
        name = cache / window['media']
        record = json.loads(name.with_suffix('.json').read_text(encoding='utf-8'))
        dropped.update(record['dropped'])
        if record['clips']:
            with np.load(name.with_suffix('.npz'), allow_pickle=False) as packet:
                if len(packet['id']) != record['clips']:
                    raise ValueError(f'Section clip count differs from its record for {name.name}.')
                packets.append(dict(packet))
    summary = dict(sections=dict(statuses), dropped=dict(dropped))
    if not packets:
        return None, [], dict(summary, clips=0, videos=0, speakers=0, participant_groups=0, classes=[0] * 3)
    features = {name: np.concatenate([packet[name] for packet in packets]) for name in packets[0]}
    ids = features['id'].tolist()
    rows = [original[clip] for clip in ids]
    if len(set(ids)) != len(ids) or len(ids) + sum(dropped.values()) != len(original):
        raise ValueError('Each eligible clip must be encoded once or have one exclusion reason.')
    scores = np.array([row['valence'] for row in rows], np.float32)
    np.testing.assert_array_equal(features['regression_labels'], scores)
    polarity = np.where(np.abs(scores) <= NEUTRAL_MARGIN, 0, np.sign(scores)).astype(int)
    groups = participant_groups(rows)
    features['participant_group'] = groups
    return features, rows, dict(summary, clips=len(rows), videos=len({row['source'] for row in rows}),
                               speakers=len({row['speaker'] for row in rows}),
                               participant_groups=len(set(groups)), classes=np.bincount(polarity + 1, minlength=3).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Write the full official-split feature packet.')
    parser.add_argument('--test-packet', type=Path, help='Write the test features for frozen transfer.')
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--roles', choices=FEATURE_ROLES, nargs='+', default=list(FEATURE_ROLES))
    parser.add_argument('--pilot', action='store_true', help='Encode only sections already fetched for the development pilot.')
    args = parser.parse_args()
    if args.pilot and (args.output or args.test_packet or 'test' in args.roles):
        parser.error('The pilot only encodes already fetched development sections.')
    if args.output and set(args.roles) != set(FEATURE_ROLES):
        parser.error('The full packet requires all three official splits.')
    if args.test_packet and 'test' not in args.roles:
        parser.error('The transfer packet requires the official test split.')
    manifest_path = args.prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    availability = read_availability(args.prepared)
    windows = [window for role in args.roles for window in manifest['windows'] if window['role'] == role]
    if args.pilot:
        windows = [window for window in windows if window['media'] in availability]
    if not windows or any(window['media'] not in availability for window in windows):
        raise ValueError('Finish downloading the requested splits before encoding them.')
    device = torch.device(args.device)
    encode_pending(windows, availability, args.prepared, args.cache_dir, args.batch, device)

    packets, splits = {}, {}
    speakers, sources = set(), set()
    for role in args.roles:
        chosen = [window for window in windows if window['role'] == role]
        packet, rows, splits[role] = assemble_role(chosen, availability, args.prepared / 'encoded')
        if packet is None and not args.pilot:
            raise ValueError(f'No usable clips in {role}.')
        planned_scores = np.asarray([clip['valence'] for window in chosen for clip in window['clips']])
        planned_labels = np.where(np.abs(planned_scores) <= NEUTRAL_MARGIN, 0,
                                  np.sign(planned_scores)).astype(int) + 1
        splits[role]['eligible_classes'] = np.bincount(planned_labels, minlength=3).tolist()
        packets[role] = packet
        new_speakers = {row['speaker'] for row in rows}
        new_sources = {row['source'] for row in rows}
        if speakers & new_speakers or sources & new_sources:
            raise ValueError('The encoded splits share a speaker or a source video.')
        speakers.update(new_speakers)
        sources.update(new_sources)
    if args.output:
        temporary = args.output.with_suffix('.part')
        with temporary.open('wb') as stream:
            pickle.dump(packets, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(args.output)
    if args.test_packet:
        atomic_npz(args.test_packet, **packets['test'])
    study = {key: manifest[key] for key in ('revision', 'filenames', 'original', 'eligible', 'excluded', 'exclusion_inputs')}
    write_json(args.summary, dict(study, manifest_sha256=sha256(manifest_path), asr=list(ASR),
                                 neutral_margin=NEUTRAL_MARGIN, batch=args.batch, device=str(device),
                                 torch=str(torch.__version__), splits=splits,
                                 packet_sha256=sha256(args.output) if args.output else None,
                                 test_sha256=sha256(args.test_packet) if args.test_packet else None,
                                 ids={role: packet['id'].tolist() if packet is not None else []
                                      for role, packet in packets.items()}))
    print(splits, flush=True)


if __name__ == '__main__':
    main()
