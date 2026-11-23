"""Encode MELD sentiment utterances with the frozen source encoders."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
import tarfile
from types import SimpleNamespace

import numpy as np
import torch

from encode_omg import FRAME_BINS, TEXT, TEXT_TOKENS, VISION, encode_sections


ANNOTATIONS = {
    'test': ('8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888', 2610),
    'dev': ('2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c', 1109),
    'train': ('d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db', 9989),
}
POLARITY = {'negative': -1, 'neutral': 0, 'positive': 1}


def read_annotations(path, split='test'):
    digest, count = ANNOTATIONS[split]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Use the pinned MELD {split} sentiment CSV.')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    windows = []
    for row in rows:
        stem = f'dia{int(row["Dialogue_ID"])}_utt{int(row["Utterance_ID"])}'
        transcript = row['Utterance'].strip()
        if not transcript:
            raise ValueError(f'Missing transcript for {stem}.')
        clip = dict(id=f'{episode(row)}$_${stem}', transcript=transcript, valence=POLARITY[row['Sentiment']])
        windows.append(dict(media=f'{stem}.mp4', start=0., full_clip=True, clips=[clip]))
    if len(windows) != count or len({w['media'] for w in windows}) != count:
        raise ValueError(f'Expected the complete set of unique MELD {split} utterances.')
    return windows


def episode(row):
    return f's{int(row["Season"]):02d}e{int(row["Episode"]):02d}'


def window_episode(window):
    return window['clips'][0]['id'].split('$_$', 1)[0]


def isolated_windows(windows, split, test_annotations, dev_annotations):
    excluded = {window_episode(window) for window in read_annotations(test_annotations)}
    if split == 'train':
        excluded.update(window_episode(window) for window in read_annotations(dev_annotations, 'dev'))
    retained = [window for window in windows if window_episode(window) not in excluded]
    expected = (207, 15) if split == 'dev' else (1237, 32)
    if (len(retained), len({window_episode(window) for window in retained})) != expected:
        raise ValueError(f'Episode isolation gave the wrong cohort for {split}.')
    return retained


def extract_media(archive, directory, expected):
    directory.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive, 'r|*') as source:
        for member in source:
            name = PurePosixPath(member.name).name
            if name not in expected:
                continue
            if not member.isfile() or name in seen:
                raise ValueError(f'Expected one regular media file for {name}.')
            seen.add(name)
            destination = directory / name
            if destination.is_file() and destination.stat().st_size == member.size:
                continue
            temporary = destination.with_suffix('.mp4.part')
            with source.extractfile(member) as original, temporary.open('wb') as copied:
                copyfileobj(original, copied)
            temporary.replace(destination)
    return sorted(expected - seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', '--test-archive', required=True, type=Path)
    parser.add_argument('--split', choices=ANNOTATIONS, default='test')
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--test-annotations', type=Path)
    parser.add_argument('--dev-annotations', type=Path)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('Choose a positive feature batch size.')
    if args.split != 'test' and not args.test_annotations:
        parser.error('Specify the original test CSV for episode isolation.')
    if args.split == 'train' and not args.dev_annotations:
        parser.error('Specify the original development CSV for episode isolation.')
    if args.output.exists():
        raise FileExistsError(args.output)

    windows = read_annotations(args.annotations, args.split)
    if args.split != 'test':
        windows = isolated_windows(windows, args.split, args.test_annotations, args.dev_annotations)
    with args.archive.open('rb') as stream:
        archive_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    prepared = args.prepared / archive_digest[:16]
    missing_media = extract_media(args.archive, prepared / 'media', {w['media'] for w in windows})
    print(f'Matched {len(windows) - len(missing_media)}/{len(windows)} videos.', flush=True)
    device = torch.device(args.device)
    cache_name = f'features_{TEXT[1][:8]}_{VISION[1][:8]}_{FRAME_BINS}_{TEXT_TOKENS}_{device.type}_{args.batch}'
    cache = prepared / cache_name
    cache.mkdir(exist_ok=True)
    sections = [(window, cache / (Path(window['media']).stem + '.npz')) for window in windows]
    pending = [(window, path) for window, path in sections if not path.is_file()]
    if pending:
        encode_sections(pending, SimpleNamespace(prepared=prepared, cache_dir=args.cache_dir, batch=args.batch),
                        device)

    parts = []
    for window, path in sections:
        with np.load(path, allow_pickle=False) as packet:
            clip = window['clips'][0]
            if packet['id'].tolist() != [clip['id']] or packet['regression_labels'].tolist() != [clip['valence']]:
                raise ValueError(f'Cached features differ from the release for {clip["id"]}.')
            parts.append(dict(packet))
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    valid_bins = np.isfinite(data['vision']).all(axis=-1) & np.any(data['vision'] != 0, axis=-1)
    report = dict(archive_sha256=archive_digest, annotations_sha256=ANNOTATIONS[args.split][0], text_revision=TEXT[1],
                  vision_revision=VISION[1], clips=len(data['id']),
                  episodes=len({name.split('$_$', 1)[0] for name in data['id']}),
                  label_counts=dict(Counter(int(x) for x in data['regression_labels'])),
                  missing_media=missing_media, fully_missing_vision=data['id'][~valid_bins.any(axis=1)].tolist(),
                  native_vision_bins=np.bincount(valid_bins.sum(axis=1), minlength=FRAME_BINS+1).tolist())
    if args.split != 'test':
        report.update(split=args.split, excluded_test_annotations_sha256=ANNOTATIONS['test'][0])
        if args.split == 'train':
            report['excluded_dev_annotations_sha256'] = ANNOTATIONS['dev'][0]
    temporary = args.output.with_suffix('.npz.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(args.output)
    args.summary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
