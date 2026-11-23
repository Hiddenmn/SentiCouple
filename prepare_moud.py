"""Encode the author-released Spanish movie and product review utterances."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from encode_omg import encode_sections


TRANSLATOR = ('Helsinki-NLP/opus-mt-es-en', 'c96e2c5399ebfae4fc43d9669556b9afa74bb69d')
RELEASE_ROOT = PurePosixPath('MOUD/VideoReviews')
LABELS = {'negative': -1, 'neutral': 0, 'positive': 1}


def read_release(path):
    windows, csv_labels, category_labels = [], {}, {}
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.name.startswith('.'):
                continue
            if name == RELEASE_ROOT / 'cats.txt':
                categories = archive.extractfile(member).read().decode('utf-8-sig')
                for line in categories.splitlines():
                    if line.strip():
                        key, label = line.split()
                        if key in category_labels:
                            raise ValueError(f'Duplicate released label {key}.')
                        category_labels[key] = LABELS[label.lower()]
            if name.parent != RELEASE_ROOT / 'transcriptions' or name.suffix != '.csv':
                continue
            review = name.stem
            if not re.fullmatch(r'[A-Za-z0-9_-]+', review):
                raise ValueError(f'Unsupported review identifier {review!r}.')
            contents = archive.extractfile(member).read().decode('utf-8-sig')
            rows = csv.reader(io.StringIO(contents), delimiter=';')
            first, last, text_column, label_column = [value.strip().lower() for value in next(rows)]
            if ((first, last) != ('#starttime', '#endtime') or text_column not in ('speech', 'transcription')
                    or label_column not in ('sentimentannotation', 'sentimentannotations')):
                raise ValueError(f'Unexpected transcription columns for {review}.')
            clips = []
            for number, (begin, finish, transcript, label) in enumerate(rows, 1):
                sentiment = int(label)
                start, end = float(begin), float(finish)
                if sentiment not in LABELS.values() or not 0 <= start < end or not np.isfinite(end):
                    raise ValueError(f'Invalid annotation for {review}, utterance {number}.')
                key = f'{review}_{number}'
                if key in csv_labels:
                    raise ValueError(f'Duplicate transcript annotation {key}.')
                csv_labels[key] = sentiment
                clips.append(dict(id=f'{review}$_${number}', start=start, end=end,
                                  transcript=transcript.strip(), valence=sentiment))
            windows.append(dict(source=review, media=f'{review}.mp4', start=0., clips=clips))
    if not windows or csv_labels != category_labels:
        mismatched = sum(csv_labels.get(key) != category_labels.get(key)
                         for key in csv_labels.keys() | category_labels.keys())
        raise ValueError(f'Review CSVs and sentiment index disagree on {mismatched} labels.')
    return sorted(windows, key=lambda window: window['source'])


def extract_reviews(archive_path, media_dir, windows):
    media_dir.mkdir(parents=True, exist_ok=True)
    expected = {window['media'] for window in windows}
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.parent != RELEASE_ROOT or name.name not in expected or not member.isfile():
                continue
            path = media_dir / name.name
            if not path.is_file() or path.stat().st_size != member.size:
                temporary = path.with_suffix('.tmp')
                with archive.extractfile(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(path)
            expected.remove(name.name)
    if expected:
        print(f'{len(expected)} annotated source reviews are missing media in the archive.', flush=True)
    return expected


@torch.inference_mode()
def translate(clips, directory, device, batch):
    checkpoint = snapshot_download(TRANSLATOR[0], revision=TRANSLATOR[1], local_dir=directory,
                                   allow_patterns=['*.json', '*.spm', 'pytorch_model.bin'])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True, weights_only=True,
                                                dtype=dtype).to(device).eval()
    for start in range(0, len(clips), batch):
        group = clips[start:start + batch]
        originals = [clip['transcript'] for clip in group]
        tokens = tokenizer(originals, return_tensors='pt', padding=True, truncation=True,
                           max_length=256).to(device)
        translations = model.generate(**tokens, num_beams=4, do_sample=False, max_new_tokens=192)
        for clip, sentence in zip(group, tokenizer.batch_decode(translations, skip_special_tokens=True)):
            clip['transcript'] = sentence.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--complete-intervals', action='store_true',
                        help='Use the earlier sensitivity cohort requiring each full annotated interval.')
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('Choose a positive batch size.')
    if args.output.exists():
        raise FileExistsError(args.output)
    windows = read_release(args.archive)
    all_clips = {clip['id']: clip['valence'] for window in windows for clip in window['clips']}
    missing = extract_reviews(args.archive, args.prepared / 'media', windows)
    windows = [window for window in windows if window['media'] not in missing]
    if not windows:
        raise ValueError('The archive has no matching review videos.')

    cache_kind = 'review_features' if args.complete_intervals else 'review_features_centers'
    cache = args.prepared / f'{cache_kind}_{args.device}_batch{args.batch}'
    cache.mkdir(exist_ok=True)
    sections = [(window, cache / f'{window["source"]}.npz') for window in windows]
    pending = [(window, output) for window, output in sections if not output.exists()]
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if pending:
        clips = [clip for window, _ in pending for clip in window['clips'] if clip['transcript']]
        print(f'Translating {len(clips)} transcripts from {len(pending)} source videos.', flush=True)
        translate(clips, args.prepared / 'translator', device, args.batch * 2)
        for window, _ in pending:
            window['clips'] = [clip for clip in window['clips'] if clip['transcript']]
        encode_sections(pending, args, device, allow_partial=not args.complete_intervals)

    parts = []
    for _, output in sections:
        if output.is_file():
            with np.load(output, allow_pickle=False) as packet:
                parts.append(dict(packet))
    if not parts:
        raise ValueError('The review videos did not produce encoded utterances.')
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    retained, values = data['id'].tolist(), data['regression_labels'].tolist()
    if len(set(retained)) != len(retained) or any(all_clips.get(key) != value for key, value in zip(retained, values)):
        raise ValueError('Encoded review IDs or labels differ from the release.')
    with args.archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(archive_sha256=digest, translator=TRANSLATOR,
                  sampling_scope='full_interval' if args.complete_intervals else 'observed_centers',
                  candidate_clips=len(all_clips),
                  candidate_videos=len({key.split('$_$', 1)[0] for key in all_clips}),
                  clips=len(retained), videos=len({key.split('$_$', 1)[0] for key in retained}),
                  label_counts=dict(Counter(int(value) for value in values)),
                  excluded_ids=sorted(set(all_clips) - set(retained)))
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **data)
    args.summary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
