"""Join OMG-Emotion annotations and cache the corresponding public video sections."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

from sklearn.model_selection import GroupShuffleSplit


REVISION = '5931b237e92d68d04931bb932854fac6d9cd6a41'
PARTITIONS = ('Train', 'Validation', 'Test')


def read_rows(path):
    # The released test transcript file has a .tsv suffix and comma-separated rows.
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_id(link):
    parsed = urlsplit(link)
    ids = parse_qs(parsed.query).get('v', [])
    if parsed.hostname != 'www.youtube.com' or len(ids) != 1 or not re.fullmatch(r'[\w-]{11}', ids[0], re.ASCII):
        raise ValueError('Expected a release URL with one canonical source-video ID.')
    return ids[0]


def clip_key(row):
    return source_id(row['link']), row['video'], row['utterance']


def join_release(metadata):
    if (metadata / 'revision.txt').read_text(encoding='ascii').strip() != REVISION:
        raise ValueError('Use the pinned OMG-Emotion release metadata.')
    annotations = {split: read_rows(metadata / (f'omg_{split}Videos' +
                   ('_WithLabels' if split == 'Test' else '') + '.csv')) for split in PARTITIONS}
    development = {source_id(row['link']) for split in PARTITIONS[:2] for row in annotations[split]}
    groups = sorted(development)
    _, tuning = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=1729).split(groups, groups=groups))
    tuning = {groups[i] for i in tuning}
    by_window = defaultdict(list)
    counters = Counter()
    for partition, rows in annotations.items():
        suffix = 'tsv' if partition == 'Test' else 'csv'
        transcript_rows = read_rows(metadata / f'omg_{partition}Transcripts.{suffix}')
        transcripts = {clip_key(row): (row.get('transcript') or '').strip() for row in transcript_rows}
        for row in rows:
            source, segment, utterance = clip_key(row)
            if partition == 'Test' and source in development:
                counters['test_rows_from_development_sources'] += 1
                continue
            transcript = transcripts.get((source, segment, utterance))
            if not transcript:
                counters['rows_without_transcript'] += 1
                continue
            start, end, score = (float(row[key]) for key in ('start', 'end', 'valence'))
            if not (0 <= start < end and -1 <= score <= 1):
                raise ValueError('The release contains an invalid time interval or valence.')
            role = 'test' if partition == 'Test' else ('valid' if source in tuning else 'train')
            by_window[role, source, segment].append(dict(
                id=f'{source}$_${partition}/{segment}/{utterance}', start=start, end=end,
                transcript=transcript, valence=score))
    windows = []
    for (role, source, segment), clips in sorted(by_window.items()):
        start = max(0., min(clip['start'] for clip in clips) - .25)
        end = max(clip['end'] for clip in clips) + .25
        name = hashlib.sha256(f'{role}/{source}/{segment}'.encode()).hexdigest()[:20] + '.mp4'
        windows.append(dict(role=role, source=source, start=start, end=end, media=name, clips=clips))
    return windows, counters


def decode_duration(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not capture.read()[0]:
            return 0.
        return capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    finally:
        capture.release()


def section_status(window, output):
    if not output.is_file():
        return 'download_error'
    duration = decode_duration(output)
    complete = sum(clip['end'] - window['start'] <= duration + .04 for clip in window['clips'])
    if complete == len(window['clips']):
        return 'available'
    return 'partial_section' if complete else 'short_section'


def fetch_section(window, media, node, ffmpeg, *,
                  format_selector='bv*[height<=360][ext=mp4][protocol=https]', timeout=120):
    output = media / window['media']
    if output.exists():
        return section_status(window, output)
    command = [sys.executable, '-m', 'yt_dlp', '--js-runtimes', f'node:{node}', '--no-playlist',
               '--socket-timeout', '12', '--retries', '1', '--quiet', '--no-warnings', '--ffmpeg-location', ffmpeg,
               '--download-sections', f'*{window["start"]:.6f}-{window["end"]:.6f}', '--force-keyframes-at-cuts',
               '--format', format_selector, '--output', str(output),
               f'https://www.youtube.com/watch?v={window["source"]}']
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 'timeout'
    if process.returncode:
        message = process.stderr.lower()
        if any(reason in message for reason in ('private video', 'video is unavailable')):
            return 'unavailable'
        if 'sign in' in message:
            return 'access_required'
        return 'download_error'
    return section_status(window, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--retry-errors', action='store_true',
                        help='Retry timeouts and download errors from an earlier acquisition.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Process only this many sections when testing media extraction.')
    args = parser.parse_args()
    if args.retry_errors and not args.download:
        parser.error('--retry-errors requires --download.')
    windows, excluded = join_release(args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    media = args.output / 'media'
    summaries = {role: {'clips': sum(len(window['clips']) for window in windows if window['role'] == role),
                        'videos': len({window['source'] for window in windows if window['role'] == role})}
                 for role in ('train', 'valid', 'test')}
    report = dict(revision=REVISION, splits=summaries, excluded=excluded, windows=windows)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summaries, excluded, flush=True)
    if not args.download:
        return

    import imageio_ffmpeg

    node = shutil.which('node')
    if node is None:
        raise RuntimeError('Node.js is required by the public media downloader.')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    media.mkdir(exist_ok=True)
    status = {}
    report_path = args.output / 'availability.json'
    if report_path.exists():
        status = json.loads(report_path.read_text(encoding='utf-8'))
    selected = [window for window in windows if not args.retry_errors or
                status.get(window['media']) in ('timeout', 'download_error')][:args.limit]
    if not selected:
        print('No additional video sections to acquire.', flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(fetch_section, window, media, node, ffmpeg) for window in selected]
        for index, (window, task) in enumerate(zip(selected, tasks), 1):
            status[window['media']] = task.result()
            if index % 20 == 0 or index == len(selected):
                print(f'{index}/{len(selected)} sections {dict(Counter(status.values()))}', flush=True)
                temporary = report_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
                temporary.replace(report_path)


if __name__ == '__main__':
    main()
