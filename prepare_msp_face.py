"""Join the MSP-Face release and acquire the annotated public media sections."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess

from data import source_groups
from encode_omg import sha256
from prepare_omg import fetch_section, source_id


REVISION = '18fc6d717aab5a5731ad6bd59495ead3f1d80916'
PARTITIONS = {'Train': 'train', 'Validation': 'valid', 'Test': 'test'}
FILES = ('Links/link_videos.txt', 'Doc/segments_data.txt', 'Doc/Partitions.txt',
         'Doc/speaker_id_segments.txt', 'Labels/label_consensus.json')
FORMAT = ('b[height<=360][ext=mp4][protocol=https]/'
          'bv[height<=360][ext=mp4][protocol=https]+ba[ext=m4a][protocol=https]')


def records(entries, description):
    result = {}
    for key, value in entries:
        if key in result:
            raise ValueError(f'Duplicate {description} key {key}.')
        result[key] = value
    return result


def read_manifest(metadata, source_exclusions):
    revision = subprocess.check_output(['git', '-C', str(metadata), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    if revision != REVISION:
        raise ValueError('Check out the pinned MSP-Face release.')
    links = records(((Path(name).stem, url) for name, url in
                     (row.split() for row in (metadata / FILES[0]).read_text().splitlines())),
                    'source video')
    sources = {}
    for name, link in links.items():
        try:
            sources[name] = source_id(link)
        except ValueError:
            sources[name] = None
    intervals = records(((name, (float(start), float(end))) for start, end, name in
                         (row.split() for row in (metadata / FILES[1]).read_text().splitlines())),
                        'interval')
    partitions = records(((name.strip(), PARTITIONS[role.strip()]) for role, name in
                          (row.split(';') for row in (metadata / FILES[2]).read_text().splitlines())),
                         'partition')
    speakers = records(((name, speaker.strip()) for name, speaker in
                        (row.split(',') for row in (metadata / FILES[3]).read_text().splitlines())),
                       'speaker')
    ratings = json.loads((metadata / FILES[4]).read_text(encoding='utf-8'))
    if set(partitions) != set(ratings):
        raise ValueError('The consensus and official partition files contain different clips.')
    by_source = defaultdict(list)
    speaker_roles = defaultdict(set)
    original = defaultdict(list)
    excluded = Counter()
    for name, (row,) in sorted(ratings.items()):
        released_video = name.rsplit('_', 1)[0]
        source = sources[released_video]
        start, end = intervals[name]
        speaker = f'Speaker_{row["SpkrID"]}'
        role = partitions[name]
        rating = float(row['EmoVal'])
        if (role != PARTITIONS[row['Split_Set']] or speakers[name] != speaker
                or not (0 <= start < end and 1 <= rating <= 7)):
            raise ValueError(f'Inconsistent annotation for {name}.')
        item = dict(id='$_$'.join((source or released_video, Path(name).stem)),
                    source=source or released_video, role=role, speaker=speaker,
                    start=start, end=end, valence=(rating - 4) / 3)
        original[role].append(item)
        if source is None:
            excluded[role, 'invalid_link'] += 1
            continue
        by_source[source].append(item)
        speaker_roles[speaker].add(role)
    if any(len(roles) > 1 for roles in speaker_roles.values()):
        raise ValueError('An official speaker belongs to more than one partition.')

    windows = []
    for source, clips in sorted(by_source.items()):
        roles = {clip['role'] for clip in clips}
        reason = 'cross_partition' if len(roles) > 1 else None
        if source in source_exclusions:
            reason = 'previous_corpus'
        if reason is not None:
            for clip in clips:
                excluded[clip['role'], reason] += 1
            continue
        start = max(0., min(clip['start'] for clip in clips) - .25)
        end = max(clip['end'] for clip in clips) + .25
        windows.append(dict(role=roles.pop(), source=source, start=start, end=end,
                            media=f'{source}.mp4', clips=clips))

    def counts(clips):
        return dict(clips=len(clips), videos=len({clip['source'] for clip in clips}),
                    speakers=len({clip['speaker'] for clip in clips}))

    return dict(revision=revision, filenames={name: sha256(metadata / name) for name in FILES},
                original={role: counts(original[role]) for role in PARTITIONS.values()},
                excluded={role: {reason: excluded[role, reason] for reason in
                                 ('invalid_link', 'cross_partition', 'previous_corpus')}
                          for role in PARTITIONS.values()},
                eligible={role: counts([clip for window in windows if window['role'] == role
                                        for clip in window['clips']])
                          for role in PARTITIONS.values()}, windows=windows)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def acquire(windows, output, workers, retry_errors):
    import imageio_ffmpeg

    node = shutil.which('node')
    if node is None:
        raise RuntimeError('Node.js is required by the public media downloader.')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    media = output / 'media'
    media.mkdir(exist_ok=True)
    report = output / 'availability.json'
    status = json.loads(report.read_text(encoding='utf-8')) if report.is_file() else {}
    pending = [window for window in windows if window['media'] not in status
               or (retry_errors and status[window['media']] in ('timeout', 'download_error'))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_section, window, media, node, ffmpeg,
                               format_selector=FORMAT, timeout=240): window['media']
                   for window in pending}
        for index, future in enumerate(as_completed(futures), 1):
            status[futures[future]] = future.result()
            write_json(report, status)
            if index % 12 == 0 or index == len(futures):
                summary = dict(Counter(status[window['media']] for window in windows
                                       if window['media'] in status))
                print(f'Acquired {index}/{len(futures)} new sections. {summary}', flush=True)
    print(dict(Counter(status[window['media']] for window in windows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--omg-manifest', type=Path, required=True)
    parser.add_argument('--prior-packet', type=Path, action='append', required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--roles', choices=tuple(PARTITIONS.values()), nargs='+', default=['test', 'valid', 'train'])
    parser.add_argument('--pilot', action='store_true', help='Use the shortest eligible training and validation sections.')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'manifest.json'
    exclusion_keys = {'omg': sha256(args.omg_manifest),
                      'packets': {path.name: sha256(path) for path in args.prior_packet}}
    if target.is_file():
        manifest = json.loads(target.read_text(encoding='utf-8'))
        if (manifest['exclusion_inputs'] != exclusion_keys
                or manifest['filenames'] != {name: sha256(args.metadata / name) for name in FILES}):
            raise ValueError('Inputs differ from the cohort manifest. Use a separate output directory.')
    else:
        omg = json.loads(args.omg_manifest.read_text(encoding='utf-8'))
        excluded = {window['source'] for window in omg['windows']}
        for path in args.prior_packet:
            excluded.update(source_groups(path))
        manifest = read_manifest(args.metadata, excluded)
        manifest['exclusion_inputs'] = exclusion_keys
        write_json(target, manifest)
    print({key: manifest[key] for key in ('original', 'eligible', 'excluded')}, flush=True)
    if args.download:
        windows = [window for role in args.roles for window in manifest['windows'] if window['role'] == role]
        if args.pilot:
            windows = [window for role in ('train', 'valid') for window in sorted(
                (row for row in windows if row['role'] == role and len(row['clips']) >= 3),
                key=lambda row: row['end'] - row['start'])[:3]]
        acquire(windows, args.output, args.workers, args.retry_errors)


if __name__ == '__main__':
    main()
