"""Record which released OMG-Emotion clips entered the feature packet."""

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np


def describe(clips):
    if not clips:
        return dict(n_clips=0, n_videos=0)
    valence = np.asarray([row['valence'] for row in clips])
    duration = np.asarray([row['end'] - row['start'] for row in clips])
    sources = {row['id'].split('$_$', 1)[0] for row in clips}
    return dict(n_clips=len(clips), n_videos=len(sources), mean_valence=float(valence.mean()),
                median_duration_seconds=float(np.median(duration)),
                counts=[int((valence < -.10).sum()), int((np.abs(valence) <= .10).sum()),
                        int((valence > .10).sum())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Trusted feature packet created by encode_omg.py.')
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.prepared / 'manifest.json').read_text(encoding='utf-8'))
    availability = json.loads((args.prepared / 'availability.json').read_text(encoding='utf-8'))
    with args.data.open('rb') as stream:
        data = pickle.load(stream)
    retained = {role: set(packet['id']) for role, packet in data.items()}
    partitions = {role: {'retained': [], 'excluded': []} for role in retained}
    rows = []
    for window in manifest['windows']:
        role = window['role']
        status = availability[window['media']]
        for clip in window['clips']:
            keep = clip['id'] in retained[role]
            partitions[role]['retained' if keep else 'excluded'].append(clip)
            rows.append((role, clip['id'], int(keep), status))
    for role, partition in partitions.items():
        if {clip['id'] for clip in partition['retained']} != retained[role]:
            raise ValueError(f'Unmatched feature packet clips in {role}.')
    with args.index.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('split', 'clip_id', 'retained', 'section_status'))
        writer.writerows(rows)
    summary = {role: {status: describe(clips) for status, clips in partition.items()}
               for role, partition in partitions.items()}
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
