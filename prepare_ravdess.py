"""Build disjoint-actor RAVDESS feature packets from the published tracks and ratings."""

import argparse
import io
from pathlib import Path, PurePosixPath
import pickle
import re
from zipfile import ZipFile

import numpy as np


ACTOR_FOLDS = ((1, 2, 9, 10, 17, 18), (3, 4, 11, 12, 19, 20),
               (5, 6, 13, 14, 21, 22), (7, 8, 15, 16, 23, 24))
POLARITY = {1: 0, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1}
TRIAL_NAME = re.compile(r'\d{2}(?:-\d{2}){6}')


def ratings(path):
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    scores = {}
    try:
        rows = book['Speech'].iter_rows(min_row=2, values_only=True)
        header = next(rows)
        name, intensity = header.index('Filename'), header.index('Intensity(M)')
        for row in rows:
            if not isinstance(row[name], str):
                continue
            trial = Path(row[name]).stem
            modality, channel, emotion, *_ = map(int, trial.split('-'))
            if (modality, channel) == (1, 1) and emotion in POLARITY:
                strength = 0. if emotion == 1 else float(row[intensity])
                if trial in scores or (emotion != 1 and not 1 <= strength <= 5):
                    raise ValueError(f'Invalid rating for {trial}.')
                scores[trial] = POLARITY[emotion] * strength / 5
    finally:
        book.close()
    return scores


def action_unit_tracks(path, trials):
    tracks = {}
    columns = None
    with ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            trial = name.stem
            if name.suffix.lower() != '.csv' or not TRIAL_NAME.fullmatch(trial) or trial not in trials:
                continue
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig') as source:
                header = [value.strip() for value in source.readline().split(',')]
                units = [value for value in header if re.fullmatch(r'AU\d{2}_r', value)]
                if columns is None:
                    columns = units
                if not units or units != columns or trial in tracks:
                    raise ValueError(f'Inconsistent action-unit columns or repeated track for {trial}.')
                indexes = [header.index(field) for field in ('frame', *units)]
                matrix = np.loadtxt(source, delimiter=',', usecols=indexes, dtype=np.float32, ndmin=2)
            frames = matrix[:, 0].astype(np.int32) - 1
            if not len(frames) or frames.min() < 0 or np.any(np.diff(frames) <= 0):
                raise ValueError(f'Expected unique increasing frame indices for {trial}.')
            observed = np.isfinite(matrix[:, 1:]).all(axis=1)
            track = np.zeros((frames[-1] + 1, len(units) + 1), dtype=np.float32)
            track[frames[observed], :-1] = matrix[observed, 1:]
            # Preserve valid frames with zero intensities through the shared binning routine.
            track[frames[observed], -1] = 1
            tracks[trial] = track
    if set(tracks) != set(trials):
        raise ValueError(f'Missing released tracks for {len(set(trials) - set(tracks))} speech trials.')
    return tracks, columns


def build_packets(tracks, scores, output):
    trials = sorted(tracks)
    n_clips = len(trials)
    actors = np.asarray([int(trial[-2:]) for trial in trials])
    statements = np.asarray([int(trial.split('-')[4]) for trial in trials])
    if len(actors) != 1056 or np.bincount(actors, minlength=25).tolist() != [0] + [44] * 24:
        raise ValueError('Expected 44 speech trials at each of 24 actors.')
    if not np.isin(statements, [1, 2]).all():
        raise ValueError('Expected the two speech statements.')
    dimensions = next(iter(tracks.values())).shape[1]
    longest = max(len(track) for track in tracks.values())
    vision = np.zeros((n_clips, longest, dimensions), dtype=np.float32)
    for row, trial in enumerate(trials):
        vision[row, :len(tracks[trial])] = tracks[trial]
    tokens = np.zeros((n_clips, 3, 1), dtype=np.int64)
    tokens[:, 0, 0] = statements
    tokens[:, 1, 0] = 1
    packet = dict(id=np.array([f'actor{actor:02d}$_${trial}' for actor, trial in zip(actors, trials)]),
                  regression_labels=np.array([scores[trial] for trial in trials], dtype=np.float32),
                  text=np.eye(2, dtype=np.float32)[statements - 1, None, :], text_bert=tokens,
                  audio=np.zeros((n_clips, 1, 1), dtype=np.float32), vision=vision)
    output.mkdir(parents=True, exist_ok=True)
    for fold, heldout_actors in enumerate(ACTOR_FOLDS):
        test = np.isin(actors, heldout_actors)
        valid = np.isin(actors, ACTOR_FOLDS[(fold + 1) % len(ACTOR_FOLDS)][:4])
        roles = {'train': ~(test | valid), 'valid': valid, 'test': test}
        data = {role: {key: value[mask] for key, value in packet.items()} for role, mask in roles.items()}
        with (output / f'ravdess_fold{fold}.pkl').open('wb') as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scores = ratings(args.ratings)
    tracks, units = action_unit_tracks(args.tracks, scores)
    build_packets(tracks, scores, args.output)
    missing = sum(not np.any(track[:, -1]) for track in tracks.values())
    print(f'{len(tracks)} clips, {len(units)} facial action units, {missing} clips without tracking')


if __name__ == '__main__':
    main()
