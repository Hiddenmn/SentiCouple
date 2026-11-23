"""Extract a held-out three-stream forecast from the CH-SIMS v2 experiment."""

import argparse
from pathlib import Path
import pickle

import numpy as np

from data import pool_frames


def extract(run, features, output, english_transcript, streams_run=None):
    if streams_run is None:
        seed = run.name.rsplit('seed', 1)[-1]
        streams_run = run.with_name(f'sims_v2_streams_seed{seed}')
    with (np.load(run / 'predictions.npz', allow_pickle=False) as predictions,
          np.load(streams_run / 'predictions.npz', allow_pickle=False) as streams):
        for field in ('ids', 'y', 'sentiment', 'masks', 'anchor'):
            np.testing.assert_array_equal(predictions[field], streams[field], err_msg=field)
        masks = predictions['masks']
        anchor, fused = predictions['anchor'], predictions['intensity']
        log_change = np.log(fused.astype(np.float64)) - np.log(anchor.astype(np.float64))
        signed = (log_change[..., 2] - log_change[..., 0]) / 2
        candidates = np.flatnonzero(masks[0].all(axis=(0, 2)))
        if not len(candidates):
            raise ValueError('Expected test clips with complete audio and vision at the native view.')
        magnitude = np.abs(signed[1, candidates])
        sample = candidates[np.abs(magnitude - np.median(magnitude)).argmin()]
        clip_id = predictions['ids'][sample]
        selected_masks = masks[:, :, sample]
        np.testing.assert_array_equal(anchor[-1, sample], fused[-1, sample])
        trace = {
            'clip_id': clip_id, 'masks': selected_masks, 'polarity': predictions['y'][sample],
            'target': predictions['sentiment'][sample], 'anchor': anchor[:, sample],
            'fused': fused[:, sample], 'signed': signed[:, sample],
            'fused_score': predictions['intensity_score'][:, sample],
        }
        for stream in ('audio', 'vision'):
            arm = f'intensity_{stream}'
            trace[f'text_{stream}'] = streams[arm][:, sample]
            trace[f'text_{stream}_score'] = streams[f'{arm}_score'][:, sample]

    with features.open('rb') as source:
        corpora = pickle.load(source)
    for corpus in corpora.values():
        matches = np.flatnonzero(np.asarray(corpus['id']).astype(str).reshape(-1) == clip_id)
        if len(matches):
            break
    else:
        raise ValueError(f'No feature record for {clip_id}.')
    row, = matches
    trace['transcript'] = str(corpus['raw_text'][row]).strip()
    trace['transcript_en'] = english_transcript
    score = np.asarray(corpus['regression_labels'], dtype=np.float32).reshape(-1)[row]
    np.testing.assert_array_equal(score, trace['target'])
    for channel, name in enumerate(('audio', 'vision')):
        pooled, visible = pool_frames(corpus[name][row:row + 1], selected_masks.shape[-1])
        np.testing.assert_array_equal(visible[0], selected_masks[0, channel])
        trace[name] = pooled[0, :, :6].T
    np.savez_compressed(output, **trace)
    print(f'{clip_id}: {trace["transcript"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--streams-run', type=Path,
                        help='Directory of the paired audio and visual ablations.')
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--english-transcript', required=True,
                        help='English translation of the selected clip transcript.')
    args = parser.parse_args()
    extract(args.run, args.features, args.output, args.english_transcript, args.streams_run)


if __name__ == '__main__':
    main()
