"""Extract the frames, features, and forecasts of a held-out CREMA-D clip."""

import argparse
import json
from pathlib import Path
import pickle
import re

import numpy as np

from data import pool_frames
from prepare_cremad import frames_at_centers


def known_statements(release):
    instructions = (release / 'docs' / 'README.md').read_text(encoding='utf-8')
    sentence_list = instructions.split('Actors spoke from a selection of 12 sentences (', 1)[1]
    sentence_list = sentence_list.split('The sentences were presented using different emotion', 1)[0]
    statements = {code: sentence for sentence, code in
                  re.findall(r'^- (.+?) \(([A-Z]{3})\)\.?\s*$', sentence_list, flags=re.MULTILINE)}
    if len(statements) != 12:
        raise ValueError('The CREMA-D documentation must identify all 12 statements.')
    return statements


def extract(run, features, release, output):
    metadata = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    statements = known_statements(release)
    with np.load(run / 'predictions.npz', allow_pickle=False) as packet:
        ids = packet['ids']
        masks = packet['masks'][:, 1]
        anchors, fused = packet['anchor'], packet['intensity']
        log_change = np.log(fused.astype(np.float64)) - np.log(anchors.astype(np.float64))
        signed = (log_change[..., 2] - log_change[..., 0]) / 2
        neutral = log_change[..., 1] - (log_change[..., 0] + log_change[..., 2]) / 2
        eligible = masks[0].all(1) & masks[1, :, :4].all(1) & masks[1, :, -4:].all(1)
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            raise ValueError('No test clip has native visual coverage and an interior gap.')
        magnitude = np.abs(signed[1, candidates])
        index = candidates[np.abs(magnitude - np.median(magnitude)).argmin()]
        clip_id = ids[index]
        media_id = clip_id.split('$_$', 1)[1]
        statement = statements[media_id.split('_')[1]]
        frames = frames_at_centers(release / 'VideoFlash' / f'{media_id}.flv')
        slots = np.arange(1, 24, 3)
        if len(frames) != 24:
            raise ValueError('The original video must decode at all 24 bin centers.')
        with features.open('rb') as stream:
            test = pickle.load(stream)['test']
        feature_index, = np.flatnonzero(np.asarray(test['id']).astype(str) == clip_id)
        np.testing.assert_array_equal(test['regression_labels'][feature_index], packet['sentiment'][index])
        visual, observed = pool_frames(test['vision'][feature_index:feature_index + 1], 24)
        np.testing.assert_array_equal(observed[0], masks[0, index])
        channels = np.arange(6)
        source = visual[0, :, channels].astype(np.float64)
        scores, anchor_scores = packet['intensity_score'][:, index], packet['anchor_score'][:, index]
        scale = metadata['selection']['anchor']['intensity_scale']
        gain = metadata['selection']['intensity']['score_gain']
        update = np.stack((-signed[:, index], neutral[:, index], signed[:, index]), axis=-1)
        probabilities = anchors[:, index] * np.exp(update)
        np.testing.assert_allclose(probabilities / probabilities.sum(-1, keepdims=True),
                                   fused[:, index], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(scale * np.tanh(np.arctanh(anchor_scores / scale) + gain * signed[:, index]),
                                   scores, atol=3e-6, rtol=2e-6)
        np.testing.assert_array_equal(anchors[-1, index], fused[-1, index])
        if abs(signed[-1, index]) > 1e-10 or masks[-1, index].any():
            raise ValueError('The visual-absence view must return the anchor.')
        np.savez_compressed(output, clip_id=clip_id, seed=metadata['config']['seed'],
                            transcript=statement, visual=source, frame_slots=slots,
                            frames=np.stack([frames[slot] for slot in slots]),
                            channels=channels, masks=masks[:, index], clip_count=len(candidates),
                            anchor=anchors[:, index], fused=fused[:, index],
                            signed=signed[:, index], neutral=neutral[:, index], gain=gain,
                            anchor_score=anchor_scores, fused_score=scores,
                            target=packet['sentiment'][index], polarity=packet['y'][index])
    print(f'CREMA-D seed {metadata["config"]["seed"]}, {clip_id}; {len(candidates)} eligible test clips.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    extract(args.run, args.features, args.release, args.output)


if __name__ == '__main__':
    main()
