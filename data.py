"""Inputs and observation-matched references for the sentiment experiments."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass
class Clips:
    ids: np.ndarray
    groups: np.ndarray
    label: np.ndarray
    sentiment: np.ndarray
    text: np.ndarray
    text_mask: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    audio_mask: np.ndarray
    vision_mask: np.ndarray


def pool_frames(frames, bins):
    """Average observed frames in disjoint bins on each recorded sequence extent."""
    pooled = np.zeros((len(frames), bins, frames.shape[-1]), np.float32)
    observed_bins = np.zeros((len(frames), bins), bool)
    for i, sequence in enumerate(frames):
        observed = np.flatnonzero(np.isfinite(sequence).all(-1) & np.any(sequence != 0, -1))
        if len(observed) == 0:
            continue
        positions = ((2 * observed + 1) * bins) // (2 * (observed[-1] + 1))
        counts = np.bincount(positions, minlength=bins)
        np.add.at(pooled[i], positions, sequence[observed])
        pooled[i] /= counts.clip(1)[:, None]
        observed_bins[i] = counts > 0
    return pooled, observed_bins


def transcode(packet, bins, neutral_margin=0.):
    ids = np.asarray(packet['id']).astype(str).reshape(-1)
    if any('$_$' not in sample_id for sample_id in ids):
        raise ValueError('Expected clip IDs with an original-video prefix separated by $_$.')
    groups = np.asarray([sample_id.split('$_$', 1)[0] for sample_id in ids])
    text = np.asarray(packet['text'], dtype=np.float16)
    bert = np.asarray(packet['text_bert'])
    text_mask = (bert[:, 1] != 0) & ~np.isin(bert[:, 0], [0, 101, 102])
    for start in range(0, len(text), 256):
        batch = text[start:start + 256]
        visible = text_mask[start:start + 256]
        visible &= np.isfinite(batch).all(-1)
        batch[~visible] = 0
    sentiment = np.asarray(packet['regression_labels'], dtype=np.float32).reshape(-1)
    if len(sentiment) != len(ids) or not np.isfinite(sentiment).all():
        raise ValueError('Expected one finite sentiment score per clip.')
    if 'audio_bin_mask' in packet:
        audio = np.asarray(packet['audio'], dtype=np.float32)
        audio_mask = np.asarray(packet['audio_bin_mask'], dtype=bool)
        expected = (len(ids), bins)
        if audio.ndim != 3 or audio.shape[:2] != expected or audio_mask.shape != expected:
            raise ValueError('Pre-binned audio must match the clip IDs and requested time grid.')
        if not np.isfinite(audio).all() or not np.array_equal(np.any(audio != 0, axis=-1), audio_mask):
            raise ValueError('Audio bin masks must identify the finite, nonzero features.')
    else:
        audio, audio_mask = pool_frames(np.asarray(packet['audio']), bins)
    vision, vision_mask = pool_frames(np.asarray(packet['vision']), bins)
    polarity = np.sign(sentiment).astype(np.int64)
    polarity[np.abs(sentiment) <= neutral_margin] = 0
    return Clips(ids, groups, polarity + 1, sentiment, text, text_mask,
                 audio, vision, audio_mask, vision_mask)


def source_groups(path):
    with Path(path).open('rb') as stream:
        packets = pickle.load(stream)
    return {str(sample).split('$_$', 1)[0]
            for packet in packets.values() for sample in np.asarray(packet['id']).reshape(-1)}


def load_clips(path, bins, split_mode, split_seed, fold=0, group_reference=None, exclude_eval=None,
               eval_groups=None, neutral_margin=0.):
    if eval_groups is not None and (exclude_eval is not None or split_mode != 'official'):
        raise ValueError('An evaluation group list applies to an official split without exclusions.')
    # Use the original attachment or a verified, trusted distribution only.
    with Path(path).open('rb') as source:
        raw = pickle.load(source)
    if split_mode != 'pilot':
        names = ['train', 'valid', 'test']
        parts = [transcode(raw.pop(name), bins, neutral_margin) for name in names]
        del raw
        clips = Clips(**{key: np.concatenate([getattr(part, key) for part in parts])
                         for key in vars(parts[0])})
        boundaries = np.cumsum([0] + [len(part.ids) for part in parts])
        del parts
        if split_mode == 'official':
            split = {name: np.arange(start, end) for name, start, end in
                     zip(['fit', 'tune', 'eval'], boundaries[:-1], boundaries[1:])}
            if eval_groups is not None:
                requested = Path(eval_groups).read_text(encoding='utf-8').split()
                official = clips.groups[split['eval']]
                unavailable = set(requested) - set(official)
                if not requested or unavailable:
                    raise ValueError('Expected a nonempty list of source videos from the official test packet.')
                split['eval'] = split['eval'][np.isin(official, requested)]
            elif exclude_eval is not None:
                packets = [exclude_eval] if isinstance(exclude_eval, (str, Path)) else exclude_eval
                used_groups = set().union(*(source_groups(packet) for packet in packets))
                keep = ~np.isin(clips.groups[split['eval']], list(used_groups))
                split['eval'] = split['eval'][keep]
                if not len(split['eval']):
                    raise ValueError('No unused source videos remain in the evaluation split.')
        elif split_mode == 'grouped':
            outer = GroupKFold(n_splits=5)
            development, heldout = list(outer.split(clips.ids, groups=clips.groups))[fold]
            inner = GroupShuffleSplit(n_splits=1, test_size=.125, random_state=split_seed + fold)
            fit, tune = next(inner.split(development, groups=clips.groups[development]))
            split = dict(fit=development[fit], tune=development[tune], eval=heldout)
        else:
            seen = np.isin(clips.groups, list(source_groups(group_reference)))
            development, heldout = np.flatnonzero(seen), np.flatnonzero(~seen)
            if not len(development) or not len(heldout):
                raise ValueError('Expected both previously seen and new source videos.')
            inner = GroupShuffleSplit(n_splits=1, test_size=.125, random_state=split_seed)
            fit, tune = next(inner.split(development, groups=clips.groups[development]))
            split = dict(fit=development[fit], tune=development[tune], eval=heldout)
    else:
        training = raw.pop('train')
        del raw
        clips = transcode(training, bins, neutral_margin)
        del training
        outer = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=split_seed)
        development, heldout = next(outer.split(clips.ids, groups=clips.groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=.125, random_state=split_seed + 1)
        fit, tune = next(inner.split(development, groups=clips.groups[development]))
        split = dict(fit=development[fit], tune=development[tune], eval=heldout)
    if len(np.unique(clips.ids)) != len(clips.ids):
        raise ValueError('Clip IDs must be unique.')
    owner = {}
    for role, indexes in split.items():
        for group in np.unique(clips.groups[indexes]):
            if group in owner:
                raise ValueError(f'Original-video group {group} occurs in {owner[group]} and {role}.')
            owner[group] = role
    return clips, split


def normalize_observed(values, visible, fit):
    reference = values[fit][visible[fit]]
    if len(reference) == 0:
        return values
    center = reference.mean(0)
    scale = reference.std(0).clip(1e-5)
    return np.where(visible[..., None], ((values - center) / scale).clip(-8, 8), 0)


def shuffle_within_score_ranks(values, labels, scores, fit, rng):
    """Shuffle values within polarity and five fitting-score rank groups."""
    shuffled = values.copy()
    for polarity in range(3):
        members = fit[labels[fit] == polarity]
        ordered = members[np.argsort(scores[members], kind='stable')]
        for group in np.array_split(ordered, 5):
            shuffled[group] = rng.permutation(values[group])
    return shuffled


def block_mask(native, fraction, rng):
    """Drop one contiguous interval of bins between the first and last visible bin."""
    visible = native.copy()
    for i, row in enumerate(native):
        support = np.flatnonzero(row)
        if len(support) == 0:
            continue
        first, last = support[0], support[-1]
        span = last - first + 1
        width = max(1, round(span * fraction))
        start = rng.integers(first, last + 2 - width)
        visible[i, start:start + width] = False
    return visible


def observation_views(clips, rates, seed, preserve_vision_gaps=False):
    rng = np.random.default_rng(seed)
    audio_rng = np.random.default_rng(seed + 1) if preserve_vision_gaps else rng
    paired = [(clips.audio_mask, clips.vision_mask)]
    for fraction in rates:
        paired.append((block_mask(clips.audio_mask, fraction, audio_rng),
                       block_mask(clips.vision_mask, fraction, rng)))
    paired.append(tuple(np.zeros_like(m) for m in paired[0]))
    return np.asarray(paired)


def reference_banks(clips, views, fit, neighbors, seed):
    """Across-video joint donors with every required observed bin available."""
    pooled = clips.text.sum(1, dtype=np.float32) / clips.text_mask.sum(1).clip(1)[:, None]
    pooled = pooled.astype(np.float32)
    pooled -= pooled[fit].mean(0)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(1e-6)
    similarity = pooled @ pooled[fit].T
    different_video = clips.groups[:, None] != clips.groups[fit][None, :]
    native = views[0].transpose(1, 0, 2).reshape(len(clips.ids), -1)[fit]
    absent = (~native).astype(np.float32).T.copy()
    rng = np.random.default_rng(seed)
    shape = (len(views), len(clips.ids), neighbors)
    semantic = np.full(shape, -1, dtype=np.int32)
    random = np.full(shape, -1, dtype=np.int32)
    for view_index, view in enumerate(views[:-1]):
        required = view.transpose(1, 0, 2).reshape(len(clips.ids), -1)
        compatible = (required.astype(np.float32) @ absent == 0) & different_video
        for i, eligibles in enumerate(compatible):
            if not required[i].any() or not eligibles.any():
                continue
            candidates = np.flatnonzero(eligibles)
            order = np.argsort(-similarity[i, candidates], kind='stable')[:neighbors]
            nearest = candidates[order]
            # Leave the bank short for rare masks rather than favoring repeated entries.
            semantic[view_index, i, :len(nearest)] = fit[nearest]
            random_subset = rng.choice(candidates, size=len(nearest), replace=False)
            random[view_index, i, :len(nearest)] = fit[random_subset]
    return semantic, random
