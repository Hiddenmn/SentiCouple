"""Test separate pooling of signed and neutral probability corrections."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ablate import ranked_probability_score
from compare_benchmarks import load_metadata
from coupling import rps_summary
from ordinal_controls import paired_reduction
from score_anchor_effects import SEEDS
from transfer_moud import check_configuration, matched_input, replay_seed


HEADS = {'shared': 'intensity_class_mean', 'independent': 'independent_class_mean'}
GAPPED = slice(1, 3)
TRIANGLE = np.array([[-1., 0.], [1., 0.], [0., 1.]])


def coordinates(posterior):
    negative, neutral, positive = np.moveaxis(posterior.astype(np.float64), -1, 0)
    return np.stack((positive - negative, neutral), axis=-1)


def posterior(coordinate):
    signed, neutral = np.moveaxis(coordinate, -1, 0)
    polar_mass = 1 - neutral
    signed = signed.clip(-polar_mass, polar_mass)
    return np.stack(((polar_mass - signed) / 2, neutral, (polar_mass + signed) / 2), axis=-1)


def project(coordinate):
    """Return the closest valid (signed, neutral) coordinate for every point."""
    signed, neutral = np.moveaxis(coordinate, -1, 0)
    outside = (neutral < 0) | (np.abs(signed) + neutral > 1)
    if not outside.any():
        return coordinate
    edges = np.roll(TRIANGLE, -1, axis=0) - TRIANGLE
    points = coordinate[outside, None, :]
    displacement = points - TRIANGLE
    fraction = np.sum(displacement * edges, axis=-1) / np.sum(edges**2, axis=-1)
    nearest = TRIANGLE + fraction.clip(0, 1)[..., None] * edges
    choice = np.argmin(np.sum((points - nearest)**2, axis=-1), axis=-1)
    projected = coordinate.copy()
    projected[outside] = nearest[np.arange(len(nearest)), choice]
    return projected


def outcomes(label):
    return np.stack((label - 1, label == 1), axis=-1)


def least_squares_weight(delta, residual, axes):
    numerator = np.sum(delta * residual, axis=axes, keepdims=True)
    denominator = np.sum(delta**2, axis=axes, keepdims=True)
    weight = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return weight.clip(0, 1)


def fit_pool(forecast, label, recenter):
    """Fit per-seed weights on calibration observations and their two gap views."""
    coord = {name: coordinates(values[:, GAPPED]) for name, values in forecast.items()}
    observed = outcomes(label)
    offset = (observed - coord['anchor']).mean(axis=(1, 2), keepdims=True) if recenter else 0.
    anchor = project(coord['anchor'] + offset)
    residual = observed - anchor
    weights = {}
    for key in HEADS:
        delta = project(coord[key] + offset) - anchor
        weights[key] = {'one': least_squares_weight(delta, residual, (1, 2, 3)),
                        'two': least_squares_weight(delta, residual, (1, 2))}
    return offset, weights


def apply_pool(forecast, offset, weights):
    endpoints = {name: project(coordinates(values) + offset) for name, values in forecast.items()}
    anchor = endpoints['anchor']
    result = dict(forecast, **{f'{name}_full': posterior(coord) for name, coord in endpoints.items()})
    projection = {}
    for key in HEADS:
        for variant, weight in weights[key].items():
            candidate = anchor + weight * (endpoints[key] - anchor)
            valid = project(candidate)
            name = f'{key}_{variant}'
            result[name] = posterior(valid)
            projection[name] = float(np.any(np.abs(valid[:, GAPPED] - candidate[:, GAPPED]) > 1e-8, axis=-1).mean())
            np.testing.assert_allclose(result[name][:, -1], result['anchor_full'][:, -1], atol=2e-7, rtol=2e-6)
    return result, projection


def source_tuning(shared_runs, independent_runs):
    packets = []
    for shared, independent in zip(shared_runs, independent_runs):
        with (np.load(shared / 'tuning.npz', allow_pickle=False) as left,
              np.load(independent / 'tuning.npz', allow_pickle=False) as right):
            np.testing.assert_array_equal(left['y'], right['y'])
            np.testing.assert_array_equal(left['anchor'], right['anchor'])
            packets.append(dict(y=left['y'], anchor=left['anchor'],
                                shared=left[HEADS['shared']], independent=right[HEADS['independent']]))
    for packet in packets[1:]:
        np.testing.assert_array_equal(packet['y'], packets[0]['y'])
    forecasts = {key: np.stack([packet[key] for packet in packets]) for key in ('anchor', *HEADS)}
    return forecasts, packets[0]['y']


def transfer(path, source_path, config, runs, device):
    batches, source_ids, source_index, target, target_index = matched_input(source_path, path, config, device)
    forecasts = []
    for original, shared, independent in zip(*runs):
        plans = [(shared, (HEADS['shared'],)), (independent, (HEADS['independent'],))]
        forecasts.append(replay_seed(original, plans, config, batches, source_ids, source_index, target_index))
    keys = dict(anchor='anchor', **HEADS)
    posterior_by_head = {name: np.stack([item[key] for item in forecasts]) for name, key in keys.items()}
    print(f'Reproduced source predictions and transferred all {len(forecasts)} fits to {path.name}.', flush=True)
    return posterior_by_head, target.label, target.groups


def score(forecasts, label):
    return {key: ranked_probability_score(values[:, GAPPED], label).mean(axis=1)
            for key, values in forecasts.items()}


def parameters(offset, weights):
    return dict(offset=np.asarray(offset).reshape(-1, 2).tolist() if np.ndim(offset) else float(offset),
                weights={head: {variant: values[:, 0, 0, :].tolist() for variant, values in variants.items()}
                         for head, variants in weights.items()})


def describe(path, label, groups):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(clips=len(label), clusters=len(np.unique(groups)), feature_sha256=digest,
                class_counts=np.bincount(label, minlength=3).tolist())


def contrasts(loss, comparisons, groups, resamples, family):
    _, membership = np.unique(groups, return_inverse=True)
    return {f'versus_{key}': paired_reduction(loss[key] - loss['shared_two'], membership, resamples, family)
            for key in comparisons}


def coordinate_gain(forecasts, label):
    observed = outcomes(label)
    errors = {name: (coordinates(forecasts[name][:, GAPPED]) - observed)**2 / 4
              for name in ('shared_one', 'shared_two', 'anchor_full')}
    return {f'versus_{name}': (errors[name] - errors['shared_two']).mean(axis=(0, 1, 2)).tolist()
            for name in ('shared_one', 'anchor_full')}


def evaluate(local, held, resamples):
    tune, label, groups = local
    forecast, observed, episodes = held
    fitted = fit_pool(tune, label, recenter=True)
    predictions, projection = apply_pool(forecast, *fitted)
    loss = score(predictions, observed)
    primary = ('shared_one', 'anchor_full')
    secondary = ('independent_two', 'shared_full')
    left_out = []
    for cluster in np.unique(groups):
        keep = groups != cluster
        refit = fit_pool({key: values[:, :, keep] for key, values in tune.items()}, label[keep], recenter=True)
        alternate, _ = apply_pool(forecast, *refit)
        risks = score(alternate, observed)
        left_out.append({key: float((risks[key] - risks['shared_two']).mean()) for key in primary})
    return dict(parameters=parameters(*fitted), final_projection_fraction=projection,
                methods={name: rps_summary(values[:, GAPPED], observed) for name, values in predictions.items()},
                primary=contrasts(loss, primary, episodes, resamples, len(primary)),
                secondary=contrasts(loss, secondary, episodes, resamples, len(secondary)),
                rps_reduction_by_coordinate=coordinate_gain(predictions, observed),
                leave_one_calibration_episode_out={
                    key: [min(row[key] for row in left_out), max(row[key] for row in left_out)] for key in primary})


def evaluate_frozen(held, fitted, resamples):
    forecast, label, groups = held
    predictions, projection = apply_pool(forecast, *fitted)
    loss = score(predictions, label)
    return dict(final_projection_fraction=projection,
                methods={name: rps_summary(values[:, GAPPED], label) for name, values in predictions.items()},
                contrasts=contrasts(loss, ('shared_one', 'anchor'), groups, resamples, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--source-data', type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--held-out', required=True, type=Path)
    parser.add_argument('--meld', required=True, type=Path)
    parser.add_argument('--moud', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resamples', type=int, default=20000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('Choose at least one bootstrap resample.')
    prefixes = ('omg_pointwise_heads', 'omg_ordinal_controls', 'omg_independent_class_mean')
    runs = [[args.runs / f'{name}_seed{seed}' for seed in SEEDS] for name in prefixes]
    config = check_configuration(*(load_metadata(paths) for paths in runs))
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    paths = dict(calibration=args.calibration, held_out=args.held_out, meld=args.meld, moud=args.moud)
    cohorts = {key: transfer(path, args.source_data or Path(config['data']), config, runs, device)
               for key, path in paths.items()}
    cohort_info = {key: describe(path, *cohorts[key][1:]) for key, path in paths.items()}
    for key, shape in dict(calibration=(207, 15), held_out=(1237, 32), meld=(2610, 143), moud=(498, 79)).items():
        info = cohort_info[key]
        if (info['clips'], info['clusters']) != shape:
            raise ValueError(f'Unexpected cohort size for {key}.')
    calibrating, evaluating, previous = [set(cohorts[name][2]) for name in ('calibration', 'held_out', 'meld')]
    if calibrating & evaluating or calibrating & previous or evaluating & previous:
        raise ValueError('MELD calibration and evaluation episodes must be disjoint.')

    local = evaluate(cohorts['calibration'], cohorts['held_out'], args.resamples)
    tuning, source_labels = source_tuning(*runs[1:])
    source_fit = fit_pool(tuning, source_labels, recenter=False)
    frozen = {key: evaluate_frozen(cohorts[key], source_fit, args.resamples) for key in ('meld', 'moud')}
    report = dict(seeds=SEEDS, resamples=args.resamples, cohorts=cohort_info, local=local, frozen=frozen,
                  source_tuning_parameters=parameters(*source_fit), source_tuning_clips=len(source_labels))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(local=local['primary'], frozen={key: item['contrasts'] for key, item in frozen.items()}),
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
