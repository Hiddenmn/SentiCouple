"""Fit and compare exact and linear extensions of the shared intensity tilt."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from ablate import ranked_probability_score, read_predictions
from compare_benchmarks import load_matched, load_metadata
from coupling import rps_summary
from encode_omg import atomic_npz
from face_crop_effects import unchanged_settings
from ordinal_controls import paired_reduction
from polarity_pool import coordinates, outcomes
from prepare_ravdess import ACTOR_FOLDS
from published_seed_effects import verify_training
from score_anchor_effects import SEEDS
from train import MARGINAL_INTENSITY_ARMS, run as train_compact


NATURAL = {'sims_v2': 'sims_v2_compact_pooled', 'omg': 'omg_pointwise_heads'}
ACTOR_PREFIXES = tuple(f'ravdess_fold{fold}' for fold in range(len(ACTOR_FOLDS)))
TRIALS = dict(NATURAL, ravdess=ACTOR_PREFIXES)
FIELDS = ('ids', 'groups', 'y', 'sentiment', 'masks')
PUBLISHED = {'lnln': ('lnln_joint', 'lnln_joint'), 'rmnet': ('rmnet_uniform', 'rmnet_core')}


def prefixes(cohort):
    return tuple(TRIALS[cohort]) if cohort == 'ravdess' else (TRIALS[cohort],)


def fit_trials(cohort, runs, work):
    for seed in SEEDS:
        for prefix in prefixes(cohort):
            name = f'{prefix}_seed{seed}'
            original = runs / name
            output = work / cohort / name
            if (output / 'run.json').is_file():
                continue
            base, = load_metadata([original])
            settings = base['config']
            if settings['seed'] != seed or settings['selection'] != 'rps' or not settings['calibrate_score_gain']:
                raise ValueError(f'Expected the original RPS-selected, tuning-calibrated fit at {original}.')
            output.parent.mkdir(parents=True, exist_ok=True)
            print(f'Intensity marginal, {name}.', flush=True)
            with tempfile.TemporaryDirectory(prefix='.fit_', dir=output.parent) as directory:
                staged = Path(directory).resolve() / name
                if not staged.is_relative_to(output.parent.resolve()):
                    raise ValueError('The new fit must stay within the experiment directory.')
                config = dict(settings, data=Path(settings['data']), output=staged,
                              arms=list(MARGINAL_INTENSITY_ARMS))
                config.setdefault('teacher', None)
                train_compact(argparse.Namespace(**config))
                record_path = staged / 'run.json'
                result = json.loads(record_path.read_text(encoding='utf-8'))
                unchanged_settings(base, result, ('output', 'arms'))
                if result['splits'] != base['splits']:
                    raise ValueError(f'{name} does not match the original partitions.')
                result['config']['output'] = str(output)
                record_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                staged.rename(output)


def collect_fold(runs, work, cohort, prefix):
    original = [runs / f'{prefix}_seed{seed}' for seed in SEEDS]
    changed = [work / cohort / row.name for row in original]
    old_config, new_config = load_metadata(original), load_metadata(changed)
    reference = read_predictions(original[0])
    old, new = load_matched(original, reference), load_matched(changed, reference)
    for seed, baseline, experiment, before, after in zip(SEEDS, old_config, new_config, old, new, strict=True):
        unchanged_settings(baseline, experiment, ('output', 'arms'))
        if (experiment['config']['seed'] != seed or experiment['splits'] != baseline['splits']
                or set(experiment['config']['arms']) != set(MARGINAL_INTENSITY_ARMS)):
            raise ValueError(f'Unexpected partition, arm or seed in {prefix} seed {seed}.')
        np.testing.assert_allclose(before['anchor'], after['anchor'], atol=3e-6, rtol=2e-5)
        for arm in MARGINAL_INTENSITY_ARMS:
            comparator = 'independent_intensity' if arm.endswith('_independent') else 'intensity'
            if experiment['selection'][arm]['parameters'] != baseline['selection'][comparator]['parameters']:
                raise ValueError(f'{arm} must use the original fusion parameter count.')
            np.testing.assert_array_equal(after[arm][-1], after['anchor'][-1])
            np.testing.assert_allclose(after[f'{arm}_score'][-1], after[f'{arm}_class_mean_score'][-1],
                                       atol=2e-6, rtol=0)
    response = new_config[0]['polar_marginal']
    for entry in new_config[1:]:
        np.testing.assert_allclose(entry['polar_marginal']['exact'], response['exact'], atol=2e-6, rtol=0)
        if entry['polar_marginal']['scale'] != response['scale']:
            raise ValueError('Paired seeds must share the same fitting-score reference.')

    evidence = {key: reference[key] for key in FIELDS}
    evidence['response_delta'] = np.asarray(response['delta'])
    evidence['response_exact'] = np.asarray(response['exact'])
    for arm, packets in (('intensity', old), ('independent_intensity', old),
                         *((name, new) for name in MARGINAL_INTENSITY_ARMS)):
        evidence[arm] = np.stack([packet[arm][1:3] for packet in packets])
        score_keys = [f'{arm}_score']
        if arm in MARGINAL_INTENSITY_ARMS:
            score_keys.append(f'{arm}_uncalibrated_score')
        for key in score_keys:
            evidence[key] = np.stack([packet[key][1:3] for packet in packets])
    if cohort in NATURAL:
        for name, (stem, runner) in PUBLISHED.items():
            paths = [runs / f'{cohort}_{stem}_seed{seed}' for seed in SEEDS]
            metadata = load_metadata(paths)
            verify_training(runner, metadata, SEEDS, cohort)
            if name == 'rmnet' and any(row['config'].get('rmnet_class_weighting') != 'uniform'
                                      for row in metadata):
                raise ValueError('Compare with the uniform-weight RM-Net fits.')
            evidence[name] = np.stack([packet['posterior'][1:3] for packet in load_matched(paths, reference)])
    return evidence


def collect(runs, work):
    evidence = {}
    for cohort in TRIALS:
        folds = [collect_fold(runs, work, cohort, prefix) for prefix in prefixes(cohort)]
        if cohort == 'ravdess':
            for fold, expected in zip(folds, ACTOR_FOLDS, strict=True):
                if set(fold['groups']) != {f'actor{actor:02d}' for actor in expected}:
                    raise ValueError('Each RAVDESS fold must preserve its held-out actors.')
        combined = {}
        for key in folds[0]:
            values = [fold[key] for fold in folds]
            if key.startswith('response_'):
                combined[key] = np.stack(values)
            else:
                axis = (2 if key == 'masks' else 0) if key in FIELDS else 2
                combined[key] = np.concatenate(values, axis=axis)
        if len(np.unique(combined['ids'])) != len(combined['ids']):
            raise ValueError('Each evaluation clip must occur exactly once within its cohort.')
        evidence.update({f'{cohort}__{key}': value for key, value in combined.items()})
    return evidence


def summarize(cohort, evidence, resamples):
    def series(name):
        return evidence[f'{cohort}__{name}']

    label, sentiment = series('y'), series('sentiment')
    groups, membership = np.unique(series('groups'), return_inverse=True)
    weights = np.ones(len(label), dtype=np.float64)
    if cohort == 'ravdess':
        counts = np.stack([np.bincount(label[membership == g], minlength=3) for g in range(len(groups))])
        np.testing.assert_array_equal(counts, np.broadcast_to([32, 4, 8], (24, 3)))
        weights = len(label) / (3 * counts.sum(0)[label])
    arms = ('intensity', 'independent_intensity', *MARGINAL_INTENSITY_ARMS)
    outcomes_by_class = outcomes(label)
    ranked, mae, methods = {}, {}, {}
    for arm in (*arms, *(PUBLISHED if cohort in NATURAL else ())):
        probabilities = series(arm).astype(np.float64)
        np.testing.assert_allclose(probabilities.sum(-1), 1., rtol=0, atol=2e-7)
        loss = ranked_probability_score(probabilities, label)
        ranked[arm] = loss.mean(axis=1) * weights
        decomposition = (np.square(coordinates(probabilities) - outcomes_by_class)
                         * weights[None, None, :, None]).mean(axis=(0, 1, 2)) / 4
        np.testing.assert_allclose(decomposition.sum(), ranked[arm].mean(), atol=5e-8, rtol=0)
        methods[arm] = dict(**rps_summary(probabilities, label), comparison_rps=float(ranked[arm].mean()),
                            signed_rps=float(decomposition[0]), neutral_rps=float(decomposition[1]))
        if arm in arms:
            mae[arm] = np.abs(series(f'{arm}_score') - sentiment).mean(axis=1)
            methods[arm]['mae'] = float(mae[arm].mean())
        if arm in MARGINAL_INTENSITY_ARMS:
            raw = np.abs(series(f'{arm}_uncalibrated_score') - sentiment).mean()
            methods[arm]['raw_mae'] = float(raw)

    exact = ranked['tilt_exact_shared']
    sharing = ranked['tilt_exact_independent'] - exact
    tangent_sharing = ranked['tilt_tangent_independent'] - ranked['tilt_tangent_shared']
    contrasts = dict(sharing=sharing, marginal=ranked['tilt_tangent_shared'] - exact,
                     huber=ranked['intensity'] - exact, interaction=sharing - tangent_sharing)
    primary = {name: paired_reduction(diff, membership, resamples, len(TRIALS) * len(contrasts))
               for name, diff in contrasts.items()}
    secondary = {name: paired_reduction(mae[arm] - mae['tilt_exact_shared'], membership, resamples, len(TRIALS) * 2)
                 for name, arm in (('marginal', 'tilt_tangent_shared'), ('huber', 'intensity'))}
    benchmarks = {name: paired_reduction(ranked[name] - exact, membership, resamples, len(NATURAL) * len(PUBLISHED))
                  for name in PUBLISHED} if cohort in NATURAL else {}
    response = dict(delta=series('response_delta').tolist(), exact=series('response_exact').tolist())
    return dict(clips=len(label), groups=len(groups), methods=methods, primary=primary,
                secondary_mae=secondary, benchmarks=benchmarks, response=response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fit', choices=tuple(TRIALS))
    mode.add_argument('--summarize', action='store_true')
    mode.add_argument('--evidence', type=Path)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--resamples', type=int, default=20_000)
    args = parser.parse_args()
    if (args.fit or args.summarize) and not (args.runs and args.work):
        parser.error('Supply --runs and --work for fitting or collecting forecasts.')
    if args.fit:
        fit_trials(args.fit, args.runs, args.work)
        return
    if not args.output or args.resamples < 1:
        parser.error('A report needs --output and positive resamples.')
    if args.evidence:
        with np.load(args.evidence, allow_pickle=False) as packet:
            data = dict(packet)
    else:
        data = collect(args.runs, args.work)
    result = dict(seeds=SEEDS, resamples=args.resamples,
                  cohorts={cohort: summarize(cohort, data, args.resamples) for cohort in TRIALS})
    if args.write_evidence:
        atomic_npz(args.write_evidence, **data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for cohort, row in result['cohorts'].items():
        for family in ('primary', 'secondary_mae', 'benchmarks'):
            for name, effect in row[family].items():
                low, high = effect['ci95_familywise']
                print(f'{cohort} {family} / {name}: {effect["reduction"]:+.6f} [{low:+.6f}, {high:+.6f}]')


if __name__ == '__main__':
    main()
