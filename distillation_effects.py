"""Compare anchored and absolute distillation on paired held-out videos."""

import argparse
import json
from pathlib import Path

import numpy as np

from ablate import read_predictions
from compare_benchmarks import load_matched, load_metadata, summarize_forecasts
from ordinal_controls import paired_reduction, verify_pairs
from polarity_pool import coordinates, outcomes
from published_seed_effects import count_anchor, run_paths, verify_training
from score_anchor_effects import SEEDS


COHORTS = {'sims_v2': 'fixed_pooling', 'omg': 'pointwise'}
STUDENTS = {'native': 'distill_native', 'matched': 'distill_matched',
            'native_prior': 'distill_native_prior',
            'anchored_native': 'distill_anchored_native', 'anchored_matched': 'distill_anchored_matched',
            'independent': 'distill_anchored_independent'}
PRIMARY = ('original', 'native', 'native_prior', 'anchored_matched')
SECONDARY = ('independent', 'teacher')
METHODS = ('original', *STUDENTS, 'teacher')


def collect(root, cohort):
    original = COHORTS[cohort]
    paths = run_paths(root, SEEDS, cohort)
    reference = read_predictions(paths[original][0])
    originals, teachers = (load_matched(paths[key], reference) for key in (original, 'rmnet_core'))
    original_meta, teacher_meta = (load_metadata(paths[key]) for key in (original, 'rmnet_core'))
    verify_training(original, original_meta, SEEDS, cohort)
    verify_training('rmnet_core', teacher_meta, SEEDS, cohort)
    anchor_count = count_anchor(paths[original][0])
    variants = [('original', 'intensity', originals), ('teacher', 'posterior', teachers)]
    parameters = {'original': original_meta[0]['selection']['intensity']['parameters'] + anchor_count,
                  'teacher': teacher_meta[0]['trainable_parameters']}
    epochs = {}
    for prefix, names in [('reference_distill', [name for name in STUDENTS if name != 'native_prior']),
                          ('reference_distill_prior', ['native_prior'])]:
        student_paths = [root / f'{cohort}_{prefix}_seed{seed}' for seed in SEEDS]
        students, student_meta = load_matched(student_paths, reference), load_metadata(student_paths)
        comparable = [dict(row, config={key: value for key, value in row['config'].items() if key != 'teacher'})
                      for row in student_meta]
        verify_pairs(originals, students, original_meta, comparable)
        for before, student, teacher, predictions in zip(original_meta, student_meta, teacher_meta, students,
                                                         strict=True):
            expected_teacher = dict(teacher_run=Path(teacher['config']['output']).name,
                                    revision=teacher['upstream_revision'])
            if student['teacher'] != expected_teacher:
                raise ValueError('Student targets must come from the corresponding RM-Net model.')
            for name in names:
                arm = STUDENTS[name]
                baseline = 'independent_intensity' if name == 'independent' else 'intensity'
                if student['selection'][arm]['parameters'] != before['selection'][baseline]['parameters']:
                    raise ValueError(f'{arm} has a different size from its original counterpart.')
                np.testing.assert_array_equal(predictions[arm][-1], predictions['anchor'][-1], err_msg=arm)
        for name in names:
            arm = STUDENTS[name]
            variants.append((name, arm, students))
            parameters[name] = student_meta[0]['selection'][arm]['parameters'] + anchor_count
            epochs[name] = [row['selection'][arm]['epoch'] for row in student_meta]
    evidence = {key: reference[key] for key in ('ids', 'groups', 'y', 'sentiment')}
    for name, arm, packets in variants:
        score = 'score' if name == 'teacher' else f'{arm}_score'
        evidence[name] = np.stack([packet[arm][1:3] for packet in packets])
        evidence[f'{name}_score'] = np.stack([packet[score][1:3] for packet in packets])
    evidence['metadata'] = np.asarray(json.dumps(dict(
        splits=original_meta[0]['splits'], teacher_revision=teacher_meta[0]['upstream_revision'],
        parameter_count=parameters, student_selected_epochs=epochs,
    )))
    return evidence


def summarize(evidence, resamples):
    y, scores, groups = (evidence[key] for key in ('y', 'sentiment', 'groups'))
    _, membership = np.unique(groups, return_inverse=True)
    metadata = json.loads(evidence['metadata'].item())
    methods, rps, mae, error = {}, {}, {}, {}
    for method in METHODS:
        p = evidence[method]
        record, rps[method], mae[method] = summarize_forecasts(p, evidence[f'{method}_score'], y, scores)
        error[method] = (p.argmax(axis=-1) != y).mean(axis=1)
        record['parameters'] = metadata['parameter_count'][method]
        signed, neutral = ((coordinates(p) - outcomes(y)) ** 2).mean(axis=(0, 1, 2)) / 4
        record.update(rps_signed=float(signed), rps_neutral=float(neutral))
        np.testing.assert_allclose(record['rps'], signed + neutral, atol=2e-8)
        methods[method] = record

    chosen = 'anchored_native'
    n_cohorts = len(COHORTS)
    families = {}
    for name, loss, comparators in [('primary_rps', rps, PRIMARY), ('secondary_rps', rps, SECONDARY),
                                    ('mae', mae, (*PRIMARY, *SECONDARY)),
                                    ('accuracy', error, (*PRIMARY, *SECONDARY))]:
        families[name] = {reference: paired_reduction(loss[reference] - loss[chosen], membership,
                                                     resamples, n_cohorts * len(comparators))
                          for reference in comparators}
    return dict(**metadata, methods=methods, **families)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', type=Path)
    source.add_argument('--evidence', type=Path)
    parser.add_argument('--write-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', default=20_000, type=int)
    args = parser.parse_args()
    if args.resamples < 1 or (args.evidence and args.write_evidence):
        parser.error('Use a positive resample count and write evidence only from training runs.')
    if args.runs:
        evidence = {cohort: collect(args.runs, cohort) for cohort in COHORTS}
    else:
        with np.load(args.evidence, allow_pickle=False) as packet:
            np.testing.assert_array_equal(packet['seeds'], SEEDS)
            evidence = {cohort: {key.removeprefix(f'{cohort}__'): packet[key] for key in packet.files
                                 if key.startswith(f'{cohort}__')} for cohort in COHORTS}
    results = {cohort: summarize(corpus, args.resamples) for cohort, corpus in evidence.items()}
    if args.write_evidence:
        packet = {f'{name}__{key}': value for name, corpus in evidence.items() for key, value in corpus.items()}
        temporary = args.write_evidence.with_suffix('.npz.part')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, seeds=np.asarray(SEEDS), **packet)
        temporary.replace(args.write_evidence)
    summary = dict(seeds=list(SEEDS), resamples=args.resamples, cohorts=results)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for cohort, result in results.items():
        for family in ('primary_rps', 'secondary_rps'):
            for reference, contrast in result[family].items():
                lower, upper = contrast['ci95_familywise']
                print(f'{cohort} {reference}: {contrast["reduction"]:+.5f} [{lower:+.5f}, {upper:+.5f}]')


if __name__ == '__main__':
    main()
