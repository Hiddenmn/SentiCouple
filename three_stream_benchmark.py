"""Compare SentiCouple and UCMR-Net with text, audio and vision."""

import argparse
import json
from pathlib import Path

from ablate import read_predictions
from benchmark_mosei import ROOT, SEEDS
from mosei_results import profile_all, summarize_benchmark


COHORTS = {
    'sims_v2': 'sims_v2_shared_score',
    'mosei': 'mosei_three_stream',
}


def evaluate(runs, cohort, resamples, profile_upstream=None):
    shared = [runs / f'{COHORTS[cohort]}_seed{seed}' for seed in SEEDS]
    reference = [runs / f'{cohort}_three_stream_ucmr_net_seed{seed}' for seed in SEEDS]
    if not read_predictions(shared[0])['masks'][1:3, 0].any():
        raise ValueError('The comparison requires observed audio in the evaluation views.')
    paths = {'SentiCouple': shared, 'ucmr-net': reference}
    result = summarize_benchmark(paths, SEEDS, resamples)
    if profile_upstream is not None:
        if cohort == 'sims_v2':
            data = ROOT / 'data/ch_sims_v2_unaligned.pkl'
            group_reference = ROOT / 'data/ch_sims_unaligned_39.pkl'
            eval_groups = None
        else:
            data = ROOT / 'data/mosei_aligned_50.pkl'
            group_reference = None
            eval_groups = ROOT / 'splits/mosei_eval_videos.txt'
        result['timing'] = profile_all(paths, {'ucmr-net': profile_upstream}, data, eval_groups,
                                       cohort=cohort, group_reference=group_reference, visual_only=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs')
    parser.add_argument('--cohort', choices=tuple(COHORTS), required=True)
    parser.add_argument('--resamples', type=int, default=20000)
    parser.add_argument('--profile-upstream', type=Path, help='Time seed 97 on CUDA using the pinned UCMR-Net checkout.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error('At least one resample is required.')
    result = evaluate(args.runs, args.cohort, args.resamples, args.profile_upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
