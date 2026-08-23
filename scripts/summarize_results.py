#!/usr/bin/env python3
"""Summarize the locked baseline/sequence protocol and paired task-4 rerun."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _latest_rows(root: Path, group: str) -> dict[int, dict[str, float]]:
    rows = {}
    for path in sorted((root / group).glob('sd*/eval.csv')):
        with path.open() as file:
            row = list(csv.DictReader(file))[-1]
        seed = int(path.parent.name[2:5])
        rows[seed] = {key: float(value) for key, value in row.items() if key != 'step'}
    return rows


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denom
    return [center - margin, center + margin]


def _paired_outcomes(root: Path, group: str) -> dict[tuple[int, int, int], int]:
    outcomes = {}
    for path in sorted((root / group).glob('sd*/episode_outcomes.jsonl')):
        seed = int(path.parent.name[2:5])
        with path.open() as file:
            for line in file:
                record = json.loads(line)
                key = (seed, int(record['task_id']), int(record['episode']))
                outcomes[key] = int(record['success'] > 0.0)
    return outcomes


def _exact_mcnemar(baseline: dict, sequence: dict) -> dict[str, float | int]:
    shared = sorted(set(baseline) & set(sequence))
    improved = sum(baseline[key] == 0 and sequence[key] == 1 for key in shared)
    regressed = sum(baseline[key] == 1 and sequence[key] == 0 for key in shared)
    discordant = improved + regressed
    if discordant:
        lower_tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
        p_value = min(1.0, 2.0 * lower_tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        'pairs': len(shared),
        'improved': improved,
        'regressed': regressed,
        'discordant': discordant,
        'exact_two_sided_p': p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', type=Path, default=Path('experiments'))
    args = parser.parse_args()

    groups = {
        'baseline': _latest_rows(args.experiments, 'final_baseline'),
        'sequence': _latest_rows(args.experiments, 'final_sequence'),
    }
    seeds = sorted(set(groups['baseline']) & set(groups['sequence']))
    if not seeds:
        raise SystemExit('No paired final runs found.')

    task_keys = [f'evaluation/{task}_success' for task in range(1, 6)]
    result = {'seeds': seeds, 'methods': {}, 'paired_seed_delta': {}}
    for method, rows in groups.items():
        seed_scores = [rows[seed]['evaluation/overall_success'] for seed in seeds]
        task_means = {
            str(task): statistics.mean(rows[seed][key] for seed in seeds)
            for task, key in enumerate(task_keys, start=1)
        }
        successes = round(sum(rows[seed][key] * 20 for seed in seeds for key in task_keys))
        result['methods'][method] = {
            'seed_scores': seed_scores,
            'mean': statistics.mean(seed_scores),
            'sample_std': statistics.stdev(seed_scores),
            'task_means': task_means,
            'successes': successes,
            'trials': 300,
            'wilson_95': _wilson(successes, 300),
        }

    deltas = [
        groups['sequence'][seed]['evaluation/overall_success']
        - groups['baseline'][seed]['evaluation/overall_success']
        for seed in seeds
    ]
    delta_mean = statistics.mean(deltas)
    delta_se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    # Student-t 0.975 quantile for df=2, since the locked protocol has 3 seeds.
    t_critical = 4.302652729911275
    result['paired_seed_delta'] = {
        'values': deltas,
        'mean': delta_mean,
        'sample_std': statistics.stdev(deltas),
        't_95': [delta_mean - t_critical * delta_se, delta_mean + t_critical * delta_se],
    }

    for method, rows in groups.items():
        task4_successes = round(sum(rows[seed]['evaluation/4_success'] * 20 for seed in seeds))
        result['methods'][method]['task4_successes'] = task4_successes
        result['methods'][method]['task4_trials'] = 60
        result['methods'][method]['task4_wilson_95'] = _wilson(task4_successes, 60)

    baseline_pairs = _paired_outcomes(args.experiments, 'confirm_task4_baseline')
    sequence_pairs = _paired_outcomes(args.experiments, 'confirm_task4_sequence')
    if baseline_pairs and sequence_pairs:
        result['paired_task4_mcnemar'] = _exact_mcnemar(baseline_pairs, sequence_pairs)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
