"""Compare only the nine A* configurations; preserve original protocol manifests."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.sokoban7_protocol import METHODS, fingerprint, load_records, read_boards, replay_plan, sha256

ASTAR_METHODS = {key: label for key, label in METHODS.items() if 'gbfs' not in key}


def compile_reports(inputs, output, dataset=None, boards=None):
    manifest, records, seen = None, [], set()
    for folder in map(Path, inputs):
        candidate = json.loads((folder / 'manifest.json').read_text())
        if manifest is not None and candidate != manifest:
            raise ValueError('Different protocol/data/code/runtime manifests; do not pool these runs')
        if boards is None:
            path = Path(dataset) if dataset else ROOT / 'data/sokoban_7box_original_test.txt'
            if sha256(path) != candidate['dataset_sha256']:
                raise ValueError('Frozen dataset hash mismatch')
            boards = read_boards(path)
        if len(boards) != candidate['num_maps']:
            raise ValueError('Dataset size differs from run manifest')
        manifest = candidate
        for row in load_records(folder / 'results.jsonl'):
            key = row['map_id'], row['method']
            if key in seen:
                raise ValueError(f'Overlapping shard: {key}')
            seen.add(key)
            if row['protocol_id'] != fingerprint(manifest):
                raise ValueError('Result protocol fingerprint mismatch')
            if row['method'] not in METHODS or not 1 <= row['map_id'] <= len(boards):
                raise ValueError('Invalid method or map')
            expected_seed = int(fingerprint([manifest['base_seed'], row['map_id'], row['method']])[:8], 16)
            if row['seed'] != expected_seed:
                raise ValueError('Unexpected attempt seed')
            if row['solved']:
                if not row['verified'] or row['cost'] != len(row['actions']):
                    raise ValueError('Inconsistent saved solution')
                if not replay_plan(boards[row['map_id'] - 1], row['actions']):
                    raise ValueError('Saved solution failed independent replay')
            records.append(row)
    complete = [i for i in range(1, manifest['num_maps'] + 1)
                if all((i, method) in seen for method in ASTAR_METHODS)]
    cohort = [r for r in records if r['method'] in ASTAR_METHODS and r['map_id'] in complete]
    mean = lambda values: float(np.mean(values)) if values else None
    summary = []
    for method, label in ASTAR_METHODS.items():
        attempts = [r for r in cohort if r['method'] == method]
        solved = [r for r in attempts if r['solved'] and r['verified']]
        expansions = [r['expansions'] for r in solved]
        summary.append({
            'method': method, 'algorithm': label, 'evaluated': len(attempts), 'solved': len(solved),
            'mean_expansions': mean(expansions),
            'median_expansions': float(np.median(expansions)) if expansions else None,
            'p90_expansions': float(np.percentile(expansions, 90)) if expansions else None,
            'mean_cost': mean([r['cost'] for r in solved]),
            'mean_time_solved_sec': mean([r['time_sec'] for r in solved]),
            'mean_time_all_sec': mean([r['time_sec'] for r in attempts]),
        })
    paired = []
    for method, baseline in [('paper_mc_astar', 'paper_astar'),
                              ('single_mc_astar', 'single_astar'),
                              ('ensemble_risk', 'ensemble_mean'),
                              ('ensemble_conf_020', 'ensemble_mean'),
                              ('ensemble_conf_050', 'ensemble_mean')]:
        a = {r['map_id']: r for r in cohort if r['method'] == method and r['solved']}
        b = {r['map_id']: r for r in cohort if r['method'] == baseline and r['solved']}
        shared = sorted(a.keys() & b.keys())
        x, y = [a[i]['expansions'] for i in shared], [b[i]['expansions'] for i in shared]
        paired.append({'method': method, 'baseline': baseline, 'common_solved': len(shared),
                       'wins': sum(u < v for u, v in zip(x, y)), 'ties': sum(u == v for u, v in zip(x, y)),
                       'losses': sum(u > v for u, v in zip(x, y)),
                       'mean_expansion_reduction_pct': 100 * (1 - mean(x) / mean(y)) if shared and mean(y) else None})
    payload = {'profile': 'nine-astar-configurations', 'required_methods': list(ASTAR_METHODS),
               'protocol': manifest, 'complete_maps': complete, 'total_records': len(records),
               'ignored_gbfs_records': sum(r['method'] not in ASTAR_METHODS for r in records),
               'unbalanced_astar_records': sum(r['method'] in ASTAR_METHODS and r['map_id'] not in complete for r in records),
               'summary': summary, 'paired': paired}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    def fmt(value):
        return '—' if value is None else f'{value:,.2f}'
    md = '# Seven-box Sokoban — A* configurations only\n\n'
    md += f'**{len(complete)}/{manifest["num_maps"]} maps** complete across all nine A* configurations. '
    md += f'Preserved GBFS records excluded: {payload["ignored_gbfs_records"]}.\n\n'
    md += 'All methods: 360 seconds / 1,000,000 expansions. Expansions/cost use solved maps; time includes failures.\n\n'
    md += '| Method | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost | Mean time, all (s) |\n'
    md += '|---|---:|---:|---:|---:|---:|---:|\n'
    for s in summary:
        md += f'| {s["algorithm"]} | {s["solved"]}/{s["evaluated"]} | ' + ' | '.join(fmt(s[k]) for k in (
            'mean_expansions', 'median_expansions', 'p90_expansions', 'mean_cost', 'mean_time_all_sec')) + ' |\n'
    md += '\nPaired comparisons on mutually solved maps:\n\n'
    for p in paired:
        md += f'- {p["method"]} vs {p["baseline"]}: N={p["common_solved"]}; {p["wins"]}W/{p["ties"]}T/{p["losses"]}L; mean expansion reduction {fmt(p["mean_expansion_reduction_pct"])}%.\n'
    (output / 'report.md').write_text(md, encoding='utf-8')
    print(f'A* comparison: {len(complete)}/{manifest["num_maps"]} complete maps; {output / "report.md"}', flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    compile_reports(args.inputs, args.output, args.dataset)
