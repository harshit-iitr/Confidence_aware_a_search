"""Seven-box transfer using original repository models and five-box search code."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# Original modules mix flat and src-qualified imports. Preserve those imports.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from benchmarks.sokoban7_protocol import (
    TIMEOUT, MAX_EXPANSIONS, METHODS, SOURCE_FILES, CHECKPOINTS,
    sha256, fingerprint, read_boards, replay_plan, smoke_board, load_records,
)
from benchmarks.run_5box_benchmark import run_batched_paper_search, run_classical_astar
from benchmarks.run_ensemble_comparison_benchmark import run_single_scratch_astar
from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.confidence_aware_search import (
    author_state_to_tensor, get_device, get_device_name,
    ConfidenceAwareMCHeuristic, run_confidence_aware_astar, run_confidence_aware_gbfs,
)
from src.ensemble_search import DeepEnsembleHeuristic, run_ensemble_astar


def dispatch(method, state, paper, models, device, *, smoke=False):
    """Pass identical budgets into the original functions; do not rewrite search."""
    targets = SokobanEnv.get_box_targets(state)
    goal = SokobanEnv.get_goal_state(state, targets)
    common = dict(max_time=10.0 if smoke else TIMEOUT,
                  max_expansions=100 if smoke else MAX_EXPANSIONS, dim=10)
    paper.eval()
    for model in models:
        model.eval()
    if method == 'classical_astar':
        result = run_classical_astar(state, targets, **common)
    elif method in ('paper_astar', 'paper_gbfs', 'single_astar'):
        goal_tensor = torch.from_numpy(author_state_to_tensor(goal, targets, 10)).unsqueeze(0).to(device)
        if method == 'single_astar':
            result = run_single_scratch_astar(state, targets, models[0], goal_tensor, device, **common)
        else:
            result = run_batched_paper_search(
                state, targets, paper, goal_tensor, device,
                alg='gbfs' if method == 'paper_gbfs' else 'astar', **common)
    elif method in ('paper_mc_astar', 'paper_mc_gbfs', 'single_mc_astar'):
        heuristic = ConfidenceAwareMCHeuristic(
            model=models[0] if method == 'single_mc_astar' else paper,
            goal_state=goal, box_targets=targets, device=device,
            num_mc_samples=5, dropout_p=0.10, lambda_min=0.20,
            scale_constant_C=50.0, dim=10)
        search = run_confidence_aware_gbfs if method == 'paper_mc_gbfs' else run_confidence_aware_astar
        result = search(state, targets, heuristic, **common)
    elif method in ('ensemble_mean', 'ensemble_conf_020', 'ensemble_conf_050', 'ensemble_risk'):
        heuristic = DeepEnsembleHeuristic(
            models=models, goal_state=goal, box_targets=targets, device=device,
            lambda_min=0.50 if method == 'ensemble_conf_050' else 0.20,
            beta=2.5, gating_mode='std',
            search_mode='risk_averse' if method == 'ensemble_risk' else 'convex',
            fallback_type='manhattan', kappa=1.0, scale_constant_C=50.0, dim=10)
        result = run_ensemble_astar(
            state, targets, heuristic, fixed_lambda=1.0 if method == 'ensemble_mean' else None,
            prune_deadlocks=False, **common)
    else:
        raise ValueError(f'Unknown method: {method}')
    return tuple(result[:5]) + (result[5] if len(result) == 6 else {},)


def seed_for(base, map_id, method):
    # Stable across processes, shards, method subsets, and resumes.
    return int(fingerprint([base, map_id, method])[:8], 16)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def make_manifest(args, device):
    dataset = args.dataset.resolve()
    provenance = dataset.with_suffix('.provenance.json')
    if not args.smoke_test:
        meta = json.loads(provenance.read_text())
        if meta['dataset_sha256'] != sha256(dataset) or meta['count'] != 100:
            raise ValueError('Dataset does not match frozen generation provenance')
    return {
        'schema': 1, 'protocol': 'original-repository-7box-v1',
        'kind': 'smoke-only-one-box' if args.smoke_test else '7-box-test',
        'dataset_sha256': None if args.smoke_test else sha256(dataset),
        'provenance_sha256': None if args.smoke_test else sha256(provenance),
        'num_maps': 1 if args.smoke_test else 100,
        'timeout_sec': 10.0 if args.smoke_test else TIMEOUT,
        'max_expansions': 100 if args.smoke_test else MAX_EXPANSIONS,
        'base_seed': args.seed, 'method_order': list(METHODS), 'methods': METHODS,
        'parameters': {'mc_samples': 5, 'dropout_p': 0.10, 'mc_lambda_min': 0.20,
                       'ensemble_lambda_min': [0.20, 0.50], 'beta': 2.5,
                       'kappa': 1.0, 'C': 50.0, 'gating_mode': 'std',
                       'uncertainty_space': 'per-member online percentile',
                       'symbolic_deadlock_pruning': False},
        'checkpoints': {p: sha256(ROOT / p) for p in CHECKPOINTS},
        'source_sha256': {p: sha256(ROOT / p) for p in SOURCE_FILES},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'numpy': np.__version__, 'device': str(device),
                    'torch_threads': torch.get_num_threads(),
                    'torch_interop_threads': torch.get_num_interop_threads(),
                    'float32_matmul_precision': torch.get_float32_matmul_precision(),
                    'hardware': get_device_name(device), 'platform': platform.platform()},
        'notes': ['Original checkpoint retained; not a verified paper reproduction.',
                  'Original search loops and expansion counters retained.',
                  'Budgets are checked at original loop boundaries; small overruns are possible.',
                  'Search time excludes wrapper construction; wall_time_sec includes dispatch setup.',
                  'No new training, conversion, goal anchoring, or symbolic pruning.'],
    }


def validate_record(row, manifest, boards):
    if row['protocol_id'] != fingerprint(manifest):
        raise ValueError('Result fingerprint differs from manifest')
    map_id, method = row['map_id'], row['method']
    if not 1 <= map_id <= len(boards) or method not in METHODS:
        raise ValueError('Unknown map or method in results')
    expected_seed = seed_for(manifest['base_seed'], map_id, method)
    if row['seed'] != expected_seed:
        raise ValueError('Unexpected per-map/method seed')
    if row['solved']:
        if not row['verified'] or row['cost'] != len(row['actions']):
            raise ValueError('Solved record has inconsistent plan metadata')
        if not replay_plan(boards[map_id - 1], row['actions']):
            raise ValueError('Saved plan failed independent replay')
    elif row['actions'] or row['cost'] is not None:
        raise ValueError('Failed record contains a solution')


def load_models(device):
    models = []
    for relative in CHECKPOINTS:
        model = ChrestienHeuristicNet(dim=10).to(device)
        model.load_state_dict(torch.load(ROOT / relative, map_location=device, weights_only=True), strict=True)
        model.eval()
        models.append(model)
    return models[0], models[1:]


def run(args):
    started = time.monotonic()
    device = get_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; enable a GPU runtime')
    boards = [smoke_board()] if args.smoke_test else read_boards(args.dataset)
    last = args.end_map or len(boards)
    if not 1 <= args.start_map <= last <= len(boards):
        raise ValueError('Invalid one-based map range')
    if args.wall_hours <= 0:
        raise ValueError('wall-hours must be positive')
    selected = args.methods or list(METHODS)
    manifest = make_manifest(args, device)
    protocol_id = fingerprint(manifest)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.run.lock'
    # Separate jobs must use separate output folders, even for different maps.
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        mp = out / 'manifest.json'
        if mp.exists():
            if json.loads(mp.read_text()) != manifest:
                raise ValueError('Resume refused: dataset, source, model, seed, budget, or runtime changed')
        else:
            if (out / 'results.jsonl').exists():
                raise ValueError('Results exist without manifest')
            mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        previous = load_records(out / 'results.jsonl')
        for row in previous:
            validate_record(row, manifest, boards)
        done = {(r['map_id'], r['method']) for r in previous}
        pending = [(i, m) for i in range(args.start_map, last + 1)
                   for m in selected if (i, m) not in done]
        print(f'{get_device_name(device)} | {len(pending)} pending attempts | protocol={protocol_id}', flush=True)
        if not pending:
            report([out], out, boards=boards)
            return
        paper, models = load_models(device)
        stopped = False
        with (out / 'results.jsonl').open('a', encoding='utf-8', newline='\n') as f:
            for map_id, method in pending:
                if time.monotonic() - started + manifest['timeout_sec'] + 15 > args.wall_hours * 3600:
                    stopped = True
                    print('Session wall allowance reached; resume this directory in the same environment.', flush=True)
                    break
                seed = seed_for(args.seed, map_id, method)
                set_seed(seed)
                print(f'START map={map_id:03d}/{len(boards)} method={method} seed={seed}', flush=True)
                before = time.monotonic()
                solved, expansions, cost, actions, seconds, stats = dispatch(
                    method, boards[map_id - 1], paper, models, device, smoke=args.smoke_test)
                actions = list(map(int, actions))
                row = {
                    'protocol_id': protocol_id, 'map_id': map_id, 'method': method,
                    'seed': seed, 'solved': bool(solved), 'expansions': int(expansions),
                    'cost': float(cost) if solved else None, 'actions': actions,
                    'time_sec': float(seconds), 'wall_time_sec': time.monotonic() - before,
                    'verified': bool(solved and replay_plan(boards[map_id - 1], actions)),
                    'mean_lambda': stats.get('mean_lambda'),
                    'mean_variance': stats.get('mean_variance'),
                }
                validate_record(row, manifest, boards)
                f.write(json.dumps(row, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                print(f'END map={map_id:03d} method={method} solved={solved} exp={expansions} time={seconds:.3f}s', flush=True)
        report([out], out, boards=boards)
        (out / 'session.json').write_text(json.dumps({
            'finished_utc': datetime.now(timezone.utc).isoformat(),
            'wall_budget_stop': stopped, 'selected_methods': selected,
            'requested_range': [args.start_map, last],
            'session_wall_sec': time.monotonic() - started,
        }, indent=2), encoding='utf-8')
    finally:
        lock.unlink()


def report(inputs, output, *, dataset=None, boards=None):
    """Merge only matching protocols; never silently pool different GPUs or seeds."""
    manifest, rows, seen = None, [], set()
    for folder in map(Path, inputs):
        candidate = json.loads((folder / 'manifest.json').read_text())
        if manifest is not None and manifest != candidate:
            raise ValueError('Cannot merge runs with different manifests (including hardware/runtime)')
        if manifest is None and boards is None:
            if candidate['kind'] == 'smoke-only-one-box':
                boards = [smoke_board()]
            else:
                dataset_path = Path(dataset) if dataset else ROOT / 'data/sokoban_7box_original_test.txt'
                if sha256(dataset_path) != candidate['dataset_sha256']:
                    raise ValueError('Report dataset differs from evaluated dataset')
                boards = read_boards(dataset_path)
        if len(boards) != candidate['num_maps']:
            raise ValueError('Report board count differs from manifest')
        manifest = candidate
        for row in load_records(folder / 'results.jsonl'):
            validate_record(row, manifest, boards)
            key = row['map_id'], row['method']
            if key in seen:
                raise ValueError(f'Overlapping shard record: {key}')
            if row['protocol_id'] != fingerprint(manifest):
                raise ValueError('Record fingerprint mismatch')
            if row['method'] not in METHODS or not 1 <= row['map_id'] <= manifest['num_maps']:
                raise ValueError('Unknown method or map in shard')
            rows.append(row)
            seen.add(key)
    complete = [i for i in range(1, manifest['num_maps'] + 1)
                if all((i, method) in seen for method in METHODS)]
    cohort = [r for r in rows if r['map_id'] in complete]
    def stats(method):
        attempted = [r for r in cohort if r['method'] == method]
        solved = [r for r in attempted if r['solved'] and r['verified']]
        exps = [r['expansions'] for r in solved]
        mean = lambda values: float(np.mean(values)) if values else None
        return {'method': method, 'algorithm': METHODS[method], 'evaluated': len(attempted),
                'solved': len(solved), 'mean_expansions': mean(exps),
                'median_expansions': float(np.median(exps)) if exps else None,
                'p90_expansions': float(np.percentile(exps, 90)) if exps else None,
                'mean_cost': mean([r['cost'] for r in solved]),
                'mean_time_solved_sec': mean([r['time_sec'] for r in solved]),
                'mean_time_all_sec': mean([r['time_sec'] for r in attempted]),
                'mean_lambda': mean([r['mean_lambda'] for r in solved if r['mean_lambda'] is not None])}
    summary = [stats(m) for m in METHODS]
    paired = []
    for first, base in [('paper_mc_astar', 'paper_astar'),
                        ('paper_mc_gbfs', 'paper_gbfs'),
                        ('ensemble_risk', 'ensemble_mean'),
                        ('ensemble_conf_020', 'ensemble_mean'),
                        ('ensemble_conf_050', 'ensemble_mean')]:
        a = {r['map_id']: r for r in cohort if r['method'] == first and r['solved'] and r['verified']}
        b = {r['map_id']: r for r in cohort if r['method'] == base and r['solved'] and r['verified']}
        shared = sorted(a.keys() & b.keys())
        x = [a[i]['expansions'] for i in shared]
        y = [b[i]['expansions'] for i in shared]
        paired.append({'method': first, 'baseline': base, 'common_solved': len(shared),
                       'wins': sum(u < v for u, v in zip(x, y)),
                       'ties': sum(u == v for u, v in zip(x, y)),
                       'losses': sum(u > v for u, v in zip(x, y)),
                       'paired_mean_reduction_pct': 100 * (1 - float(np.mean(x)) / float(np.mean(y)))
                       if shared and np.mean(y) else None})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (len(inputs) == 1 and Path(inputs[0]).resolve() == output.resolve()):
        # Keep merged evidence portable; never rewrite the live append-only input.
        merged = {
            'manifest.json': json.dumps(manifest, indent=2),
            'results.jsonl': ''.join(json.dumps(r, allow_nan=False) + '\n' for r in sorted(
                rows, key=lambda r: (r['map_id'], list(METHODS).index(r['method'])))),
        }
        for name, content in merged.items():
            target = output / name
            if target.exists() and target.read_text(encoding='utf-8') != content:
                raise ValueError(f'Merged output already contains different evidence: {target}')
            if not target.exists():
                with target.open('x', encoding='utf-8', newline='\n') as f:
                    f.write(content)
    payload = {'protocol': manifest, 'complete_maps': complete,
               'records': len(rows), 'unbalanced_records': len(rows) - len(cohort),
               'summary': summary, 'paired': paired}
    (output / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    def fmt(value):
        return '—' if value is None else f'{value:,.2f}'
    md = '# 7-box original-repository protocol\n\n'
    md += f'Complete across all 11 configurations: **{len(complete)}/{manifest["num_maps"]} maps**. '
    md += f'Unbalanced records excluded from comparison: {len(rows) - len(cohort)}.\n\n'
    md += f'Budget: {manifest["timeout_sec"]} seconds / {manifest["max_expansions"]:,} expansions per attempt. '
    md += f'Hardware: {manifest["runtime"]["hardware"]}. Kind: {manifest["kind"]}.\n\n'
    md += 'Expansion/cost statistics use each method\'s solved subset; all-run time includes failures. '
    md += 'Use the paired comparisons below to compare expansions on identical solved maps.\n\n'
    md += '| Method | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost | Mean time, all (s) |\n'
    md += '|---|---:|---:|---:|---:|---:|---:|\n'
    for s in summary:
        md += f'| {s["algorithm"]} | {s["solved"]}/{s["evaluated"]} | ' + ' | '.join(fmt(s[k]) for k in (
            'mean_expansions', 'median_expansions', 'p90_expansions', 'mean_cost', 'mean_time_all_sec')) + ' |\n'
    md += '\nPaired mean expansion reduction (positive means fewer expansions):\n\n'
    for p in paired:
        md += f'- {p["method"]} vs {p["baseline"]}: N={p["common_solved"]}; {p["wins"]}W/{p["ties"]}T/{p["losses"]}L; {fmt(p["paired_mean_reduction_pct"])}%.\n'
    (output / 'report.md').write_text(md, encoding='utf-8')
    print(f'Report: {len(complete)}/{manifest["num_maps"]} complete maps -> {output / "report.md"}', flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    r = commands.add_parser('run')
    r.add_argument('--dataset', type=Path, default=ROOT / 'data/sokoban_7box_original_test.txt')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--device', default='auto')
    r.add_argument('--start-map', type=int, default=1)
    r.add_argument('--end-map', type=int)
    r.add_argument('--seed', type=int, default=42)
    r.add_argument('--wall-hours', type=float, default=7.5)
    r.add_argument('--methods', nargs='+', choices=list(METHODS))
    r.add_argument('--smoke-test', action='store_true', help='One-box one-push validation only; not experimental data')
    p = commands.add_parser('report')
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, help='Only needed for a non-default frozen dataset path')
    args = parser.parse_args()
    if args.command == 'run':
        if args.methods and len(set(args.methods)) != len(args.methods):
            parser.error('Duplicate methods are not allowed')
        run(args)
    else:
        report(args.inputs, args.output, dataset=args.dataset)


if __name__ == '__main__':
    main()
