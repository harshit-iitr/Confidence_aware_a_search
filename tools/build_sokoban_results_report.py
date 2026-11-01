"""Build the requested three tables, preserving the live-log snapshot provenance."""
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.sokoban7_protocol import METHODS, fingerprint, load_records, sha256

ARCHIVED = [ROOT / 'results/sokoban7_original' / name for name in (
    'maps001_005', 'maps006_010', 'maps011_100_session1')]
events_path = ROOT / 'results/sokoban7_original/snapshot_82maps/live_events.json'
text = ''.join(event.get('data', '') for event in json.loads(events_path.read_text()))
manifest = json.loads((ARCHIVED[0] / 'manifest.json').read_text())
protocol = re.search(r'protocol=([0-9a-f]{64})', text)
assert protocol and protocol[1] == fingerprint(manifest)
methods = [m for m in METHODS if 'gbfs' not in m]
starts, live = {}, {}
for line in text.splitlines():
    m = re.fullmatch(r'START map=(\d+)/100 method=(\S+) seed=(\d+)', line.strip())
    if m:
        key = int(m[1]), m[2]
        expected = int(fingerprint([manifest['base_seed'], key[0], key[1]])[:8], 16)
        assert int(m[3]) == expected
        starts[key] = int(m[3])
    m = re.fullmatch(r'END map=(\d+) method=(\S+) solved=(True|False) exp=(\d+) time=([0-9.]+)s', line.strip())
    if m:
        key = int(m[1]), m[2]
        assert key not in live and key in starts
        live[key] = {'map_id': key[0], 'method': key[1], 'solved': m[3] == 'True',
                     'expansions': int(m[4]), 'time_sec': float(m[5]),
                     'source': 'completed live-log END record; raw plan/cost not yet exported'}
records = [r for folder in ARCHIVED for r in load_records(folder / 'results.jsonl') if r['method'] in methods]
assert len(records) == 405
assert all((i, m) in live for i in range(46, 83) for m in methods)
records += [live[i, m] for i in range(46, 83) for m in methods]
assert len(records) == 738 and len({(r['map_id'], r['method']) for r in records}) == 738
assert sorted({r['map_id'] for r in records}) == list(range(1, 83))


def summarize(rows, cost=True):
    solved = [r for r in rows if str(r['solved']).lower() == 'true']
    e = [float(r['expansions']) for r in solved]
    return {'evaluated': len(rows), 'solved': len(solved),
            'mean_expansions': float(np.mean(e)), 'median_expansions': float(np.median(e)),
            'p90_expansions': float(np.percentile(e, 90)),
            'mean_cost': float(np.mean([float(r.get('cost', r.get('path_cost'))) for r in solved])) if cost else None}


historical = {3: [], 5: []}
sources = []
specs = [
    (3, 'results/3box_rigorous/rigorous_benchmark_results.csv', 'XPU / A', [
        'Classical A* (Manhattan)', 'Paper Learned A*', 'Paper Learned GBFS',
        'Fixed 50/50 Hybrid A*', 'Fixed 50/50 Hybrid GBFS',
        'MC Confidence-Aware A* — paper model', 'MC Confidence-Aware GBFS — paper model']),
    (3, 'results/ensemble_3box_benchmark_results.csv', 'XPU / B', [
        'Single Scratch Model #0 A*', 'Normal 5-Model Ensemble A*', 'Confidence-Aware Ensemble A* (conf_20)']),
    (5, 'results/5box_ood/sokoban_5box_benchmark_results.csv', 'XPU / D', [
        'Classical A* (Manhattan)', 'Paper Learned A*', 'Paper Learned GBFS',
        'MC Confidence-Aware A* — paper model', 'MC Confidence-Aware GBFS — paper model']),
    (5, 'results/ensemble_5box_benchmark_results.csv', 'XPU / B', [
        'Single Scratch Model #0 A*', 'Normal 5-Model Ensemble A*', 'Confidence-Aware Ensemble A* (conf_20)']),
    (5, 'results/calibrated_5box_experiment_results.csv', 'XPU / B', [
        'MC-Dropout on Scratch Model #0', 'Calibrated Confidence-Aware Ensemble A* (conf_50)']),
    (5, 'results/riskaverse_5box_benchmark_results.csv', 'RTX 3050 / B', [
        'Normal 5-Model Ensemble A*', 'Risk-Averse Ensemble A* (kappa=1)',
        'Calibrated Confidence-Aware Ensemble A* (conf_50)']),
]
for boxes, path, setup, labels in specs:
    sources.append(path)
    rows = list(csv.DictReader((ROOT / path).open(encoding='utf-8-sig', newline='')))
    names = list(dict.fromkeys(r['algorithm'] for r in rows))
    assert len(names) == len(labels)
    for name, label in zip(names, labels):
        group = [r for r in rows if r['algorithm'] == name]
        assert len(group) == (200 if boxes == 3 else 100)
        historical[boxes].append({'method': label, 'setup': ('CPU' + setup[3:]) if name == 'Classical A*' else setup,
                                  **summarize(group)})
labels7 = {
    'classical_astar': 'Classical A* (Manhattan)', 'paper_astar': 'Paper Learned A*',
    'paper_mc_astar': 'MC Confidence-Aware A* — paper model',
    'single_astar': 'Single Scratch Model #0 A*', 'single_mc_astar': 'MC-Dropout on Scratch Model #0',
    'ensemble_mean': 'Normal 5-Model Ensemble A*',
    'ensemble_conf_020': 'Confidence-Aware Ensemble A* (conf_20)',
    'ensemble_conf_050': 'Calibrated Confidence-Aware Ensemble A* (conf_50)',
    'ensemble_risk': 'Risk-Averse Ensemble A* (kappa=1)',
}
seven = [{'method_id': method, 'method': labels7[method],
          **summarize([r for r in records if r['method'] == method], cost=False)} for method in methods]
paired = []
for first, baseline in [('paper_mc_astar', 'paper_astar'), ('ensemble_risk', 'ensemble_mean'),
                         ('ensemble_conf_020', 'ensemble_mean'), ('ensemble_conf_050', 'ensemble_mean'),
                         ('single_mc_astar', 'single_astar')]:
    a = {r['map_id']: r for r in records if r['method'] == first and r['solved']}
    b = {r['map_id']: r for r in records if r['method'] == baseline and r['solved']}
    common = sorted(a.keys() & b.keys())
    x, y = [a[i]['expansions'] for i in common], [b[i]['expansions'] for i in common]
    paired.append({'method': first, 'baseline': baseline, 'common_solved': len(common),
                   'wins': sum(u < v for u, v in zip(x, y)), 'ties': sum(u == v for u, v in zip(x, y)),
                   'losses': sum(u > v for u, v in zip(x, y)),
                   'mean_reduction_pct': 100 * (1 - float(np.mean(x)) / float(np.mean(y)))})

md = '''# Sokoban results — 3-box, 5-box and 7-box

Reporting cutoff: **maps 1–82 for seven-box Sokoban**, chosen at the user's request on 1 November 2026. No later maps are included in these tables. The seven-box study is a partial 82/100 evaluation, not a completed 100-map benchmark.

**Data status:** 3/5-box values come from the original repository's full benchmark CSVs, excluding recent conversion reruns and small exploratory pilots. Seven-box values combine downloaded raw results for maps 1–45 with completed live-log records for maps 46–82. The active Kaggle job had reached map 83 when sampled; its final archive was not yet available. All 738 A* attempts through map 82 are represented. The live log's protocol fingerprint and per-attempt seeds match the archived runs. Saved plans for the latest 37 maps have not yet been downloaded and independently replayed locally; solution costs are therefore omitted from the seven-box table. This file is a completed reporting snapshot, not a claim that the active cloud session was stopped or archived.

**Reading the tables:** solve rates include all evaluated maps. Mean, median and P90 expansions, and mean solution cost, use each method's solved maps. Different solved sets make these expansion averages unsuitable as paired comparisons; paired findings are given below. Method names follow the original repo, including its historical “Paper Learned” label; this is not a verified paper-reproduction claim. GBFS is retained only in historical 3/5-box results; seven-box uses A* only.

**Setup keys:** A = 600 seconds / 100,000 expansions; B = 360 seconds / 1,000,000 expansions; D = historical 600 seconds, with configured ceilings of 15,000 neural / 150,000 classical expansions. Historical loop behavior can overshoot. XPU = Intel Arc 130V. RTX 3050 = NVIDIA RTX 3050 Laptop GPU. Seven-box uses Tesla T4 (classical search remains on CPU) and budget B for every method. These are separate experiment groups, not one uniform hardware/budget comparison across domains.
'''
for boxes in (3, 5):
    md += f'\n## {boxes}-box Sokoban — {"IID, 200 maps" if boxes == 3 else "OOD, 100 maps"}\n\n'
    md += '| Method | Setup | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost |\n'
    md += '|---|---|---:|---:|---:|---:|---:|\n'
    for row in historical[boxes]:
        rate = 100 * row['solved'] / row['evaluated']
        md += f'| {row["method"]} | {row["setup"]} | {row["solved"]}/{row["evaluated"]} ({rate:.1f}%) | '
        md += ' | '.join(f'{row[k]:,.1f}' for k in ('mean_expansions', 'median_expansions', 'p90_expansions', 'mean_cost')) + ' |\n'
md += '\n## 7-box Sokoban — OOD, 82 completed maps\n\n'
md += 'Same nine A* configurations, frozen maps 1–82, Tesla T4, 360 seconds / 1,000,000 expansions. Latest 37 maps are a live-log snapshot pending raw artifact export.\n\n'
md += '| Method | Solved | Mean exp. | Median exp. | P90 exp. |\n|---|---:|---:|---:|---:|\n'
for row in seven:
    md += f'| {row["method"]} | {row["solved"]}/82 ({100*row["solved"]/82:.1f}%) | '
    md += ' | '.join(f'{row[k]:,.1f}' for k in ('mean_expansions', 'median_expansions', 'p90_expansions')) + ' |\n'
md += '\n### Seven-box paired comparisons\n\n'
for p in paired:
    md += f'- **{labels7[p["method"]]} vs {labels7[p["baseline"]]}:** {p["common_solved"]} mutually solved maps; {p["wins"]} wins / {p["ties"]} ties / {p["losses"]} losses; {p["mean_reduction_pct"]:.2f}% mean expansion reduction (negative means more expansions).\n'
md += '\n### Reporting closeout\n\nThe requested report stops at 82 maps. Maps 83–100 are outside this snapshot. No further Kaggle sessions were launched for this report. The existing session was still running, so its safe export remains a separate outstanding step; it was not cancelled while its archive was unavailable. All historical and live-log evidence is preserved. Final solution-length comparisons require the latest raw artifact.\n'
md += '\n### Limitation: MC dropout on scratch M0\n\n'
md += 'MC confidence-aware search is model-dependent. On five-box problems, scratch M0 solves 93/100 while its MC variant solves 77/100. In this seven-box snapshot, those counts are 65/82 and 53/82. On the 53 mutually solved seven-box maps, scratch MC uses approximately 3.6 times as many mean expansions. This cannot be explained by extra forward-pass time alone.\n'
md += '\nPossible explanations include sensitivity to inference-time dropout, degraded prediction rankings, unreliable disagreement, and interaction with classical blending. These are hypotheses, not established causes. The comparison changes dropout sampling, percentile scaling, and confidence blending together; isolating their effects requires separate ablations. No claim is made that disagreement predicts physical distance error.\n'
md += '\n### Sources\n\n'
md += ''.join(f'- `{path}`\n' for path in sources)
md += '- `results/sokoban7_original/maps001_005/`\n- `results/sokoban7_original/maps006_010/`\n- `results/sokoban7_original/maps011_100_session1/`\n'
md += '- `results/sokoban7_original/snapshot_82maps/live_events.json` (completed END records only, through map 82)\n'
md += '- `COMPILED_SOKOBAN_RESULTS.md` retains the earlier exploratory pilots separately.\n'
snapshot = ROOT / 'results/sokoban7_original/snapshot_82maps'
snapshot.mkdir(parents=True, exist_ok=True)
payload = {'cutoff_maps': list(range(1, 83)), 'astar_attempts': 738,
           'archived_maps': list(range(1, 46)), 'live_log_maps': list(range(46, 83)),
           'raw_latest_locally_replayed': False, 'live_events_sha256': sha256(events_path),
           'protocol_id': fingerprint(manifest), 'summary': seven, 'paired': paired,
           'cloud_session_state_when_sampled': 'RUNNING', 'new_sessions_launched': False}
(snapshot / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
(snapshot / 'live_46_82_summary_records.json').write_text(json.dumps(
    [live[i, m] for i in range(46, 83) for m in methods], indent=2), encoding='utf-8')
output = ROOT / 'SOKOBAN_RESULTS_3_5_7_BOX.md'
assert md.count('|---|') == 3
output.write_text(md, encoding='utf-8')
print('Saved:', output)
print('Table rows:', len(historical[3]), len(historical[5]), len(seven))
for row in seven:
    print(row['method_id'], f'{row["solved"]}/82', f'{100*row["solved"]/82:.1f}%',
          f'mean={row["mean_expansions"]:.1f}', f'median={row["median_expansions"]:.1f}')
print('Paired:', json.dumps(paired))
