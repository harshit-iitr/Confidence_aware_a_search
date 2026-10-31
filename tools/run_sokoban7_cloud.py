"""Bounded cloud job: freeze all maps before search; preserve data and results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/sokoban7_original_output'))
    parser.add_argument('--start-map', type=int, default=1)
    parser.add_argument('--end-map', type=int, default=5)
    parser.add_argument('--wall-hours', type=float, default=7.5)
    args = parser.parse_args()
    if not 1 <= args.start_map <= args.end_map <= 100 or args.wall_hours <= 0:
        raise ValueError('Invalid cloud map range or wall budget')
    deadline = time.monotonic() + args.wall_hours * 3600
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / 'data/sokoban_7box_original_test.txt'
    status = {'phase': 'setup', 'range': [args.start_map, args.end_map],
              'started_utc': datetime.now(timezone.utc).isoformat()}
    def update(phase):
        status['phase'] = phase
        (output / 'cloud_session.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print('Cloud phase:', phase, flush=True)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable; enable CUDA before launching this job')
        print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, flush=True)
        if not dataset.exists():
            update('generating-100-fixed-maps-before-search')
            subprocess.run([sys.executable, '-u', str(ROOT / 'tools/generate_sokoban7_original.py')],
                           cwd=ROOT, check=True, timeout=max(1, deadline - time.monotonic() - 600))
        update('evaluating-original-models')
        remaining = (deadline - time.monotonic()) / 3600
        if remaining <= 0:
            raise TimeoutError('Generation consumed the session; preserved progress can be resumed')
        subprocess.run([
            sys.executable, '-u', str(ROOT / 'benchmarks/run_sokoban7_benchmark.py'),
            'run', '--device', 'cuda:0', '--dataset', str(dataset),
            '--start-map', str(args.start_map), '--end-map', str(args.end_map),
            '--wall-hours', str(remaining), '--output', str(output / 'run'),
        ], cwd=ROOT, check=True, timeout=max(1, deadline - time.monotonic() + 60))
        update('session-finished-check-run-summary-for-completion')
    except BaseException:
        update('interrupted-or-failed-artifacts-preserved')
        raise
    finally:
        data_out = output / 'data'
        data_out.mkdir(exist_ok=True)
        for path in (ROOT / 'data').glob('sokoban_7box_original_test.*'):
            if path.is_file():
                shutil.copy2(path, data_out / path.name)
        bundle_manifest = ROOT / 'BUNDLE_SHA256.json'
        if bundle_manifest.exists():
            shutil.copy2(bundle_manifest, output / bundle_manifest.name)
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'cloud_session.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        archive = shutil.make_archive(str(output), 'zip', output)
        print('Saved cloud artifacts:', archive, flush=True)


if __name__ == '__main__':
    main()
