"""Privately upload the original-protocol bundle and start one bounded GPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'output/sokoban7_astar_frozen.zip')
    parser.add_argument('--start-map', type=int, default=1)
    parser.add_argument('--end-map', type=int, default=5)
    parser.add_argument('--session-hours', type=float, default=12.0)
    parser.add_argument('--work-hours', type=float, default=11.75)
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.start_map <= args.end_map <= 100:
        raise ValueError('Invalid map range')
    if not 0 < args.work_hours < args.session_hours <= 12:
        raise ValueError('Require 0 < work-hours < session-hours <= 12')
    # Read the supplied key in memory only. Never print or copy it into payloads.
    credentials = json.loads(args.credentials.read_text(encoding='utf-8-sig'))
    os.environ['KAGGLE_USERNAME'] = credentials['username']
    os.environ['KAGGLE_KEY'] = credentials['key']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    username = api.config_values['username']
    bundle = args.bundle.resolve()
    bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    tag = bundle_hash[:10]
    dataset_ref = f'{username}/sokoban7-original-{tag}'
    kernel_ref = f'{username}/sokoban7-original-{args.start_map:03d}-{args.end_map:03d}-{tag}'
    folder = ROOT / 'output' / f'kaggle_original_{tag}'
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / f'receipt_{args.start_map:03d}_{args.end_map:03d}.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'dataset': dataset_ref, 'kernel': kernel_ref, 'private': True,
        'bundle_sha256': bundle_hash, 'range': [args.start_map, args.end_map],
        'profile': 'nine-astar-configurations',
        'session_limit_hours': args.session_hours, 'work_allowance_hours': args.work_hours,
        'status': 'prepared', 'url': f'https://www.kaggle.com/code/{kernel_ref}'}
    if receipt['bundle_sha256'] != bundle_hash or receipt['kernel'] != kernel_ref:
        raise ValueError('Receipt does not match this bundle/account/range')
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    def status():
        response = api.kernels_status(kernel_ref)
        value = str(getattr(response, 'status', response))
        receipt['execution_status'] = value
        save()
        print(json.dumps(receipt, indent=2), flush=True)
    if args.status_only or receipt['status'] == 'submitted':
        status()
        return
    if receipt['status'] in ('submitting', 'submission_error'):
        raise RuntimeError('Previous kernel submission requires inspection before retry')
    data = folder / 'dataset'
    kernel = folder / f'kernel_{args.start_map:03d}_{args.end_map:03d}'
    data.mkdir(exist_ok=True)
    kernel.mkdir(exist_ok=True)
    metadata = {'id': dataset_ref, 'title': f'Sokoban7 original {tag}',
                'licenses': [{'name': 'other'}],
                'description': 'Private original-protocol Sokoban research bundle. '
                               'Contains code, original six checkpoints and generation instructions. '
                               'Existing licenses and ownership retained; no new redistribution license.'}
    (data / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    shutil.copyfile(bundle, data / 'sokoban7_original_protocol.zip')
    # Inspect the authenticated owner's exact private dataset before any retry.
    owned = [d for d in api.dataset_list(mine=True, search=f'sokoban7-original-{tag}')
             if d.ref == dataset_ref]
    if owned:
        if not owned[0].is_private:
            raise RuntimeError('Existing dataset is not private; refusing to continue')
        receipt['status'] = 'dataset_uploaded'
        save()
    elif receipt['status'] == 'prepared':
        receipt['status'] = 'uploading'
        save()
        response = api.dataset_create_new(str(data), public=False, quiet=True, convert_to_csv=False)
        if getattr(response, 'error', None):
            raise RuntimeError(f'Dataset upload failed: {response.error}')
        receipt['status'] = 'dataset_uploaded'
        save()
    else:
        raise RuntimeError('Uncertain previous upload; inspect receipt instead of creating a duplicate')
    deadline = time.monotonic() + 900
    while True:
        try:
            response = api.dataset_status(dataset_ref)
            value = response.lower() if isinstance(response, str) else str(getattr(response, 'status', '')).lower()
        except Exception as exc:
            if getattr(getattr(exc, 'response', None), 'status_code', None) != 403:
                raise
            owned = [d for d in api.dataset_list(mine=True, search=f'sokoban7-original-{tag}')
                     if d.ref == dataset_ref and d.is_private]
            try:
                files = {f.name for f in api.dataset_list_files(dataset_ref, page_size=100).files}
            except Exception as listing_exc:
                # An owned dataset can briefly reject file reads while indexing.
                if getattr(getattr(listing_exc, 'response', None), 'status_code', None) != 403:
                    raise
                files = set()
            expected = {'BUNDLE_SHA256.json', 'finalSok3_pytorch.pt',
                        'tools/run_sokoban7_cloud.py'}
            value = 'ready' if owned and (expected <= files or 'sokoban7_original_protocol.zip' in files) else 'processing'
        print('Private dataset:', value, flush=True)
        if value == 'ready':
            break
        if value in ('failed', 'error') or time.monotonic() > deadline:
            raise RuntimeError('Private dataset did not become ready; receipt preserved')
        time.sleep(15)
    # This bootstrap needs no credentials inside Kaggle. It supports automatic
    # archive expansion by Kaggle and verifies every payload file before running.
    code = '''import hashlib, json, pathlib, subprocess, sys, zipfile
root = pathlib.Path('/kaggle/working/sokoban7_original')
root.mkdir(exist_ok=True)
archives = list(pathlib.Path('/kaggle/input').rglob('sokoban7_original_protocol.zip'))
def install(read, hashes):
    for name, expected in hashes.items():
        dest = (root / name).resolve()
        if not dest.is_relative_to(root.resolve()):
            raise RuntimeError('Unsafe bundle path')
        payload = read(name)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError('Payload checksum mismatch: ' + name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    (root / 'BUNDLE_SHA256.json').write_text(json.dumps(hashes, indent=2))
if archives:
    if len(archives) != 1:
        raise RuntimeError('Ambiguous bundle inputs')
    with zipfile.ZipFile(archives[0]) as archive:
        hashes = json.loads(archive.read('BUNDLE_SHA256.json'))
        if set(archive.namelist()) != set(hashes) | {'BUNDLE_SHA256.json'}:
            raise RuntimeError('Unexpected archive members')
        install(archive.read, hashes)
else:
    manifests = list(pathlib.Path('/kaggle/input').rglob('BUNDLE_SHA256.json'))
    if len(manifests) != 1:
        raise RuntimeError('Ambiguous expanded bundle inputs')
    source = manifests[0].parent
    def read(name):
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise RuntimeError('Unsafe input path')
        return path.read_bytes()
    install(read, json.loads(manifests[0].read_text()))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'gym-sokoban==0.0.6'], check=True)
subprocess.run([sys.executable, '-u', str(root / 'tools/run_sokoban7_cloud.py'),
                '--start-map', 'START_VALUE', '--end-map', 'END_VALUE',
                '--wall-hours', 'WORK_VALUE'], cwd=root, check=True)
'''.replace('START_VALUE', str(args.start_map)).replace('END_VALUE', str(args.end_map)).replace('WORK_VALUE', str(args.work_hours))
    compile(code, 'evaluate.py', 'exec')
    (kernel / 'evaluate.py').write_text(code, encoding='utf-8')
    km = {'id': kernel_ref, 'title': f'Sokoban7 original {args.start_map:03d}-{args.end_map:03d} {tag}',
          'code_file': 'evaluate.py', 'language': 'python', 'kernel_type': 'script',
          'is_private': True, 'enable_gpu': True, 'enable_internet': True,
          'dataset_sources': [dataset_ref], 'competition_sources': [], 'kernel_sources': []}
    (kernel / 'kernel-metadata.json').write_text(json.dumps(km, indent=2), encoding='utf-8')
    if receipt['status'] not in ('dataset_uploaded',):
        raise RuntimeError('Uncertain kernel submission status; inspect before retry')
    receipt['status'] = 'submitting'
    save()
    response = api.kernels_push(str(kernel), timeout=int(args.session_hours * 3600))
    if getattr(response, 'error', None):
        receipt['status'] = 'submission_error'
        save()
        raise RuntimeError(f'GPU submission failed: {response.error}')
    receipt['status'] = 'submitted'
    save()
    print(json.dumps(receipt, indent=2), flush=True)
    status()


if __name__ == '__main__':
    main()
