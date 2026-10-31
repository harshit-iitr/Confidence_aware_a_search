"""Build an explicit allowlisted Kaggle payload; never include credentials."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.sokoban7_protocol import SOURCE_FILES, CHECKPOINTS, read_boards, sha256


def package(output, generate_in_cloud=False):
    output = Path(output)
    dataset = ROOT / 'data/sokoban_7box_original_test.txt'
    provenance = dataset.with_suffix('.provenance.json')
    data_files = []
    if dataset.exists() or provenance.exists():
        read_boards(dataset)
        if json.loads(provenance.read_text())['dataset_sha256'] != sha256(dataset):
            raise ValueError('Frozen dataset hash mismatch')
        data_files = ['data/sokoban_7box_original_test.txt',
                      'data/sokoban_7box_original_test.provenance.json']
    elif not generate_in_cloud:
        raise FileNotFoundError('Generate the frozen dataset first, or explicitly use --generate-in-cloud')
    files = sorted(set(SOURCE_FILES + CHECKPOINTS + [
        'Optimize-Planning-Heuristics-to-Rank/LICENSE', 'requirements.txt', 'docs/SOKOBAN7_ORIGINAL_PROTOCOL.md',
        'tools/generate_sokoban7_original.py', 'tools/package_sokoban7_original.py',
        'tools/run_sokoban7_cloud.py',
        'tools/report_sokoban7_astar.py',
    ] + data_files))
    manifest = {p: sha256(ROOT / p) for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode='x', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(ROOT / relative, relative)
        archive.writestr('BUNDLE_SHA256.json', json.dumps(manifest, indent=2))
    print(f'Bundle: {output}\nBytes: {output.stat().st_size}\nSHA256: {sha256(output)}')
    print('No upload or cloud session has been started.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/sokoban7_astar_frozen.zip')
    parser.add_argument('--generate-in-cloud', action='store_true')
    args = parser.parse_args()
    package(args.output, args.generate_in_cloud)
