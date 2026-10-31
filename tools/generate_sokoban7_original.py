"""Freeze 100 seven-box maps using the original five-box generator's procedure."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.sokoban7_protocol import read_boards, sha256, validate_board


def generate(output, seed=42, max_attempts=100_000):
    from gym_sokoban.envs.room_utils import generate_room
    output = Path(output)
    provenance = output.with_suffix('.provenance.json')
    if output.exists() or provenance.exists():
        raise FileExistsError('Refusing to replace a frozen dataset or its provenance')
    np.random.seed(seed)
    random.seed(seed)
    states, seen, errors = [], set(), {}
    for attempt in range(1, max_attempts + 1):
        try:
            _, state, _ = generate_room(dim=(10, 10), num_boxes=7, num_steps=30, tries=20)
        except (RuntimeError, ValueError) as exc:
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
            continue
        grid = np.array(state, dtype=np.int32, copy=True)
        grid[grid == 5] = 3
        try:
            validate_board(grid)
        except ValueError:
            continue
        flat = tuple(map(int, grid.flat))
        if flat in seen:
            continue
        states.append(flat)
        seen.add(flat)
        if len(states) % 10 == 0:
            print(f'Frozen {len(states)}/100 maps after {attempt} attempts', flush=True)
        if len(states) == 100:
            break
    if len(states) != 100:
        raise RuntimeError(f'Generated only {len(states)} maps; no partial dataset written')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as f:
        f.write(''.join(' '.join(map(str, s)) + '\n' for s in states))
    read_boards(output)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_sha256': sha256(output), 'count': 100, 'seed': seed,
        'python_random_seeded': True, 'numpy_random_seeded': True,
        'generator': 'gym_sokoban.envs.room_utils.generate_room',
        'gym_sokoban_version': importlib.metadata.version('gym-sokoban'),
        'numpy_version': np.__version__, 'attempts': attempt,
        'generator_exceptions': errors,
        'parameters': {'dim': [10, 10], 'num_boxes': 7, 'num_steps': 30, 'tries': 20},
        'selection': 'Tile counts, closed boundary, and uniqueness only; no solver filtering',
        'reference': 'benchmarks/generate_5box_dataset.py; box count changed from 5 to 7',
        'generator_source_sha256': sha256(__file__),
        'note': 'num_steps=30 is a generator parameter, not an optimal solution length',
    }
    with provenance.open('x', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved {output}: {metadata["dataset_sha256"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/sokoban_7box_original_test.txt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.seed)
