import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import run_sokoban7_benchmark as runner
from benchmarks.sokoban7_protocol import (
    METHODS, TIMEOUT, MAX_EXPANSIONS, fingerprint, read_boards,
    replay_plan, smoke_board, validate_board, load_records,
)


class ProtocolTests(unittest.TestCase):
    def test_dataset_rejects_duplicates_and_wrong_box_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'boards.txt'
            board = np.zeros((10, 10), dtype=np.int32)
            board[1:9, 1:9] = 1
            board[1, 1] = 3
            board[2, 1:8] = 4
            board[3, 1:8] = 2
            validate_board(board)
            line = ' '.join(map(str, board.flat)) + '\n'
            p.write_text(line + line)
            with self.assertRaises(ValueError):
                read_boards(p, expected_count=2)
            board[2, 1] = 1
            with self.assertRaises(ValueError):
                validate_board(board)

    def test_independent_replay_rejects_illegal_or_wrong_action_types(self):
        board = smoke_board()
        self.assertTrue(replay_plan(board, [4]))
        for actions in [[], [8], [3], [4, 4], [99]]:
            self.assertFalse(replay_plan(board, actions))

    def test_every_dispatch_receives_uniform_budget_and_original_settings(self):
        return_value = (True, 2, 1.0, [4], 0.1, {})
        functions = ['run_classical_astar', 'run_batched_paper_search',
                     'run_single_scratch_astar', 'run_confidence_aware_astar',
                     'run_confidence_aware_gbfs', 'run_ensemble_astar']
        paper, models = MagicMock(), [MagicMock() for _ in range(5)]
        mocks = []
        with patch.object(runner, 'ConfidenceAwareMCHeuristic') as mc, \
                patch.object(runner, 'DeepEnsembleHeuristic') as ensemble:
            from contextlib import ExitStack
            with ExitStack() as stack:
                for function in functions:
                    mocks.append(stack.enter_context(patch.object(runner, function, return_value=return_value)))
                for method in METHODS:
                    runner.dispatch(method, smoke_board(), paper, models, torch.device('cpu'))
                calls = [call for mock in mocks for call in mock.call_args_list]
                self.assertEqual(len(calls), 11)
                for call in calls:
                    self.assertEqual(call.kwargs['max_time'], TIMEOUT)
                    self.assertEqual(call.kwargs['max_expansions'], MAX_EXPANSIONS)
                for call in mc.call_args_list:
                    self.assertEqual(call.kwargs['num_mc_samples'], 5)
                    self.assertEqual(call.kwargs['dropout_p'], 0.10)
                self.assertEqual([c.kwargs['lambda_min'] for c in ensemble.call_args_list], [.20, .20, .50, .20])
                for call in mocks[-1].call_args_list:
                    self.assertFalse(call.kwargs['prune_deadlocks'])
                self.assertEqual(ensemble.call_args_list[-1].kwargs['search_mode'], 'risk_averse')

    def test_seed_is_stable_and_changes_by_map_and_method(self):
        self.assertEqual(runner.seed_for(42, 1, 'paper_mc_astar'), runner.seed_for(42, 1, 'paper_mc_astar'))
        self.assertNotEqual(runner.seed_for(42, 1, 'paper_mc_astar'), runner.seed_for(42, 2, 'paper_mc_astar'))
        self.assertNotEqual(runner.seed_for(42, 1, 'paper_mc_astar'), runner.seed_for(42, 1, 'single_mc_astar'))

    def test_resume_rejects_truncation_and_duplicate_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'results.jsonl'
            p.write_text('{')
            with self.assertRaises(ValueError):
                load_records(p)
            line = json.dumps({'map_id': 1, 'method': 'paper_astar'}) + '\n'
            p.write_text(line + line)
            with self.assertRaises(ValueError):
                load_records(p)

    def test_reports_exclude_unbalanced_maps_and_reject_shard_overlap(self):
        manifest = {'num_maps': 2, 'timeout_sec': 360, 'max_expansions': MAX_EXPANSIONS,
                    'runtime': {'hardware': 'test'}, 'kind': 'fixture', 'base_seed': 42}
        records = []
        for i in (1, 2):
            for method in METHODS:
                if i == 2 and method != 'paper_astar':
                    continue
                records.append({'map_id': i, 'method': method, 'solved': method != 'classical_astar',
                                'verified': method != 'classical_astar', 'expansions': 10,
                                'cost': None if method == 'classical_astar' else 1,
                                'actions': [] if method == 'classical_astar' else [4],
                                'seed': runner.seed_for(42, i, method),
                                'time_sec': 360 if method == 'classical_astar' else 1,
                                'mean_lambda': None, 'protocol_id': fingerprint(manifest)})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            (folder / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            boards = [smoke_board(), smoke_board()]
            result = runner.report([folder], folder / 'report', boards=boards)
            self.assertEqual(result['complete_maps'], [1])
            self.assertEqual(result['unbalanced_records'], 1)
            classical = result['summary'][0]
            self.assertEqual(classical['mean_time_all_sec'], 360)
            self.assertIsNone(classical['mean_expansions'])
            with self.assertRaises(ValueError):
                runner.report([folder, folder], folder / 'bad', boards=boards)
            records[1]['actions'] = [8]
            (folder / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            with self.assertRaises(ValueError):
                runner.report([folder], folder / 'tampered', boards=boards)

    def test_incompatible_resume_fingerprints_are_rejected(self):
        board = smoke_board()
        manifest = {'base_seed': 42}
        row = {'protocol_id': fingerprint(manifest), 'map_id': 1,
               'method': 'paper_astar', 'seed': runner.seed_for(42, 1, 'paper_astar'),
               'solved': True, 'verified': True, 'cost': 1, 'actions': [4]}
        runner.validate_record(row, manifest, [board])
        with self.assertRaises(ValueError):
            runner.validate_record(row, {'base_seed': 43}, [board])


if __name__ == '__main__':
    unittest.main()
