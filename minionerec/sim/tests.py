"""运行：python -m unittest minionerec.sim.tests -v；全部使用临时合成数据。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .data import CandidateCollator, SemanticSearch, file_digest, load_records
from .engine import ranking_metrics
from .model import SIMRanker
from .prepare import prepare_records, restore_history
from .rerank import rerank
from .train import train


class TestSIM(unittest.TestCase):
    """覆盖历史边界、召回衔接、空序列、梯度及训练推理闭环。"""

    def setUp(self):
        """固定合成测试的随机种子和 CPU 线程数。"""
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.vectors = np.eye(4, dtype=np.float32)

    def test_long_history_retrieves_early_interest(self):
        """即使相关行为位于很久以前，GSU 也能找到它，不预先截掉长历史。"""
        history = [0] + [1] * 100
        for chunk in (7, 4096):
            self.assertEqual(SemanticSearch(self.vectors, 1, chunk).search(history, 0), ([0], [100]))

    def test_history_cutoff_and_ambiguity(self):
        """恢复前缀不包含目标和未来；重复窗口不能靠猜测定位。"""
        self.assertEqual(restore_history([0, 1, 2, 3], [1], 2), [0, 1])
        with self.assertRaises(ValueError):
            restore_history([0, 1, 0, 1], [0], 1)
        self.assertEqual(restore_history([0, 1, 0, 1], [0], 1, 1), [0])
        with self.assertRaises(ValueError):
            restore_history([0, 1, 2], [0], 1, 2)
        self.assertEqual(restore_history([0, 1, 2], [2], None, 3), [0, 1, 2])

    def test_collision_invalid_and_no_target_injection(self):
        """同 SID 商品都参与精排，重复候选去重；漏召回目标不能被偷偷补入。"""
        index = {'0': ['<a_0>'], '1': ['<a_0>'], '2': ['<a_2>']}
        prediction = dict(user_id='A0', history_item_id=[0], item_id=2, output='<a_2>',
                          predict=['<a_0>', '<invalid>', '<a_0>'], predict_scores=[-0.2, -0.3, -0.4])
        rows, stats = prepare_records([prediction], index, 'digest')
        self.assertEqual(rows[0]['candidate_ids'], [0, 1])
        self.assertEqual(rows[0]['recall_ranks'], [1, 1])
        self.assertEqual(rows[0]['recall_scores'], [-0.2, -0.2])
        self.assertEqual(stats['invalid_sids'], 1)
        self.assertEqual(stats['target_recalled'], 0)

    def test_candidate_features_do_not_use_target(self):
        """改变真实标签不应改变模型输入。"""
        row = dict(history_ids=[0, 1], candidate_ids=[2], recall_ranks=[1], recall_scores=[None], target_id=2)
        collate = CandidateCollator(SemanticSearch(self.vectors, 3))
        features, labels = collate([(row, 0)])
        changed = dict(row, target_id=1)
        other, other_labels = collate([(changed, 0)])
        for key in features:
            torch.testing.assert_close(features[key], other[key])
        self.assertNotEqual(labels.item(), other_labels.item())

    def test_empty_history_single_batch_gradient_and_padding(self):
        """空历史、B=1 仍输出 [1]；padding 不改变结果且梯度有限。"""
        row = dict(history_ids=[], candidate_ids=[1], recall_ranks=[1], recall_scores=[-1.0], target_id=1)
        model = SIMRanker(torch.from_numpy(self.vectors), hidden_dim=8, heads=2, dropout=0)
        batch, labels = CandidateCollator(SemanticSearch(self.vectors, 2))([(row, 0)])
        output = model(**batch)
        self.assertEqual(tuple(output.shape), (1,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertIsNone(model.embedding.weight.grad)
        padded = {k: v.clone() for k, v in batch.items()}
        padded['selected'] = torch.zeros((1, 6), dtype=torch.long)
        padded['ages'] = torch.zeros((1, 6), dtype=torch.long)
        torch.testing.assert_close(model(**padded), output)

    def test_metrics_include_misses_and_empty_queries(self):
        """精排命中率的分母必须包含召回失败，不能只评估命中组。"""
        rows = [dict(candidate_ids=[0, 1], target_id=1), dict(candidate_ids=[], target_id=2)]
        baseline = ranking_metrics(rows, ks=(1, 2))
        reranked = ranking_metrics(rows, [[0.0, 1.0], []], ks=(1, 2))
        self.assertEqual(baseline['HR@1'], 0)
        self.assertEqual(reranked['HR@1'], 0.5)
        self.assertEqual(reranked['candidate_recall'], baseline['candidate_recall'])

    def test_train_checkpoint_rerank_and_fingerprint(self):
        """训练、保存、加载、无标签推理闭环；错误编号体系被拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings, index = root / 'items.npy', root / 'index.json'
            np.save(embeddings, self.vectors)
            index.write_text(json.dumps({str(i): [f'<a_{i}>'] for i in range(4)}), encoding='utf-8')
            digest = file_digest(index)
            def record(query, history, target):
                return dict(query_id=query, user_id=query, history_ids=history, candidate_ids=[0, 1, 2, 3],
                            recall_ranks=[1, 2, 3, 4], recall_scores=[None] * 4,
                            target_id=target, index_digest=digest)
            training = [record('t0', [0, 1], 2), record('t1', [1, 2], 3)]
            validation = [record('v0', [0, 1, 2], 3), record('v1', [], 1)]
            def write(name, rows):
                path = root / name
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                return str(path)
            training_path, validation_path = write('train.jsonl', training), write('valid.jsonl', validation)
            checkpoint = str(root / 'sim.pt')
            report = train(training_path, validation_path, str(embeddings), str(index), checkpoint,
                           epochs=2, batch_size=4, search_k=2, hidden_dim=8, heads=2, dropout=0)
            self.assertEqual(len(report['epochs']), 2)
            payload = rerank(validation_path, checkpoint, str(embeddings), str(index), str(root / 'result.json'), top_k=2)
            self.assertEqual(len(payload['results']), 2)
            self.assertTrue(all(len(r['recommendations']) == 2 for r in payload['results']))
            no_labels = copy.deepcopy(validation)
            for row in no_labels:
                row.pop('target_id')
            online = rerank(write('online.jsonl', no_labels), checkpoint, str(embeddings), str(index), str(root / 'online.json'), top_k=2)
            self.assertEqual(payload['results'], online['results'])
            self.assertNotIn('metrics', online)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                train(training_path, training_path, str(embeddings), str(index), checkpoint)
            with self.assertRaises(ValueError):
                load_records(validation_path, 4, index_digest='wrong')
            changed = np.ones((4, 4), dtype=np.float32)
            np.save(embeddings, changed)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                rerank(validation_path, checkpoint, str(embeddings), str(index), str(root / 'wrong.json'))


if __name__ == '__main__':
    unittest.main()
