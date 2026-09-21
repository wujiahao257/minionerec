"""候选数据协议、定长 batch 和从长历史中检索相关行为。"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def file_digest(path):
    """计算数据指纹，防止训练与推理换用不同编号的目录。

    Args:
        path (str | Path): 文件路径。
    Returns:
        str: 文件内容的 SHA256。
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_vectors(path):
    """读取并归一化商品文本向量；第 i 行必须对应商品 i。

    Args:
        path (str | Path): shape [N,D] 的 .npy 路径。
    Returns:
        np.ndarray: float32 的 [N,D] 单位向量；零向量保持为零。
    """
    vectors = np.load(path, allow_pickle=False).astype(np.float32)
    if vectors.ndim != 2 or min(vectors.shape) < 1 or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must be a finite nonempty [N,D] matrix')
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def load_records(path, item_count, require_target=False, index_digest=None):
    """加载分组候选并验证编号、长度、唯一性及目录指纹。

    Args:
        path (str | Path): prepare 写出的 JSONL。
        item_count (int): 商品向量行数。
        require_target (bool): 训练或评估时是否必须有真实目标。
        index_digest (str | None): 预期 SID 索引指纹。
    Returns:
        list[dict]: 每条查询的历史、候选、召回分数与可选标签。
    """
    records, seen = [], set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['query_id'] in seen:
            raise ValueError('Duplicate query_id: ' + row['query_id'])
        seen.add(row['query_id'])
        ids = row['history_ids'] + row['candidate_ids']
        target = row.get('target_id')
        if require_target and target is None:
            raise ValueError('Training/evaluation requires target_id')
        if target is not None:
            ids = ids + [target]
        if any(type(i) is not int or not 0 <= i < item_count for i in ids):
            raise ValueError('Item ID is outside embedding row range')
        candidates = row['candidate_ids']
        if len(set(candidates)) != len(candidates):
            raise ValueError('Candidate IDs must be deduplicated')
        if len(row['recall_scores']) != len(candidates) or len(row['recall_ranks']) != len(candidates):
            raise ValueError('Candidate feature lengths do not match')
        if any(s is not None and not math.isfinite(s) for s in row['recall_scores']):
            raise ValueError('Recall scores must be finite or null')
        if any(type(r) is not int or r < 1 for r in row['recall_ranks']):
            raise ValueError('Recall ranks must be positive integers')
        if index_digest is not None and row.get('index_digest') != index_digest:
            raise ValueError('SID index fingerprint mismatch')
        records.append(row)
    if not records:
        raise ValueError('No query records')
    return records


class SemanticSearch:
    """GSU：用固定商品向量余弦相似度检索长历史，分块控制内存。

    Args:
        vectors (np.ndarray): 已归一化的 [N,D] 商品向量。
        top_k (int): 每个候选最多保留的历史行为数。
        chunk_size (int): 单次扫描的历史长度，不是历史截断长度。
    """

    def __init__(self, vectors, top_k=32, chunk_size=4096):
        if top_k < 1 or chunk_size < 1:
            raise ValueError('top_k and chunk_size must be positive')
        self.vectors, self.top_k, self.chunk_size = vectors, top_k, chunk_size

    def search(self, history, candidate):
        """检索相关行为；相似度相同时优先较近行为，最终恢复时间顺序。

        Args:
            history (list[int]): 从旧到新的历史，仅包含请求时已发生的行为。
            candidate (int): 候选商品编号，不能传真实目标代替候选。
        Returns:
            tuple[list[int], list[int]]: 选中商品和距历史末尾的行为步数。
        """
        positions = np.empty(0, dtype=np.int64)
        scores = np.empty(0, dtype=np.float32)
        for start in range(0, len(history), self.chunk_size):
            block = history[start:start + self.chunk_size]
            positions = np.concatenate((positions, np.arange(start, start + len(block))))
            scores = np.concatenate((scores, self.vectors[block] @ self.vectors[candidate]))
            keep = np.lexsort((-positions, -scores))[:self.top_k]
            positions, scores = positions[keep], scores[keep]
        positions.sort()
        return [history[p] for p in positions], [len(history) - 1 - int(p) for p in positions]


class CandidateDataset(Dataset):
    """把分组候选展开为候选级训练样本；不补入未召回的目标。

    Args:
        records (list[dict]): 已验证的查询记录。
    """

    def __init__(self, records):
        self.records = records
        self.pairs = [(i, j) for i, row in enumerate(records) for j in range(len(row['candidate_ids']))]

    def __len__(self):
        """Returns: int: 候选总数。"""
        return len(self.pairs)

    def __getitem__(self, index):
        """Args: index (int): 候选样本索引。Returns: tuple[dict, int]: 查询及候选位置。"""
        i, j = self.pairs[index]
        return self.records[i], j


class CandidateCollator:
    """仅把检索出的 K 条历史和短期历史送入 ESU，避免长历史 GPU 张量。

    Args:
        search (SemanticSearch): 固定向量检索器。
        short_length (int): 用于短期兴趣的最近行为数。
    """

    def __init__(self, search, short_length=10):
        if short_length < 1:
            raise ValueError('short_length must be positive')
        self.search, self.short_length = search, short_length

    def __call__(self, samples):
        """组 batch；商品索引统一加一，0 专门用于 padding。

        Args:
            samples (list[tuple[dict,int]]): 查询与候选位置。
        Returns:
            tuple[dict[str,Tensor], Tensor]: 模型特征和 [B] 二元标签；推理不使用标签。
        """
        batch = {'candidate': [], 'selected': [], 'ages': [], 'short': [], 'recall': []}
        labels = []
        for row, j in samples:
            candidate = row['candidate_ids'][j]
            selected, ages = self.search.search(row['history_ids'], candidate)
            pad = self.search.top_k - len(selected)
            batch['candidate'].append(candidate + 1)
            batch['selected'].append([i + 1 for i in selected] + [0] * pad)
            batch['ages'].append([min(int(math.log2(a + 1)), 31) for a in ages] + [0] * pad)
            short = row['history_ids'][-self.short_length:]
            batch['short'].append([i + 1 for i in short] + [0] * (self.short_length - len(short)))
            score = row['recall_scores'][j]
            batch['recall'].append([
                1.0 / row['recall_ranks'][j],
                0.0 if score is None else math.copysign(math.log1p(abs(score)), score),
                float(score is not None),
            ])
            labels.append(float(candidate == row.get('target_id')))
        tensors = {k: torch.tensor(v, dtype=torch.float32 if k == 'recall' else torch.long)
                   for k, v in batch.items()}
        return tensors, torch.tensor(labels, dtype=torch.float32)
