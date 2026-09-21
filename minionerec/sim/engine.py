"""SIM 训练与推理共用的打分、指标和 checkpoint 协议。"""

import math

import torch
from torch.utils.data import DataLoader

from .data import CandidateCollator, CandidateDataset, SemanticSearch
from .model import SIMRanker


def score_records(model, records, collator, device, batch_size=128):
    """按原候选顺序返回分数，候选为空的查询也保留。

    Args:
        model (SIMRanker): 已初始化或加载的精排模型。
        records (list[dict]): 查询记录，推理可没有 target_id。
        collator (CandidateCollator): 与训练配置一致的特征构造器。
        device (torch.device | str): 推理设备。
        batch_size (int): 候选级 batch 大小。
    Returns:
        list[list[float]]: 与每条 candidate_ids 对齐的 logits。
    """
    model.eval()
    loader = DataLoader(CandidateDataset(records), batch_size=batch_size, collate_fn=collator)
    flat = []
    with torch.no_grad():
        for batch, _ in loader:
            scores = model(**{k: v.to(device) for k, v in batch.items()})
            if not torch.isfinite(scores).all():
                raise ValueError('Non-finite SIM scores')
            flat.extend(scores.cpu().tolist())
    grouped, start = [], 0
    for row in records:
        count = len(row['candidate_ids'])
        grouped.append(flat[start:start + count])
        start += count
    return grouped


def ranking_metrics(records, scores=None, ks=(5, 10, 20)):
    """对所有查询计算商品级指标，包括未召回目标和空候选查询。

    Args:
        records (list[dict]): 必须含 target_id 的查询；一个查询只有一个真实目标。
        scores (list[list[float]] | None): None 使用原召回顺序，否则按 logits 降序。
        ks (tuple[int]): Top-K 截断位置。
    Returns:
        dict: 候选命中上限、HR@K 和 NDCG@K。
    """
    if not records or any(k < 1 for k in ks):
        raise ValueError('Metrics require queries and positive cutoffs')
    totals = {'candidate_recall': 0.0}
    totals.update({f'{name}@{k}': 0.0 for k in ks for name in ('HR', 'NDCG')})
    for i, row in enumerate(records):
        candidates = row['candidate_ids']
        if scores is not None:
            candidates = [candidates[j] for j in sorted(range(len(candidates)), key=lambda j: -scores[i][j])]
        if row['target_id'] not in candidates:
            continue
        rank = candidates.index(row['target_id']) + 1
        totals['candidate_recall'] += 1
        for k in ks:
            if rank <= k:
                totals[f'HR@{k}'] += 1
                totals[f'NDCG@{k}'] += 1 / math.log2(rank + 1)
    return {key: value / len(records) for key, value in totals.items()}


def load_ranker(checkpoint, vectors, index_digest, embedding_digest, device):
    """加载权重并验证数据指纹，防止相同行数但编号不同的数据混用。

    Args:
        checkpoint (str): train 保存的 checkpoint 路径。
        vectors (np.ndarray): 当前归一化的 [N,D] 商品向量。
        index_digest (str): 当前 SID 索引文件 SHA256。
        embedding_digest (str): 当前 npy 文件 SHA256。
        device (torch.device | str): 模型运行设备。
    Returns:
        tuple[SIMRanker,CandidateCollator,dict]: 模型、batch 构造器及 checkpoint 元数据。
    """
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['index_digest'] != index_digest or saved['embedding_digest'] != embedding_digest:
        raise ValueError('Checkpoint index/embedding fingerprint mismatch')
    model = SIMRanker(torch.from_numpy(vectors), **saved['model_config'])
    model.load_state_dict(saved['state_dict'])
    model.to(device).eval()
    collator = CandidateCollator(SemanticSearch(vectors, saved['search_k']), saved['short_length'])
    return model, collator, saved
