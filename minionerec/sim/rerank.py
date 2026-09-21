"""用 SIM 重排 MiniOneRec 候选，输出商品 ID 和可选离线对比指标。"""

import argparse
import json
from pathlib import Path

from .data import file_digest, load_records, load_vectors
from .engine import load_ranker, ranking_metrics, score_records


def rerank(candidates, checkpoint, embeddings, index, output, top_k=10, batch_size=128, device='cpu'):
    """加载 SIM 后只重排已有候选，绝不从真实标签补充推荐结果。

    Args:
        candidates (str): prepare 输出的 JSONL；在线推理可无 target_id。
        checkpoint (str): train 输出的权重路径。
        embeddings (str): 训练使用的 npy 路径。
        index (str): 训练使用的 SID 索引路径。
        output (str): 输出 JSON 路径。
        top_k (int): 最多输出多少个不同商品。
        batch_size (int): 候选级推理 batch 大小。
        device (str): cpu、cuda 或 mps。
    Returns:
        dict: Top-K 结果以及标签存在时的召回、精排对比指标。
    """
    if top_k < 1 or batch_size < 1:
        raise ValueError('top_k and batch_size must be positive')
    vectors = load_vectors(embeddings)
    digest = file_digest(index)
    rows = load_records(candidates, len(vectors), index_digest=digest)
    model, collator, _ = load_ranker(checkpoint, vectors, digest, file_digest(embeddings), device)
    scores = score_records(model, rows, collator, device, batch_size)
    sid_index = json.loads(Path(index).read_text(encoding='utf-8'))
    results = []
    for row, values in zip(rows, scores):
        order = sorted(range(len(values)), key=lambda j: -values[j])[:top_k]
        results.append(dict(query_id=row['query_id'], user_id=row['user_id'],
                            recommendations=[dict(item_id=row['candidate_ids'][j],
                                                  sid=''.join(sid_index[str(row['candidate_ids'][j])]),
                                                  sim_score=values[j], recall_rank=row['recall_ranks'][j]) for j in order]))
    payload = {'results': results}
    if all('target_id' in row for row in rows):
        ks = tuple(sorted({5, 10, 20, top_k}))
        payload['metrics'] = {'minionerec': ranking_metrics(rows, ks=ks),
                              'minionerec_sim': ranking_metrics(rows, scores, ks)}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def main():
    """解析路径和设备参数，保存精排结果并打印可用指标。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'checkpoint', 'embeddings', 'index', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    payload = rerank(**vars(parser.parse_args()))
    print(json.dumps(payload.get('metrics', {'queries': len(payload['results'])}), indent=2))


if __name__ == '__main__':
    main()
