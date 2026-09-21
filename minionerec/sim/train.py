"""从真实召回候选训练 SIM，按验证集 NDCG@10 保存模型。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CandidateCollator, CandidateDataset, SemanticSearch, file_digest, load_records, load_vectors
from .engine import ranking_metrics, score_records
from .model import SIMRanker


def train(train_path, valid_path, embeddings, index, output, epochs=5, batch_size=128,
          learning_rate=0.001, search_k=32, short_length=10, hidden_dim=64,
          heads=4, dropout=0.1, device='cpu', seed=42):
    """训练候选级二分类；未观察到的候选只是采样负例，不是真实曝光负例。

    Args:
        train_path (str): 训练候选 JSONL；必须由训练阶段可用的召回器生成。
        valid_path (str): 独立验证候选 JSONL，用于选择 checkpoint。
        embeddings (str): 同编号商品 npy 路径。
        index (str): 对应 SID 索引 JSON 路径。
        output (str): checkpoint 输出路径。
        epochs (int): 训练轮数。
        batch_size (int): 每批候选数。
        learning_rate (float): AdamW 学习率。
        search_k (int): GSU 从完整历史检索的行为数。
        short_length (int): 短期历史长度。
        hidden_dim (int): ESU 隐藏维度。
        heads (int): ESU 注意力头数。
        dropout (float): Dropout 概率。
        device (str): cpu、cuda 或 mps，由使用者显式选择。
        seed (int): 随机种子。
    Returns:
        dict: 最佳验证指标、训练日志及基线指标。
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('epochs, batch_size and learning_rate must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vectors = load_vectors(embeddings)
    digest = file_digest(index)
    training = load_records(train_path, len(vectors), True, digest)
    validation = load_records(valid_path, len(vectors), True, digest)
    if {r['query_id'] for r in training} & {r['query_id'] for r in validation}:
        raise ValueError('Train/validation queries overlap; use separate chronological splits')
    dataset = CandidateDataset(training)
    positives = sum(r['target_id'] in r['candidate_ids'] for r in training)
    if not 0 < positives < len(dataset):
        raise ValueError('Training requires recalled positives and negative candidates')
    config = dict(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
    model = SIMRanker(torch.from_numpy(vectors), **config).to(device)
    collator = CandidateCollator(SemanticSearch(vectors, search_k), short_length)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    baseline = ranking_metrics(validation)
    best, report = -1.0, {'baseline': baseline, 'epochs': [], 'training_positive_candidates': positives}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, labels in loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(**features), labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scores = score_records(model, validation, collator, device, batch_size)
        metrics = ranking_metrics(validation, scores)
        log = dict(epoch=epoch + 1, loss=total_loss / len(dataset), validation=metrics)
        report['epochs'].append(log)
        print(json.dumps(log, ensure_ascii=False))
        if metrics['NDCG@10'] > best:
            best = metrics['NDCG@10']
            report['best_validation'] = metrics
            report['best_epoch'] = epoch + 1
            torch.save(dict(format_version=1, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            model_config=config, search_k=search_k, short_length=short_length,
                            index_digest=digest, embedding_digest=file_digest(embeddings),
                            train_digest=file_digest(train_path), valid_digest=file_digest(valid_path),
                            training_config=dict(seed=seed, epochs=epochs, batch_size=batch_size,
                                                 learning_rate=learning_rate),
                            validation=metrics), output)
    output.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    """从命令行启动 SIM 训练，默认 CPU，可通过 --device cuda 使用 GPU。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'valid', 'embeddings', 'index', 'output'):
        parser.add_argument('--' + name, required=True)
    for name, default in (('epochs', 5), ('batch-size', 128), ('search-k', 32), ('short-length', 10),
                          ('hidden-dim', 64), ('heads', 4), ('seed', 42)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', default='cpu')
    args = vars(parser.parse_args())
    args['train_path'], args['valid_path'] = args.pop('train'), args.pop('valid')
    train(**args)


if __name__ == '__main__':
    main()
