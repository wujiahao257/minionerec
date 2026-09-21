"""SIM 学习实现：固定语义 GSU（data.py）+ 可训练 ESU + 短期兴趣。"""

import torch
from torch import nn


class SIMRanker(nn.Module):
    """候选相关的长短期兴趣精排模型，输出候选级 logit。

    Args:
        vectors (Tensor): 归一化商品向量 [N,D]；固定，不训练 Qwen。
        hidden_dim (int): ESU 隐藏维度，必须能被 heads 整除。
        heads (int): 多头候选注意力的头数。
        dropout (float): 注意力与打分 MLP 的 Dropout 概率。
    """

    def __init__(self, vectors, hidden_dim=64, heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads or not 0 <= dropout < 1:
            raise ValueError('Invalid hidden_dim, heads or dropout')
        table = torch.cat((torch.zeros(1, vectors.shape[1]), vectors.float()), dim=0)
        self.embedding = nn.Embedding.from_pretrained(table, freeze=True, padding_idx=0)
        self.projection = nn.Linear(vectors.shape[1], hidden_dim, bias=False)
        self.age_embedding = nn.Embedding(32, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 3, hidden_dim * 2), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.PReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate, selected, ages, short, recall):
        """仅依赖历史和候选特征；真实目标只用于外部 loss。

        Args:
            candidate (LongTensor): [B]，商品 ID 加一。
            selected (LongTensor): GSU 选中的行为 [B,K]，0 为 padding。
            ages (LongTensor): [B,K]，行为距离的 log2 桶（不是实际时间间隔）。
            short (LongTensor): 最近行为 [B,S]，0 为 padding。
            recall (FloatTensor): [B,3]，倒数排名、变换后的 beam 分数、分数是否存在。
        Returns:
            Tensor: [B] 未归一化分数；不是经过线上校准的 CTR。
        """
        query = self.projection(self.embedding(candidate))  # [B,D] -> [B,H]
        mask = selected.ne(0)
        history = self.projection(self.embedding(selected)) + self.age_embedding(ages)
        history = history * mask.unsqueeze(-1)  # [B,K,H]
        # MultiheadAttention 全部 key 被 mask 时会产生 NaN，空历史临时开放零位置。
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~mask.any(dim=1)
        interest, _ = self.attention(query.unsqueeze(1), history, history,
                                     key_padding_mask=~safe_mask, need_weights=False)
        interest = interest.squeeze(1) * mask.any(dim=1, keepdim=True)  # [B,H]
        short_mask = short.ne(0).unsqueeze(-1)
        short_vectors = self.projection(self.embedding(short)) * short_mask
        recent = short_vectors.sum(1) / short_mask.sum(1).clamp_min(1)  # [B,H]
        features = torch.cat((query, interest, recent, query * interest, query * recent, recall), dim=1)
        return self.scorer(features).squeeze(-1)
