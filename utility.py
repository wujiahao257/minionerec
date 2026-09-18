import os
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch
# import tensorflow as tf


def extract_axis_1(data, indices):
    """为每个 batch 样本抽取指定序列位置的特征，并保留长度为 1 的序列维。

    Args:
        data (torch.Tensor): 序列特征，shape [B,T,H]。
        indices (torch.LongTensor): 每个样本所选的序列位置，shape [B]。

    Returns:
        torch.Tensor: shape [B,1,H]。
    """
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    """将关键字参数提供的 DataFrame 分别保存成同名 .df pickle 文件。

    Args:
        data_directory (str): 需要创建或写入输出文件的目录路径。
        kwargs (dict[str, pandas.DataFrame]): 文件名到 DataFrame 的映射，每个表分别保存。

    Returns:
        None: 在目标目录写入各 DataFrame。
    """
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    """保留最近 length 条商品；不足时在右侧原地补入 padding 编号。

    Args:
        itemlist (list[int]): 历史商品编号列表；不足目标长度时会被原地追加 padding。
        length (int): 历史保留或补齐后的目标长度。
        pad_item (int): 补齐历史使用的商品编号，通常等于真实商品数。

    Returns:
        list[int]: 长度为 length 的历史列表。
    """
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


# def extract_axis_1(data, ind):
#     """
#     Get specified elements along the first axis of tensor.
#     :param data: Tensorflow tensor that will be subsetted.
#     :param ind: Indices to take (one for each element along axis 0 of data).
#     :return: Subsetted tensor.
#     """

#     batch_range = tf.range(tf.shape(data)[0])
#     indices = tf.stack([batch_range, ind], axis=1)
#     res = tf.gather_nd(data, indices)

#     return res


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    """执行遗留 TensorFlow 层归一化；当前文件未导入 tf，直接调用会失败。

    Args:
        inputs (tensorflow.Tensor): 至少二维的输入，沿末维归一化；当前文件未导入 TensorFlow。
        epsilon (float): 归一化分母中的稳定常数，避免零方差除零。
        scope (str): 遗留 TensorFlow 层归一化变量作用域名。
        reuse (bool | None): 遗留 TensorFlow 变量作用域的参数复用配置。

    Returns:
        tensorflow.Tensor: 与输入同形状的归一化表示（需调用环境提供 tf）。
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    """按点击和购买奖励分类，原地累积 Top-K 命中、NDCG 与奖励。

    Args:
        sorted_list (numpy.ndarray): 按分数从低到高排序的商品 ID，shape [B,N]，取尾部作为 Top-K。
        topk (list[int]): 需要统计的推荐截断位置，例如 [1,3,5,10,20]。
        true_items (list[int] | numpy.ndarray): 每条样本真实目标商品编号，长度 B。
        rewards (list[float] | numpy.ndarray): 每条交互的行为奖励，用于区分点击和购买并累计总奖励。
        r_click (float): 用于识别点击行为的奖励值，等于此值的样本归入点击指标。
        total_reward (list[float] | numpy.ndarray): 按各个 K 累积的奖励或点击/购买指标，函数原地更新此容器。
        hit_click (list[float] | numpy.ndarray): 按各个 K 累积的奖励或点击/购买指标，函数原地更新此容器。
        ndcg_click (list[float] | numpy.ndarray): 按各个 K 累积的奖励或点击/购买指标，函数原地更新此容器。
        hit_purchase (list[float] | numpy.ndarray): 按各个 K 累积的奖励或点击/购买指标，函数原地更新此容器。
        ndcg_purchase (list[float] | numpy.ndarray): 按各个 K 累积的奖励或点击/购买指标，函数原地更新此容器。

    Returns:
        None: 修改传入的各项指标容器。
    """
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


# class Memory():
#     def __init__(self):
#         self.buffer = deque()
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random.choice(np.arange(len(self.buffer)),
#                                size=batch_size,
#                                replace=False)
#         return [self.buffer[ii] for ii in idx]

# NeuProcessEncoder
class NeuProcessEncoder(nn.Module):
    """遗留的随机潜变量编码器，将集合特征聚合后采样高斯潜表示；未接入主推荐训练。

    Args:
        input_size (int): 输入特征宽度；MemoryUnit 中还决定生成权重矩阵的输入列数。
        hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
        output_size (int): 输出特征宽度；MemoryUnit 中决定生成权重矩阵的输出行数。
        dropout_prob (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
        device (str | torch.device | None): 张量或模型的目标设备，例如 cuda:0 或 cpu；某些辅助类仅保存此值。
    """
    def __init__(self, input_size=64, hidden_size=64, output_size=64, dropout_prob=0.4, device=None):
        """初始化 NeuProcessEncoder：遗留的随机潜变量编码器，将集合特征聚合后采样高斯潜表示；未接入主推荐训练。

        Args:
            self (NeuProcessEncoder): 当前实例，由 Python 在调用实例方法时自动传入。
            input_size (int): 输入特征宽度；MemoryUnit 中还决定生成权重矩阵的输入列数。
            hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
            output_size (int): 输出特征宽度；MemoryUnit 中决定生成权重矩阵的输出行数。
            dropout_prob (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
            device (str | torch.device | None): 张量或模型的目标设备，例如 cuda:0 或 cpu；某些辅助类仅保存此值。

        Returns:
            None: 完成实例初始化。
        """
        super(NeuProcessEncoder, self).__init__()
        self.device = device
        
        # Encoder for item embeddings
        layers = [nn.Linear(input_size, hidden_size),
                torch.nn.Dropout(dropout_prob),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, output_size)]
        self.input_to_hidden = nn.Sequential(*layers)

        # Encoder for latent vector z
        self.z1_dim = input_size # 64
        self.z2_dim = hidden_size # 64
        self.z_dim = output_size # 64
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, self.z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, self.z_dim)

    def emb_encode(self, input_tensor):
        """用两层 MLP 编码集合中的每个输入特征。

        Args:
            self (NeuProcessEncoder): 当前实例，由 Python 在调用实例方法时自动传入。
            input_tensor (torch.Tensor): 集合特征 [...,T,input_size]。

        Returns:
            torch.Tensor: 末维变为 output_size，其余 batch/集合维保持。
        """
        hidden = self.input_to_hidden(input_tensor)

        return hidden

    def aggregate(self, input_tensor):
        """沿倒数第二维对集合特征取平均。

        Args:
            self (NeuProcessEncoder): 当前实例，由 Python 在调用实例方法时自动传入。
            input_tensor (torch.Tensor): 编码后的集合特征 [...,T,output_size]，沿 T 求平均。

        Returns:
            torch.Tensor: 去掉集合维后的平均表示。
        """
        return torch.mean(input_tensor, dim=-2)

    def z_encode(self, input_tensor):
        """预测高斯均值和对数方差，并通过重参数采样生成潜变量。

        Args:
            self (NeuProcessEncoder): 当前实例，由 Python 在调用实例方法时自动传入。
            input_tensor (torch.Tensor): 聚合后的特征 [...,z1_dim]；前一阶段输出需与 z1_dim 匹配。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: z、mu、log_sigma，末维为 z_dim。
        """
        hidden = torch.relu(self.z_to_hidden(input_tensor))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, log_sigma
    
    def encoder(self, input_tensor):
        """依次执行特征 MLP、集合平均和潜变量采样，缓存 self.z。

        Args:
            self (NeuProcessEncoder): 当前实例，由 Python 在调用实例方法时自动传入。
            input_tensor (torch.Tensor): 集合输入 [...,T,input_size]；当前潜变量链要求聚合末维与 z1_dim 匹配。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: z、均值和对数方差。
        """
        z_ = self.emb_encode(input_tensor)
        z = self.aggregate(z_)
        self.z, mu, log_sigma = self.z_encode(z)
        return self.z, mu, log_sigma

    def forward(self, input_tensor):
        """编码集合并采样随机潜表示，仅返回 z。

        Args:
            self (NeuProcessEncoder): 当前实例，由 Python 在调用实例方法时自动传入。
            input_tensor (torch.Tensor): 集合输入 [...,T,input_size]；当前潜变量链要求聚合末维与 z1_dim 匹配。

        Returns:
            torch.Tensor: 潜变量表示，末维为 output_size。
        """
        self.z, _, _ = self.encoder(input_tensor)
        return self.z


class MemoryUnit(nn.Module):
    # clusters_k is k keys
    """按查询向量加权读取记忆槽并生成权重矩阵；当前文件未导入所用 init，不能直接构造。

    Args:
        input_size (int): 输入特征宽度；MemoryUnit 中还决定生成权重矩阵的输入列数。
        output_size (int): 输出特征宽度；MemoryUnit 中决定生成权重矩阵的输出行数。
        emb_size (int): MemoryUnit 查询向量与可学习 key 的特征宽度。
        clusters_k (int): 可学习记忆 key/value 槽位数量。
    """
    def __init__(self, input_size, output_size, emb_size, clusters_k=10):
        """初始化 MemoryUnit：按查询向量加权读取记忆槽并生成权重矩阵；当前文件未导入所用 init，不能直接构造。

        Args:
            self (MemoryUnit): 当前实例，由 Python 在调用实例方法时自动传入。
            input_size (int): 输入特征宽度；MemoryUnit 中还决定生成权重矩阵的输入列数。
            output_size (int): 输出特征宽度；MemoryUnit 中决定生成权重矩阵的输出行数。
            emb_size (int): MemoryUnit 查询向量与可学习 key 的特征宽度。
            clusters_k (int): 可学习记忆 key/value 槽位数量。

        Returns:
            None: 完成实例初始化。
        """
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.input_size = input_size
        self.output_size = output_size
        self.array = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, input_size*output_size)))
        self.index = nn.Parameter(init.xavier_uniform_(torch.FloatTensor(self.clusters_k, emb_size)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, bias_emb):
        """以查询表示对记忆 key 做 softmax 加权，组合 value 并重塑为参数矩阵。

        Args:
            self (MemoryUnit): 当前实例，由 Python 在调用实例方法时自动传入。
            bias_emb (torch.Tensor): 查询记忆的表示，shape [B,1,emb_size]。

        Returns:
            torch.Tensor: shape [B,output_size,input_size] 的动态权重矩阵。
        """
        att_scores = torch.matmul(bias_emb, self.index.transpose(-1, -2)) # [batch_size, clusters_k]
        att_scores = self.softmax(att_scores)

        # [batch_size, input_size, output_size]
        para_new = torch.matmul(att_scores, self.array) # [batch_size, input_size*output_size]
        para_new = para_new.view(-1, self.output_size, self.input_size)

        return para_new

    def reg_loss(self, reg_weights=1e-2):
        """计算记忆值矩阵与 key 矩阵的 L2 范数加权和。

        Args:
            self (MemoryUnit): 当前实例，由 Python 在调用实例方法时自动传入。
            reg_weights (float): 记忆 key/value 参数 L2 范数正则项的系数。

        Returns:
            torch.Tensor: 零维正则损失。
        """
        loss_1 = reg_weights * self.array.norm(2)
        loss_2 = reg_weights * self.index.norm(2)

        return loss_1 + loss_2