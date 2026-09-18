import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SASRecModules_ori import *
import random
import json
import copy
import ast
import wandb

logging.getLogger().setLevel(logging.INFO)

        

def parse_args():
    """解析当前脚本的命令行参数，返回后续数据或模型构造配置。

    Args:
        无显式参数。

    Returns:
        argparse.Namespace: 当前入口定义的参数集合。
    """
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='Goodreads_5',
                        help='Toys_and_Games, Goodreads, Industrial_and_Scientific, CDs_and_Vinyl')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=32,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=1,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-5,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument("--model", type=str, default="SASRec",
                        help='the model name, GRU, Caser, SASRec')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument("--early_stop", type=int, default=20,
                        help='the epoch for early stop')
    parser.add_argument("--eval_num", type=int, default=1,
                        help='evaluate every eval_num epoch' )
    parser.add_argument("--seed", type=int, default=1,
                        help="the random seed")
    parser.add_argument("--result_json_path", type=str, default="./result_temp/temp.json")
    parser.add_argument("--sample_num", type=int, default = 65536)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="bce")
    return parser.parse_args()

def setup_seed(seed):
    """固定 Python、NumPy 或 PyTorch 随机状态，减少重复实验中的随机差异。

    Args:
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。

    Returns:
        None: 修改当前进程的随机状态和相关后端设置。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GRU(nn.Module):
    """以商品 Embedding 和 GRU 编码历史，再输出全商品分数的基线模型。

    Args:
        hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
        item_num (int): 真实商品总数 N；Embedding 的额外第 N 行用作 padding。
        state_size (int): 固定历史序列宽度 S，用于 padding、位置 Embedding 或卷积窗口。
        gru_layers (int): GRU 堆叠层数；当前输出整形最直接对应单层配置。
    """
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1):
        """初始化 GRU：以商品 Embedding 和 GRU 编码历史，再输出全商品分数的基线模型。

        Args:
            self (GRU): 当前实例，由 Python 在调用实例方法时自动传入。
            hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
            item_num (int): 真实商品总数 N；Embedding 的额外第 N 行用作 padding。
            state_size (int): 固定历史序列宽度 S，用于 padding、位置 Embedding 或卷积窗口。
            gru_layers (int): GRU 堆叠层数；当前输出整形最直接对应单层配置。

        Returns:
            None: 完成实例初始化。
        """
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states, len_states):
        # Supervised Head
        """对商品 Embedding 序列打包并用 GRU 编码，将最终隐藏状态映射为商品分数。

        Args:
            self (GRU): 当前实例，由 Python 在调用实例方法时自动传入。
            states (torch.LongTensor): 商品编号序列，shape [B,S]，右侧用 item_num 对应的 padding 编号补齐。
            len_states (torch.LongTensor): 每条序列的有效历史长度，shape [B]；用于 GRU 打包或定位最后有效位置。

        Returns:
            torch.Tensor: 单层配置下 [B,N]；多层 hidden 被展平，输出行数会相应增加。
        """
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

    def forward_eval(self, states, len_states):
        # Supervised Head
        """对商品 Embedding 序列打包并用 GRU 编码，将最终隐藏状态映射为商品分数。

        Args:
            self (GRU): 当前实例，由 Python 在调用实例方法时自动传入。
            states (torch.LongTensor): 商品编号序列，shape [B,S]，右侧用 item_num 对应的 padding 编号补齐。
            len_states (torch.LongTensor): 每条序列的有效历史长度，shape [B]；用于 GRU 打包或定位最后有效位置。

        Returns:
            torch.Tensor: 单层配置下 [B,N]；多层 hidden 被展平，输出行数会相应增加。
        """
        emb = self.item_embeddings(states)
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, len_states.cpu(), batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output


class Caser(nn.Module):
    """以水平和垂直卷积聚合固定长度历史、输出全商品分数的基线模型。

    Args:
        hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
        item_num (int): 真实商品总数 N；Embedding 的额外第 N 行用作 padding。
        state_size (int): 固定历史序列宽度 S，用于 padding、位置 Embedding 或卷积窗口。
        num_filters (int): Caser 每种水平卷积窗口使用的卷积核数量。
        filter_sizes (str): 水平卷积窗口高度列表的字符串，例如 [2,3,4]；构造器通过 eval 解析。
        dropout_rate (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
    """
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate):
        """初始化 Caser：以水平和垂直卷积聚合固定长度历史、输出全商品分数的基线模型。

        Args:
            self (Caser): 当前实例，由 Python 在调用实例方法时自动传入。
            hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
            item_num (int): 真实商品总数 N；Embedding 的额外第 N 行用作 padding。
            state_size (int): 固定历史序列宽度 S，用于 padding、位置 Embedding 或卷积窗口。
            num_filters (int): Caser 每种水平卷积窗口使用的卷积核数量。
            filter_sizes (str): 水平卷积窗口高度列表的字符串，例如 [2,3,4]；构造器通过 eval 解析。
            dropout_rate (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。

        Returns:
            None: 完成实例初始化。
        """
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, states, len_states):
        """用水平卷积加池化和垂直卷积提取历史特征，拼接后输出商品分数。

        Args:
            self (Caser): 当前实例，由 Python 在调用实例方法时自动传入。
            states (torch.LongTensor): 商品编号序列，shape [B,S]，右侧用 item_num 对应的 padding 编号补齐。
            len_states (torch.LongTensor): 每条序列的有效历史长度，shape [B]；用于 GRU 打包或定位最后有效位置。

        Returns:
            torch.Tensor: 正常批量输入下 [B,N]；内部 squeeze 对单样本维度需额外留意。
        """
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

    def forward_eval(self, states, len_states):
        """用水平卷积加池化和垂直卷积提取历史特征，拼接后输出商品分数。

        Args:
            self (Caser): 当前实例，由 Python 在调用实例方法时自动传入。
            states (torch.LongTensor): 商品编号序列，shape [B,S]，右侧用 item_num 对应的 padding 编号补齐。
            len_states (torch.LongTensor): 每条序列的有效历史长度，shape [B]；用于 GRU 打包或定位最后有效位置。

        Returns:
            torch.Tensor: 正常批量输入下 [B,N]；内部 squeeze 对单样本维度需额外留意。
        """
        input_emb = self.item_embeddings(states)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)
        
        return supervised_output

class SASRec(nn.Module):
    """以位置 Embedding 和单个因果注意力模块编码历史的序列推荐基线。

    Args:
        hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
        item_num (int): 真实商品总数 N；Embedding 的额外第 N 行用作 padding。
        state_size (int): 固定历史序列宽度 S，用于 padding、位置 Embedding 或卷积窗口。
        dropout (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
        device (str | torch.device | None): 张量或模型的目标设备，例如 cuda:0 或 cpu；某些辅助类仅保存此值。
        num_heads (int): 注意力头数；隐藏维度必须能被它整除。
    """
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        """初始化 SASRec：以位置 Embedding 和单个因果注意力模块编码历史的序列推荐基线。

        Args:
            self (SASRec): 当前实例，由 Python 在调用实例方法时自动传入。
            hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
            item_num (int): 真实商品总数 N；Embedding 的额外第 N 行用作 padding。
            state_size (int): 固定历史序列宽度 S，用于 padding、位置 Embedding 或卷积窗口。
            dropout (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
            device (str | torch.device | None): 张量或模型的目标设备，例如 cuda:0 或 cpu；某些辅助类仅保存此值。
            num_heads (int): 注意力头数；隐藏维度必须能被它整除。

        Returns:
            None: 完成实例初始化。
        """
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        """以商品和位置向量编码历史，经因果注意力和前馈后抽取最后有效位置预测商品。

        Args:
            self (SASRec): 当前实例，由 Python 在调用实例方法时自动传入。
            states (torch.LongTensor): 商品编号序列，shape [B,S]，右侧用 item_num 对应的 padding 编号补齐。
            len_states (torch.LongTensor): 每条序列的有效历史长度，shape [B]；用于 GRU 打包或定位最后有效位置。

        Returns:
            torch.Tensor: 通常 [B,N]；无参数 squeeze 在 B=1 时会去掉 batch 维。
        """
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        # state_hidden = extract_axis_1(ff_out, len_states - 1)
        indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        """以商品和位置向量编码历史，经因果注意力和前馈后抽取最后有效位置预测商品。

        Args:
            self (SASRec): 当前实例，由 Python 在调用实例方法时自动传入。
            states (torch.LongTensor): 商品编号序列，shape [B,S]，右侧用 item_num 对应的 padding 编号补齐。
            len_states (torch.LongTensor): 每条序列的有效历史长度，shape [B]；用于 GRU 打包或定位最后有效位置。

        Returns:
            torch.Tensor: 通常 [B,N]；无参数 squeeze 在 B=1 时会去掉 batch 维。
        """
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        # state_hidden = extract_axis_1(ff_out, len_states - 1)
        indices = (len_states -1 ).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        state_hidden = torch.gather(ff_out, 1, indices)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output


def evaluate_games(model, test_data, device, topk, save_logits=False, eval_type="test"):

    """对基线模型计算全商品排名与 HR/NDCG，可保存 logits；读取目录固定为测试目录。

    Args:
        model (torch.nn.Module): 提供 forward_eval 的 SASRec、GRU 或 Caser 基线，输出全商品分数。
        test_data (str): 评测 CSV 文件名；当前函数使用 data_directory_test 拼接路径。
        device (str | torch.device | None): 张量或模型的目标设备，例如 cuda:0 或 cpu；某些辅助类仅保存此值。
        topk (list[int]): 需要统计的推荐截断位置，例如 [1,3,5,10,20]。
        save_logits (bool): 是否保存 SASRec 的全商品预测分数到 NumPy 文件。
        eval_type (str): 日志阶段标记，如 test 或 val；不改变当前函数固定的读取目录。

    Returns:
        tuple[float, list[float], list[float]]: 最大 K 的 NDCG、各 K 的 HR、各 K 的 NDCG。
    """
    def calculate_hit_games_cuda(prediction, topk_list, target, hit_all, ndcg_all):
        """在 GPU 上计算目标商品排名，累积各 K 的命中数和折扣增益。

        Args:
            prediction (torch.Tensor): 每条历史对所有商品的分数矩阵，shape [B,N]。
            topk_list (list[int]): 需要统计的推荐截断位置，例如 [1,3,5,10,20]。
            target (torch.LongTensor): 真实目标商品编号，shape [B]。
            hit_all (list[float]): 按各个 K 累积的命中数或折扣增益，函数会原地更新。
            ndcg_all (list[float]): 按各个 K 累积的命中数或折扣增益，函数会原地更新。

        Returns:
            tuple[list[float], list[float]]: 更新后的 hit_all 和 ndcg_all。
        """
        rank_list = (prediction.shape[1] - 1 - torch.argsort(torch.argsort(prediction)))
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1).clone()
        ndcg_temp_full = 1 / torch.log2(target_rank + 2)
        for i, top_k in enumerate(topk_list):
            mask = (target_rank < top_k)
            mask = mask.float()
            recall_temp = mask.sum()
            ndcg_temp = (ndcg_temp_full * mask).sum()
            hit_all[i] += recall_temp.cpu().item()
            ndcg_all[i] += ndcg_temp.cpu().item()
        return hit_all, ndcg_all

    # def calculate_hit_games(sorted_list, topk, true_items, hit_list, ndcg_list):
    #     for i in range(len(topk)):
    #         rec_list = sorted_list[:, -topk[i]:]
    #         for j in range(len(true_items)):
    #             if true_items[j] in rec_list[j]:
    #                 rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
    #                 hit_list[i] += 1.0
    #                 ndcg_list[i] += 1.0 / np.log2(rank + 1)

    # eval_seqs=pd.read_pickle(os.path.join(data_directory, test_data))
    eval_seqs = pd.read_csv(os.path.join(data_directory_test, test_data))
    eval_seqs = eval_seqs[['history_item_id', 'item_id']]
    eval_seqs = eval_seqs.rename(columns={'history_item_id': 'seq', 'item_id': 'next'})
    # transform '[1,2,3]' to [1,2,3]
    eval_seqs['seq'] = eval_seqs['seq'].apply(ast.literal_eval)
    eval_seqs['len_seq'] = eval_seqs['seq'].apply(lambda x: len(x))

    # right padding
    eval_seqs['seq'] = eval_seqs['seq'].apply(lambda x: x + [item_num] * (seq_size - len(x)))

    batch_size=1024
    hit_all = []
    ndcg_all = []
    for i in topk:
        hit_all.append(0)
        ndcg_all.append(0)
    total_samples = len(eval_seqs)
    total_batch_num = int(total_samples/batch_size) + (total_samples > batch_size * int(total_samples/batch_size))
    sasrec_logits = []
    for i in range(total_batch_num):
        begin = i * batch_size
        end = (i + 1) * batch_size
        if end > total_samples:
            batch = eval_seqs[begin:]
        else:
            batch = eval_seqs[begin:end]

        seq = list(batch['seq'].tolist())
        len_seq = list(batch['len_seq'])
        target=list(batch['next'])

        seq = torch.LongTensor(seq)

        seq = seq.to(device)

        target = torch.LongTensor(target).to(device)

        _ = model.eval()
        with torch.no_grad():
            prediction = model.forward_eval(seq, torch.tensor(np.array(len_seq)).to(device))
            sasrec_logits.append(prediction)
            # # print(prediction)
            # prediction = prediction.cpu()
            # prediction = prediction.detach().numpy()
            # print(prediction)
            # prediction=sess.run(GRUnet.output, feed_dict={GRUnet.inputs: states,GRUnet.len_state:len_states,GRUnet.keep_prob:1.0})
            # sorted_list=np.argsort(prediction)
            hit_all, ndcg_all = calculate_hit_games_cuda(prediction,topk, target, hit_all, ndcg_all)
    print('#############################################################')
    # logging.info('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    # logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    sasrec_logits = torch.cat(sasrec_logits, dim=0)
    # save sasrec_logits as npy file
    if save_logits and args.model == "SASRec":
        # torch.save(sasrec_logits, f"./code/baselines/result_temp/{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_logits.npy")
        np.save(f"./result_temp/{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}_dropout{args.dropout_rate}_logits.npy", sasrec_logits.detach().cpu().numpy())
    hr_list = []
    ndcg_list = []
    # logging.info('#############################################################')
    for i in range(len(topk)):
        hr_purchase=hit_all[i]/len(eval_seqs)
        ng_purchase=ndcg_all[i]/len(eval_seqs)

        hr_list.append(hr_purchase)
        try:
            ndcg_list.append(ng_purchase)
        except:
            if ng_purchase == 0:
                ndcg_list.append(0)
            else:
                return "error"

    ndcg_last = ndcg_list[-1]

    str1 = ''
    str2 = ''
    for i in range(len(topk)):
        if eval_type == "test":
            str1 += 'hr@{}\tndcg@{}\t'.format(topk[i], topk[i])
            str2 += '{:.6f}\t{:.6f}\t'.format(hr_list[i], ndcg_list[i])
            wandb.log({
            f'Recall@{topk[i]}': hr_list[i],
            f'NDCG@{topk[i]}': ndcg_list[i]
            })

    print(str1)
    print(str2)
    print('#############################################################')
    if eval_type == "test":
        metrics_dict = {f'HR@{topk[i]}': hr_list[i] for i in range(len(topk))}
        metrics_dict.update({f'NDCG@{topk[i]}': ndcg_list[i] for i in range(len(topk))})
        wandb.log(metrics_dict)


    return ndcg_last, hr_list, ndcg_list


def calcu_propensity_score(buffer):
    """对目标商品频次加一平滑、归一化，再取 0.05 次幂，供 DRO 项加权。

    Args:
        buffer (pandas.DataFrame): 基线训练表，包含 seq、len_seq、next 等列。

    Returns:
        numpy.ndarray: shape [item_num] 的平滑频率权重。
    """
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps

class RecDataset(Dataset):
    """为基线训练提供历史整数序列、有效长度和下一商品编号。

    Args:
        data_df (pandas.DataFrame): 基线训练表，包含 seq、len_seq、next 等列。
    """
    def __init__(self, data_df):
        """初始化 RecDataset：为基线训练提供历史整数序列、有效长度和下一商品编号。

        Args:
            self (RecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_df (pandas.DataFrame): 基线训练表，包含 seq、len_seq、next 等列。

        Returns:
            None: 完成实例初始化。
        """
        self.data = data_df

    def __getitem__(self, i):
        """取出基线的一条历史、长度和下一商品编号并转换为 Tensor。

        Args:
            self (RecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            i (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 序列 [S]、长度标量、目标标量。
        """
        temp = self.data.iloc[i]
        seq = torch.tensor(temp['seq'])
        len_seq = torch.tensor(temp['len_seq'])
        next = torch.tensor(temp['next'])
        return seq, len_seq, next

    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (RecDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.data)

def main(topk, data_file_train, data_file_test, data_file_valid):
    """训练所选 GRU/Caser/SASRec 基线，定期评测并保留验证指标最好的模型。

    Args:
        topk (list[int]): 需要统计的推荐截断位置，例如 [1,3,5,10,20]。
        data_file_train (str): 对应训练、测试或验证交互 CSV 的文件名，由 main 与目录配置关联。
        data_file_test (str): 对应训练、测试或验证交互 CSV 的文件名，由 main 与目录配置关联。
        data_file_valid (str): 对应训练、测试或验证交互 CSV 的文件名，由 main 与目录配置关联。

    Returns:
        tuple[torch.nn.Module, list[float], list[float]]: 最佳模型及对应验证 NDCG、HR。
    """
    if not args.debug:
        run = wandb.init(
            project="Rec",
            name=(
                f"{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}_dropout{args.dropout_rate}"
            ),  # Set the run name directly in the `init` method
            config={  # You can add your configuration here if needed
                "data": args.data,
                "model": args.model,
                "hidden_factor": args.hidden_factor,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "loss_type": args.loss_type,
            },
        )
        wandb.run.name = run.name
    else:
        os.environ["WANDB_DISABLED"] = "true"



    if args.model=='SASRec':
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    if args.model=="GRU":
        model = GRU(args.hidden_factor,item_num, seq_size)
    if args.model=="Caser":
        model = Caser(args.hidden_factor,item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    if args.loss_type == "bce":
        model_loss = nn.BCEWithLogitsLoss()
    elif args.loss_type == "ce":
        model_loss = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss type: {args.loss_type}")

    model.to(device)

    train = pd.read_csv(os.path.join(data_directory_train, data_file_train))
    train = train[['history_item_id', 'item_id']]
    train = train.rename(columns={'history_item_id': 'seq', 'item_id': 'next'})
    # transform '[1,2,3]' to [1,2,3]
    train['seq'] = train['seq'].apply(ast.literal_eval)
    train['len_seq'] = train['seq'].apply(lambda x: len(x))

    # right padding
    train['seq'] = train['seq'].apply(lambda x: x + [item_num] * (seq_size - len(x)))

    # train_data_org = train
    # train_data = train_data_org.sample(n=args.sample_num ,random_state=args.seed)
    train_data = train
    

    train_dataset = RecDataset(train_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    ps = calcu_propensity_score(train_data)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    ndcg_max = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size) + (int(num_rows/args.batch_size) * args.batch_size != num_rows)
    for i in range(args.epoch):
        # for j in tqdm(range(num_batches)):
        for j, (seq, len_seq, target) in tqdm(enumerate(train_loader)):
            target_neg = []
            for index in range(len(len_seq)):
                neg=np.random.randint(item_num)
                while neg==target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            target_neg = torch.LongTensor(target_neg)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)

            if args.model=="GRU":
                len_seq = len_seq.cpu()

            model_output = model.forward(seq, len_seq)


            target = target.view((-1, 1))
            target_neg = target_neg.view((-1, 1))

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)

            pos_labels = torch.ones((len(len_seq), 1))
            neg_labels = torch.zeros((len(len_seq), 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(device)

            if args.loss_type == "bce":
                loss = model_loss(scores, labels)
            elif args.loss_type == "ce":
                loss = model_loss(model_output, target.squeeze(-1).long())
            else:
                raise ValueError(f"Invalid loss type: {args.loss_type}")


            pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
            pos_scores_dro = torch.squeeze(pos_scores_dro)
            pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
            pos_loss_dro = torch.squeeze(pos_loss_dro)

            inner_dro = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1) - torch.exp((pos_scores_dro / args.beta)) + torch.exp((pos_loss_dro / args.beta)) 


            loss_dro = torch.log(inner_dro + 1e-24)
            if args.alpha == 0.0:
                loss_all = loss
            else:
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            loss_all.backward()
            optimizer.step()

            if True:

                total_step+=1


                if total_step % (num_batches * args.eval_num) == 0:
                        print('VAL PHRASE:')
                        ndcg_last, val_hr, val_ndcg = evaluate_games(model, data_file_valid, device, topk, eval_type="val")
                        # ndcg_last, val_hr, val_ndcg = evaluate_games_old(model, 'val_sessions.df', device, topk)
                        print('TEST PHRASE:')
                        _, test_hr, test_ndcg = evaluate_games(model, data_file_test, device, topk, eval_type="test")

                        model = model.train()

                        if ndcg_last > ndcg_max:

                            ndcg_max = ndcg_last
                            best_epoch = i
                            early_stop = 0
                            best_hr = val_hr
                            best_ndcg = val_ndcg
                            best_model = copy.deepcopy(model)
                        
                        else:
                            early_stop += 1
                            if early_stop > args.early_stop:
                                return best_model, best_ndcg, best_hr
                        
                        print('BEST EPOCH:{}'.format(best_epoch))
                        print('EARLY STOP:{}'.format(early_stop))
                        print("best hr:")
                        print(best_hr)
                        print("best ndcg")
                        print(best_ndcg)
    return best_model, best_ndcg, best_hr
    


if __name__ == '__main__':
    topk=[1,3,5,10,20]
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    data_directory_train = './data/Amazon/train/'
    data_directory_test = './data/Amazon/test/'
    data_directory_valid = './data/Amazon/valid/' 
    data_directory_info = './data/Amazon/info/'
    # data = pd.read_csv(data_directory + '/train/train.csv')
    # find the one csv file with arg.data in the name
    data_file_train = [f for f in os.listdir(data_directory_train) if args.data in f and f.endswith('.csv')]
    data_file_test = [f for f in os.listdir(data_directory_test) if args.data in f and f.endswith('.csv')]
    data_file_valid = [f for f in os.listdir(data_directory_valid) if args.data in f and f.endswith('.csv')]
    data_file_info = [f for f in os.listdir(data_directory_info) if args.data in f and f.endswith('.txt')]

    print(data_file_train)

    assert len(data_file_train) == 1, "There should be only one csv file with the name containing " + args.data
    assert len(data_file_test) == 1, "There should be only one csv file with the name containing " + args.data
    assert len(data_file_valid) == 1, "There should be only one csv file with the name containing " + args.data
    assert len(data_file_info) == 1, "There should be only one txt file with the name containing " + args.data
    data_file_train = data_file_train[0]
    data_file_test = data_file_test[0]
    data_file_valid = data_file_valid[0]
    data_file_info = data_file_info[0]

    with open(os.path.join(data_directory_info, data_file_info), 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
        data_info = info

    # data_info = pd.read_csv(os.path.join(data_directory_info, data_file_info))
    seq_size = 10  # the length of history to define the seq
    item_num = len(data_info)  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model, test_ndcg, test_hr = main(topk, data_file_train, data_file_test, data_file_valid)
    # temp = main(topk)
    result_dict = {}
    result_dict["NDCG"] = {}
    result_dict["HR"] = {}
    for i,k in enumerate(topk):
        result_dict["NDCG"][k] = test_ndcg[i]
        result_dict["HR"][k] = test_hr[i]

    result_folder = ""
    for path_name in args.result_json_path.split("/")[:-1]:
        result_folder += path_name + "/"

    os.makedirs(result_folder, exist_ok=True)
    with open(args.result_json_path ,'w',encoding='utf-8') as f:
        json.dump(result_dict, f,ensure_ascii=False, indent=1)
    # torch.save(best_model, result_folder + f"/best_{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}")
    torch.save(best_model.state_dict(), result_folder + f"/best_{args.data}_{args.model}_emb{args.hidden_factor}_bs{args.batch_size}_lr{args.lr}_decay{args.l2_decay}_seed{args.seed}_loss_{args.loss_type}_dropout{args.dropout_rate}_state.pth")

    evaluate_games(best_model, data_file_test, device, topk, save_logits=True, eval_type="test")

