import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    """通过 MLP 编码、残差量化与 MLP 解码学习商品离散表示；此实现无高斯先验采样。

    Args:
        in_dim (int): 商品连续向量的特征维度 D。
        num_emb_list (list[int] | None): 每层码本大小列表，长度决定量化层数；构造模型时需要实际列表。
        e_dim (int): 单个码本向量及量化器输入的末维大小 d，须与 encoder 输出维度一致。
        layers (list[int] | None): MLP 的层宽列表；RQVAE 中只传中间层，MLPLayers 中包含首尾维度。
        dropout_prob (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
        bn (bool): 是否在 MLP 的非末层加入 BatchNorm1d。
        loss_type (str): 向量重建损失类型：mse 或 l1。
        quant_loss_weight (float): 量化损失相对于重建损失的权重。
        beta (float): 量化 commitment loss 的权重，不是 RL 中的 KL 系数。
        kmeans_init (bool): 是否在首个训练 forward 时用当前 batch 的 KMeans 中心初始化码本。
        kmeans_iters (int): KMeans 优化的最大迭代次数。
        sk_epsilons (list[float] | None): 各层 Sinkhorn 温度，必须与码本层数对应；非正值关闭相应层的平衡分配。
        sk_iters (int): Sinkhorn 交替归一化的迭代次数。
    """
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        """初始化 RQVAE：通过 MLP 编码、残差量化与 MLP 解码学习商品离散表示；此实现无高斯先验采样。

        Args:
            self (RQVAE): 当前实例，由 Python 在调用实例方法时自动传入。
            in_dim (int): 商品连续向量的特征维度 D。
            num_emb_list (list[int] | None): 每层码本大小列表，长度决定量化层数；构造模型时需要实际列表。
            e_dim (int): 单个码本向量及量化器输入的末维大小 d，须与 encoder 输出维度一致。
            layers (list[int] | None): MLP 的层宽列表；RQVAE 中只传中间层，MLPLayers 中包含首尾维度。
            dropout_prob (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
            bn (bool): 是否在 MLP 的非末层加入 BatchNorm1d。
            loss_type (str): 向量重建损失类型：mse 或 l1。
            quant_loss_weight (float): 量化损失相对于重建损失的权重。
            beta (float): 量化 commitment loss 的权重，不是 RL 中的 KL 系数。
            kmeans_init (bool): 是否在首个训练 forward 时用当前 batch 的 KMeans 中心初始化码本。
            kmeans_iters (int): KMeans 优化的最大迭代次数。
            sk_epsilons (list[float] | None): 各层 Sinkhorn 温度，必须与码本层数对应；非正值关闭相应层的平衡分配。
            sk_iters (int): Sinkhorn 交替归一化的迭代次数。

        Returns:
            None: 完成实例初始化。
        """
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        """依次执行 MLP 编码、残差量化和解码重建。

        Args:
            self (RQVAE): 当前实例，由 Python 在调用实例方法时自动传入。
            x (torch.Tensor): 原始商品向量 [B,in_dim]，编码后在 e_dim 空间量化。
            use_sk (bool): 是否允许 Sinkhorn 分配；还需该层 sk_epsilon>0，否则仍用最近邻。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 重建 [B,D]、量化损失标量、代码 [B,M]。
        """
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """仅执行编码器和残差量化，提取商品代码，不经过重建 decoder。

        Args:
            self (RQVAE): 当前实例，由 Python 在调用实例方法时自动传入。
            xs (torch.Tensor): 原始商品连续向量，shape [B,in_dim]；作为编码输入或重建损失的监督目标，compute_loss 虽默认 None 但计算时必须提供。
            use_sk (bool): 是否允许 Sinkhorn 分配；还需该层 sk_epsilon>0，否则仍用最近邻。

        Returns:
            torch.LongTensor: shape [B,M] 的多层代码。
        """
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs=None):

        """计算输入向量的重建误差，并加上加权量化损失。

        Args:
            self (RQVAE): 当前实例，由 Python 在调用实例方法时自动传入。
            out (torch.Tensor): decoder 产生的重建商品向量，shape [B,in_dim]，与原始 xs 逐元素计算重建误差。
            quant_loss (torch.Tensor): 零维损失 Tensor；quant_loss 为各层量化损失的平均值。
            xs (torch.Tensor | None): 原始商品连续向量，shape [B,in_dim]；作为编码输入或重建损失的监督目标，compute_loss 虽默认 None 但计算时必须提供。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 总损失标量和重建损失标量。
        """
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon