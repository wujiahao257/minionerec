import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):

    """用可学习码本量化输入，支持最近邻或 Sinkhorn 分配，并以直通估计传递梯度。

    Args:
        n_e (int): 聚类中心或单层码本中的向量数量 K。
        e_dim (int): 单个码本向量及量化器输入的末维大小 d，须与 encoder 输出维度一致。
        beta (float): 量化 commitment loss 的权重，不是 RL 中的 KL 系数。
        kmeans_init (bool): 是否在首个训练 forward 时用当前 batch 的 KMeans 中心初始化码本。
        kmeans_iters (int): KMeans 优化的最大迭代次数。
        sk_epsilon (float): 本层 Sinkhorn 温度；小于等于 0 时关闭平衡分配。
        sk_iters (int): Sinkhorn 交替归一化的迭代次数。
    """
    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,):
        """初始化 VectorQuantizer：用可学习码本量化输入，支持最近邻或 Sinkhorn 分配，并以直通估计传递梯度。

        Args:
            self (VectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。
            n_e (int): 聚类中心或单层码本中的向量数量 K。
            e_dim (int): 单个码本向量及量化器输入的末维大小 d，须与 encoder 输出维度一致。
            beta (float): 量化 commitment loss 的权重，不是 RL 中的 KL 系数。
            kmeans_init (bool): 是否在首个训练 forward 时用当前 batch 的 KMeans 中心初始化码本。
            kmeans_iters (int): KMeans 优化的最大迭代次数。
            sk_epsilon (float): 本层 Sinkhorn 温度；小于等于 0 时关闭平衡分配。
            sk_iters (int): Sinkhorn 交替归一化的迭代次数。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        """返回当前可学习码本权重。

        Args:
            self (VectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            torch.nn.Parameter: shape [n_e,e_dim]。
        """
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        """按整数索引查码本向量，并按需重塑输出。

        Args:
            self (VectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。
            indices (torch.LongTensor): 码本行号 Tensor，值应在 [0,n_e) 内。
            shape (tuple[int, ...] | torch.Size | None): 查表后向量的目标形状；None 时不额外 reshape。

        Returns:
            torch.Tensor: 默认 shape 为 indices.shape + [e_dim]。
        """
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        """以输入样本的 KMeans 中心初始化当前码本并标记已初始化。

        Args:
            self (VectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。
            data (torch.Tensor): 初始化码本的 latent 样本，shape [B,e_dim]。

        Returns:
            None: 原地写入 embedding.weight。
        """
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        """按全局距离最大最小值中心化并缩放距离，以稳定 Sinkhorn 指数运算。

        Args:
            distances (torch.Tensor): 样本到各码本中心的距离矩阵，shape [B,K]。

        Returns:
            torch.Tensor: 与输入同形状的中心化距离矩阵。
        """
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        """计算码本距离、选择代码并构造量化损失，用直通估计保留输入梯度。

        Args:
            self (VectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。
            x (torch.Tensor): 待量化 latent 或残差，shape [...,e_dim]；商品训练中为 [B,e_dim]。
            use_sk (bool): 是否允许 Sinkhorn 分配；还需该层 sk_epsilon>0，否则仍用最近邻。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 量化向量与输入同形、损失标量、去掉末维后的代码索引。
        """
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


