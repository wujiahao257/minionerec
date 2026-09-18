import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """逐层量化剩余残差，累加量化向量并返回各层代码。

    Args:
        n_e_list (list[int]): 各量化层的码本大小，列表长度决定层数；必须提供实际列表。
        e_dim (int): 单个码本向量及量化器输入的末维大小 d，须与 encoder 输出维度一致。
        sk_epsilons (list[float]): 各层 Sinkhorn 温度，必须与码本层数对应；非正值关闭相应层的平衡分配。
        beta (float): 量化 commitment loss 的权重，不是 RL 中的 KL 系数。
        kmeans_init (bool): 是否在首个训练 forward 时用当前 batch 的 KMeans 中心初始化码本。
        kmeans_iters (int): KMeans 优化的最大迭代次数。
        sk_iters (int): Sinkhorn 交替归一化的迭代次数。

    References:
        SoundStream: An End-to-End Neural Audio Codec.
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        """初始化 ResidualVectorQuantizer：逐层量化剩余残差，累加量化向量并返回各层代码。

        Args:
            self (ResidualVectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。
            n_e_list (list[int]): 各量化层的码本大小，列表长度决定层数；必须提供实际列表。
            e_dim (int): 单个码本向量及量化器输入的末维大小 d，须与 encoder 输出维度一致。
            sk_epsilons (list[float]): 各层 Sinkhorn 温度，必须与码本层数对应；非正值关闭相应层的平衡分配。
            beta (float): 量化 commitment loss 的权重，不是 RL 中的 KL 系数。
            kmeans_init (bool): 是否在首个训练 forward 时用当前 batch 的 KMeans 中心初始化码本。
            kmeans_iters (int): KMeans 优化的最大迭代次数。
            sk_iters (int): Sinkhorn 交替归一化的迭代次数。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        """堆叠各层码本；各层大小需相同才能 stack。

        Args:
            self (ResidualVectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            torch.Tensor: shape [M,K,d] 的码本。
        """
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        """逐层量化并减去残差，累加各层向量、平均损失并堆叠索引。

        Args:
            self (ResidualVectorQuantizer): 当前实例，由 Python 在调用实例方法时自动传入。
            x (torch.Tensor): 待量化 latent 或残差，shape [...,e_dim]；商品训练中为 [B,e_dim]。
            use_sk (bool): 是否允许 Sinkhorn 分配；还需该层 sk_epsilon>0，否则仍用最近邻。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 量化和 [...,d]、平均损失标量、各层代码 [...,M]。
        """
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices