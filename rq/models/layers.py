import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans


class MLPLayers(nn.Module):

    """按维度列表构造带可选 Dropout、BatchNorm 和激活的多层感知机。

    Args:
        layers (list[int] | None): MLP 的层宽列表；RQVAE 中只传中间层，MLPLayers 中包含首尾维度。
        dropout (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
        activation (str | type[torch.nn.Module] | None): 激活名称或模块类；none/None 表示不加激活，MLP 最后一层不加激活。
        bn (bool): 是否在 MLP 的非末层加入 BatchNorm1d。
    """
    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        """初始化 MLPLayers：按维度列表构造带可选 Dropout、BatchNorm 和激活的多层感知机。

        Args:
            self (MLPLayers): 当前实例，由 Python 在调用实例方法时自动传入。
            layers (list[int] | None): MLP 的层宽列表；RQVAE 中只传中间层，MLPLayers 中包含首尾维度。
            dropout (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
            activation (str | type[torch.nn.Module] | None): 激活名称或模块类；none/None 表示不加激活，MLP 最后一层不加激活。
            bn (bool): 是否在 MLP 的非末层加入 BatchNorm1d。

        Returns:
            None: 完成实例初始化。
        """
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers)-2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        """将 Linear 权重 Xavier 正态初始化并把其偏置置零。

        Args:
            self (MLPLayers): 当前实例，由 Python 在调用实例方法时自动传入。
            module (torch.nn.Module): 由 Module.apply 递归传入的子模块，只对 Linear 执行 Xavier 初始化。

        Returns:
            None: 原地修改指定子模块。
        """
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        """按初始化时构建的 Linear、归一化和激活顺序变换输入。

        Args:
            self (MLPLayers): 当前实例，由 Python 在调用实例方法时自动传入。
            input_feature (torch.Tensor): MLP 输入，末维需等于 layers[0]；商品训练中 shape [B,D]。

        Returns:
            torch.Tensor: 末维由 layers[0] 变为 layers[-1]。
        """
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):

    """按名称或模块类创建激活函数。

    Args:
        activation_name (str | type[torch.nn.Module] | None): 激活名称或模块类；none/None 表示不加激活，MLP 最后一层不加激活。
        emb_dim (int | None): 预留的激活特征维度参数；当前激活工厂没有使用。

    Returns:
        torch.nn.Module | None: 激活模块；none/None 返回 None。
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
):
    """将 Tensor 样本移到 CPU，用 sklearn KMeans 拟合中心，再移回原设备。

    Args:
        samples (torch.Tensor): 用于 KMeans 初始化的样本矩阵，shape [B,d]，会复制到 CPU NumPy。
        num_clusters (int): 聚类中心或单层码本中的向量数量 K。
        num_iters (int): KMeans 优化的最大迭代次数。

    Returns:
        torch.Tensor: shape [num_clusters,d] 的中心。
    """
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    """对 exp(-distance/epsilon) 反复按样本和中心维归一化，形成平衡分配分数。

    Args:
        distances (torch.Tensor): 样本到各码本中心的距离矩阵，shape [B,K]。
        epsilon (float): Sinkhorn 的正则温度，需大于零以避免除零。
        sinkhorn_iterations (int): Sinkhorn 交替归一化的迭代次数。

    Returns:
        torch.Tensor: shape [B,K] 的分配矩阵，函数不计算梯度。
    """
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q