import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    """读取商品 NumPy 向量，清理 NaN/Inf，按索引返回 float32 Tensor。

    Args:
        data_path (str): 商品向量 .npy 文件路径，数组 shape [N,D]。
    """
    def __init__(self, data_path):

        """初始化 EmbDataset：读取商品 NumPy 向量，清理 NaN/Inf，按索引返回 float32 Tensor。

        Args:
            self (EmbDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_path (str): 商品向量 .npy 文件路径，数组 shape [N,D]。

        Returns:
            None: 完成实例初始化。
        """
        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        
        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0
            
        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0
            
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")
        
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        """按行号读取一个或一组商品向量，并转为 float32 Tensor。

        Args:
            self (EmbDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            index (int | list[int] | numpy.ndarray): 商品向量行号；支持单条索引和冲突组的批量索引。

        Returns:
            torch.Tensor: 单条 [D]，批量索引 [B,D]。
        """
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (EmbDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.embeddings)
