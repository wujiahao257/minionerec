import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionwiseFeedForward(nn.Module):
    """以两个 kernel=1 的卷积执行逐位置前馈变换，并加入残差和归一化。

    Args:
        d_in (int): 逐位置前馈层的输入与输出通道数。
        d_hid (int): 逐位置前馈层中间通道数。
        dropout (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        """初始化 PositionwiseFeedForward：以两个 kernel=1 的卷积执行逐位置前馈变换，并加入残差和归一化。

        Args:
            self (PositionwiseFeedForward): 当前实例，由 Python 在调用实例方法时自动传入。
            d_in (int): 逐位置前馈层的输入与输出通道数。
            d_hid (int): 逐位置前馈层中间通道数。
            dropout (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """转置后执行两层逐位置卷积，加入残差并进行层归一化。

        Args:
            self (PositionwiseFeedForward): 当前实例，由 Python 在调用实例方法时自动传入。
            x (torch.Tensor): 输入序列表示 [B,T,H]，内部转置为 [B,H,T] 供 Conv1d 处理。

        Returns:
            torch.Tensor: 与输入同形状 [B,T,H]。
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output



class MultiHeadAttention(nn.Module):
    """通过拆分 batch 维实现多头因果注意力，同时屏蔽 padding 并加入查询残差。

    Args:
        hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
        num_units (int): Q/K/V 线性投影宽度；当前分头实现按 hidden_size 切分，通常设为相同值。
        num_heads (int): 注意力头数；隐藏维度必须能被它整除。
        dropout_rate (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。
    """
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        """初始化 MultiHeadAttention：通过拆分 batch 维实现多头因果注意力，同时屏蔽 padding 并加入查询残差。

        Args:
            self (MultiHeadAttention): 当前实例，由 Python 在调用实例方法时自动传入。
            hidden_size (int): 隐藏表示维度；基线中也是商品 Embedding 的宽度。
            num_units (int): Q/K/V 线性投影宽度；当前分头实现按 hidden_size 切分，通常设为相同值。
            num_heads (int): 注意力头数；隐藏维度必须能被它整除。
            dropout_rate (float): Dropout 丢弃概率；训练时随机屏蔽部分表示，eval 模式禁用随机丢弃。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        """计算多头 Q/K/V 注意力，屏蔽 padding 与未来位置，再合头并加查询残差。

        Args:
            self (MultiHeadAttention): 当前实例，由 Python 在调用实例方法时自动传入。
            queries (torch.Tensor): 查询表示，shape [B,Tq,H]，在注意力输出后还用于残差相加。
            keys (torch.Tensor): 键和值的输入表示，shape [B,Tk,H]，同时用于构造 padding 掩码。

        Returns:
            torch.Tensor: shape [B,Tq,H] 的注意力表示。
        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res
        