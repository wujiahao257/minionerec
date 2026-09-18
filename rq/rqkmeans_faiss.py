#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS ResidualQuantizer  +  Sinkhorn-based Uniform Semantic Mapping
===================================================================
"""

import argparse
import json
import os
from collections import defaultdict

import faiss
import numpy as np
from tqdm import tqdm


def pairwise_sq_dists_batch(X, C, C_norm2=None):
    """用平方范数和矩阵乘法计算样本到中心的平方欧氏距离。

    Args:
        X (numpy.ndarray): 待聚类或计算距离的浮点向量矩阵，shape [N,D] 或一个批次 [B,D]。
        C (numpy.ndarray): 聚类中心矩阵，shape [K,D]。
        C_norm2 (numpy.ndarray | None): 预计算的中心平方范数，shape [K]；None 时现场计算。

    Returns:
        numpy.ndarray: shape [B,K] 或 [N,K] 的距离矩阵。
    """
    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)              # (K,)
    X_norm2 = np.sum(X * X, axis=1, keepdims=True)   # (B,1)
    dots = X @ C.T                                   # (B,K)
    return X_norm2 + C_norm2[None, :] - 2.0 * dots


def train_faiss_rq(data, num_levels=3, codebook_size=256, verbose=True):
    """按层数和码本位数训练 FAISS 残差量化器。

    Args:
        data (numpy.ndarray): 商品浮点向量矩阵，shape [N,D]。
        num_levels (int): 残差量化层数，每层产生一个整数代码。
        codebook_size (int): 每层码本大小，FAISS 实现按 int(log2(size)) 转成位数，预期为 2 的幂。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。

    Returns:
        faiss.ResidualQuantizer: 已训练的量化器。
    """
    N, d = data.shape
    if verbose:
        print("Training FAISS ResidualQuantizer")
        print(f"  data={N}  dim={d}  levels={num_levels}  "
              f"codebook={codebook_size}  total_codes={codebook_size ** num_levels:,}")

    nbits = int(np.log2(codebook_size))
    rq = faiss.ResidualQuantizer(d, num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1

    rq.train(np.ascontiguousarray(data.astype(np.float32)))
    if verbose:
        print("  training completed\n")
    return rq

def unpack_rq_codes(codes, nbits, num_levels):
    """按小端位序解包 FAISS 的紧凑代码字节。

    Args:
        codes (numpy.ndarray): FAISS 位打包字节矩阵 [N,ceil(num_levels*nbits/8)]，每行对应一件商品。
        nbits (int): 每一层代码占用的位数，用于解包 FAISS 紧凑编码。
        num_levels (int): 残差量化层数，每层产生一个整数代码。

    Returns:
        numpy.ndarray: int32 整数代码，shape [N,num_levels]。
    """
    N = codes.shape[0]
    # FAISS uses Little Endian packing
    packed_ints = np.zeros(N, dtype=np.int64)
    for i in range(codes.shape[1]):
        packed_ints |= codes[:, i].astype(np.int64) << (8 * i)
    unpacked_codes = np.zeros((N, num_levels), dtype=np.int32)
    mask = (1 << nbits) - 1  # e.g., mask for 9 bits is 511 (0x1FF)
    for i in range(num_levels):
        unpacked_codes[:, i] = (packed_ints >> (i * nbits)) & mask
    return unpacked_codes

def encode_with_rq(rq, data, codebook_size, verbose=True):
    """使用 FAISS 量化器编码向量，并在需要时解包非整字节位宽的代码。

    Args:
        rq (faiss.ResidualQuantizer): 已经构建或训练的 FAISS 残差量化器。
        data (numpy.ndarray): 商品浮点向量矩阵，shape [N,D]。
        codebook_size (int): 每层码本大小，FAISS 实现按 int(log2(size)) 转成位数，预期为 2 的幂。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。

    Returns:
        numpy.ndarray: 默认 8 位配置下为 [N,M] 的 int32 代码。
    """
    data = np.ascontiguousarray(data.astype(np.float32))
    nbits = int(np.log2(codebook_size))
    if verbose:
        print(f"Encoding {data.shape[0]} vectors ...")
    codes_packed = rq.compute_codes(data)
    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        codes = unpack_rq_codes(codes_packed, nbits, rq.M)
    if codes_packed.ndim == 1:
        n_bytes = (rq.M * nbits + 7) // 8
        codes_packed = codes_packed.reshape(-1, n_bytes)
    codes = codes.astype(np.int32)
    if verbose:
        print(f"  done, codes.shape={codes.shape}\n")
    return codes


def get_rq_codebooks(rq):
    """从 FAISS 量化器取出并重排各层中心权重。

    Args:
        rq (faiss.ResidualQuantizer): 已经构建或训练的 FAISS 残差量化器。

    Returns:
        numpy.ndarray: shape [M,K,D] 的 float32 码本。
    """
    M, d = rq.M, rq.d
    nbits0 = get_first_nbits(rq)      
    K = 1 << nbits0
    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
    return cb_flat.reshape(M, K, d)     # (M,K,d)


def compute_residuals_upto_level(rq, data, codes, upto_level, codebooks=None):
    """从原向量中减去指定数量的前层选中中心，得到当前层残差。

    Args:
        rq (faiss.ResidualQuantizer): 已经构建或训练的 FAISS 残差量化器。
        data (numpy.ndarray): 商品浮点向量矩阵，shape [N,D]。
        codes (numpy.ndarray): 商品多层整数代码矩阵，通常 shape [N,M]；解包函数接收位打包字节矩阵。
        upto_level (int): 需要减去的前序量化层数；处理层索引范围为 [0,upto_level)。
        codebooks (numpy.ndarray | None): 各层码本，shape [M,K,D]；None 时从 FAISS 量化器读取。

    Returns:
        numpy.ndarray: shape [N,D] 的残差副本。
    """
    if codebooks is None:
        codebooks = get_rq_codebooks(rq)
    residuals = np.ascontiguousarray(data.astype(np.float32)).copy()
    for l in range(upto_level):
        residuals -= codebooks[l][codes[:, l]]
    return residuals


def estimate_tau(residuals, centroids, sample_size=4000,
                 percentile=90, min_tau=1e-6):
    """抽样计算中心距离差的分布宽度，估计平衡分配使用的正则温度。

    Args:
        residuals (numpy.ndarray): 当前层尚未重建的残差向量，shape [N,D]。
        centroids (numpy.ndarray): 聚类中心矩阵，shape [K,D]。
        sample_size (int): 估计 Sinkhorn 温度时最多抽取的向量数。
        percentile (float): 估计距离差分布宽度使用的百分位数。
        min_tau (float): 温度估计结果的正数下限，避免温度为零。

    Returns:
        float: 不低于 min_tau 的温度。
    """
    N = residuals.shape[0]
    idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    X = residuals[idx]
    Cn2 = np.sum(centroids * centroids, axis=1)
    D = pairwise_sq_dists_batch(X, centroids, Cn2)
    spread = np.percentile(D - D.min(axis=1, keepdims=True),
                           percentile, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)


def sinkhorn_balance_level(residuals, centroids, capacities=None, *,
                           batch_size=8192, iters=30, tau=None,
                           verbose=True, topk=32, seed=42):
    """计算最优传输分配，再按中心剩余容量贪心选择每件商品的代码。

    Args:
        residuals (numpy.ndarray): 当前层尚未重建的残差向量，shape [N,D]。
        centroids (numpy.ndarray): 聚类中心矩阵，shape [K,D]。
        capacities (numpy.ndarray | None): 各中心目标容量，shape [K]，总和必须等于样本数；None 时尽量均匀分配。
        batch_size (int): 预留的批大小配置，仅传递和打印；当前 Sinkhorn 实现仍一次构造完整 [N,K] 距离矩阵。
        iters (int): 平衡过程的迭代配置；当前底层 ot.sinkhorn 调用未转发它作为迭代次数。
        tau (float | None): 最优传输的正则温度；None 时根据距离差自动估计。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。
        topk (int): 贪心容量分配优先检查的中心数；不足时从所有仍有容量的中心中选择。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。

    Returns:
        numpy.ndarray: shape [N] 的 int32 中心编号。
    """
    import ot                                     
    rng = np.random.RandomState(seed)
    N, d = residuals.shape
    K = centroids.shape[0]

    if capacities is None:
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1
    capacities = capacities.astype(np.int64)
    assert capacities.sum() == N

    if tau is None:
        tau = estimate_tau(residuals, centroids)
    if verbose:
        print(f"  Sinkhorn level: N={N}  K={K}  tau={tau:.5g}  "
              f"iters={iters}  batch={batch_size}")

    a = np.ones(N) / N                              
    b = capacities / float(N)                     
    Cn2 = np.sum(centroids * centroids, axis=1)


    def cost_fun(X):
        """用平方范数和矩阵乘法计算样本到中心的平方欧氏距离。

        Args:
            X (numpy.ndarray): 待聚类或计算距离的浮点向量矩阵，shape [N,D] 或一个批次 [B,D]。

        Returns:
            numpy.ndarray: shape [B,K] 或 [N,K] 的距离矩阵。
        """
        return pairwise_sq_dists_batch(X, centroids, Cn2)

    D_full = cost_fun(residuals).astype(np.float64)
    P = ot.sinkhorn(a, b, D_full, tau)             

    remaining = capacities.copy()
    assign = np.empty(N, dtype=np.int32)
    order = np.arange(N)
    rng.shuffle(order)
    for i in order:
        probs = P[i]
        if topk and topk < K:
            cand = np.argpartition(-probs, topk - 1)[:topk]
            cand = cand[np.argsort(-probs[cand])]
        else:
            cand = np.argsort(-probs)
        chosen = -1
        for c in cand:
            if remaining[c] > 0:
                chosen = c
                break
        if chosen < 0:
            avail = np.flatnonzero(remaining > 0)
            chosen = int(avail[np.argmax(probs[avail])])
        remaining[chosen] -= 1
        assign[i] = chosen

    if verbose:
        used = capacities - remaining
        print(f"    level balanced: min={used.min()}  max={used.max()}")

    return assign


def sinkhorn_uniform_mapping(rq, data, codes, *, batch_size=8192,
                             iters=30, tau=None, verbose=True,
                             topk=32, seed=42):
    """逐层重新计算残差并平衡分配，更新各层整数代码。

    Args:
        rq (faiss.ResidualQuantizer): 已经构建或训练的 FAISS 残差量化器。
        data (numpy.ndarray): 商品浮点向量矩阵，shape [N,D]。
        codes (numpy.ndarray): 商品多层整数代码矩阵，通常 shape [N,M]；解包函数接收位打包字节矩阵。
        batch_size (int): 预留的批大小配置，仅传递和打印；当前 Sinkhorn 实现仍一次构造完整 [N,K] 距离矩阵。
        iters (int): 平衡过程的迭代配置；当前底层 ot.sinkhorn 调用未转发它作为迭代次数。
        tau (float | None): 最优传输的正则温度；None 时根据距离差自动估计。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。
        topk (int): 贪心容量分配优先检查的中心数；不足时从所有仍有容量的中心中选择。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。

    Returns:
        numpy.ndarray: shape [N,M] 的平衡代码。
    """
    codebooks = get_rq_codebooks(rq)
    N, M = codes.shape
    K = codebooks.shape[1]

    codes_bal = codes.copy()
    for l in range(M):
        if verbose:
            print(f"\n=== Sinkhorn uniform mapping  level {l+1}/{M} ===")
        residuals = compute_residuals_upto_level(
            rq, data, codes_bal, upto_level=l, codebooks=codebooks)

        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1

        new_ids = sinkhorn_balance_level(
            residuals, codebooks[l], capacities=capacities,
            batch_size=batch_size, iters=iters, tau=tau,
            verbose=verbose, topk=topk, seed=seed + l)

        codes_bal[:, l] = new_ids
    return codes_bal


def analyze_codes(codes, title="", verbose=True):
    """打印每层使用的码数和完整路径碰撞率。

    Args:
        codes (numpy.ndarray): 商品多层整数代码矩阵，通常 shape [N,M]；解包函数接收位打包字节矩阵。
        title (str): 统计日志的标题，可为空字符串。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。

    Returns:
        None: 仅输出代码分布统计。
    """
    N, M = codes.shape
    if verbose:
        if title:
            print(title)
        print(f"  total={N}")
        for l in range(M):
            print(f"  L{l+1}: unique={len(np.unique(codes[:, l]))}")
        combos = len(set(map(tuple, codes)))
        print(f"  unique full-paths={combos}  "
              f"collision_rate={1 - combos / N:.4f}")
    return


def save_indices_json(codes, path, use_prefix=True):
    """将代码按数组行号写成商品 ID 到 SID token 或整数列表的映射。

    Args:
        codes (numpy.ndarray): 商品多层整数代码矩阵，通常 shape [N,M]；解包函数接收位打包字节矩阵。
        path (str): 当前读写操作使用的文件或目录路径。
        use_prefix (bool): 是否把整数代码写成带层前缀的 SID token；False 时保存整数列表。

    Returns:
        None: 写出索引 JSON。
    """
    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    idx = {}
    for i, code in enumerate(codes):
        if use_prefix:
            idx[i] = [tpl[j].format(int(c)) for j, c in enumerate(code)]
        else:
            idx[i] = [int(c) for c in code]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)
    print("Saved indices:", path)


def main():
    """训练 FAISS 残差量化器，可选平衡分配，输出 SID 索引与量化器文件。

    Args:
        无显式参数。

    Returns:
        None: 执行对应命令行流程并写出结果。
    """
    parser = argparse.ArgumentParser(
        description="FAISS-RQ + Sinkhorn uniform mapping")
    parser.add_argument("--dataset", default="Industrial_and_Scientific")
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--uniform", action="store_true",
                        help="enable Sinkhorn uniform mapping")
    parser.add_argument("--iters", type=int, default=30,
                        help="Sinkhorn iterations")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--output_root", default="../data")
    args = parser.parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = f"../data/Amazon/index/{args.dataset}.emb-qwen-td.npy"

    out_dir = os.path.join(args.output_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{args.dataset}.faiss-rq.index.json")
    out_faiss = out_json.replace(".json", ".faiss")

    print("loading:", data_path)
    data = np.load(data_path)
    print("shape:", data.shape)

    rq = train_faiss_rq(data, args.num_levels, args.codebook_size)
    codes_raw = encode_with_rq(rq, data, args.codebook_size, verbose=True)

    analyze_codes(codes_raw, "Before balancing:")

    if args.uniform:
        codes_bal = sinkhorn_uniform_mapping(
            rq, data, codes_raw,
            batch_size=args.batch_size,
            iters=args.iters,
            verbose=True)
        analyze_codes(codes_bal, "After  balancing:")
        codes_final = codes_bal
    else:
        codes_final = codes_raw

    save_indices_json(codes_final, out_json, use_prefix=True)

    try:
        nbits_val = get_first_nbits(rq)     
        index = faiss.IndexResidualQuantizer(rq.d, rq.M, nbits_val)
        index.rq = rq
        index.is_trained = True
        faiss.write_index(index, out_faiss)
        print("Saved faiss quantizer:", out_faiss)
    except Exception as e:
        print("save faiss index failed:", e)

def get_first_nbits(rq):
    """兼容 FAISS 的标量或向量位宽表示，读取第一层编码位数。

    Args:
        rq (faiss.ResidualQuantizer): 已经构建或训练的 FAISS 残差量化器。

    Returns:
        int: 第一层位宽。
    """
    if isinstance(rq.nbits, int):
        return rq.nbits            
    return int(faiss.vector_to_array(rq.nbits).ravel()[0])

if __name__ == "__main__":
    main()
