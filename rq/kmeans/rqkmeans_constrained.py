#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ-KMeans with Constrained Balanced Clustering
===============================================
Uses k-means-constrained to ensure balanced cluster sizes
"""

import os
import numpy as np
import polars as pl
import time
import argparse
import json
from collections import defaultdict

try:
    from k_means_constrained import KMeansConstrained
    HAS_CONSTRAINED = True
except ImportError:
    HAS_CONSTRAINED = False
    print("Warning: k-means-constrained not available")
    print("Install with: pip install k-means-constrained")


def balanced_kmeans_level_constrained(X, K, max_iter=100, tol=1e-7, random_state=None, verbose=False):
    """对单层向量执行带簇容量上下界的 KMeans，限制码本使用不均衡。

    Args:
        X (numpy.ndarray): 待聚类或计算距离的浮点向量矩阵，shape [N,D] 或一个批次 [B,D]。
        K (int): 每层聚类中心数量；数据预处理函数中表示用户和商品的最小交互次数。
        max_iter (int): KMeans 优化的最大迭代次数。
        tol (float): KMeans 收敛容差。
        random_state (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: 标签 [N] 与中心 [K,D]。
    """
    start_time = time.time()
    n, d = X.shape
    X = X.astype(np.float32, copy=False)

    # Calculate min and max cluster size
    min_size = max(1, n // K - 1)  # allow some imbalance
    max_size = n // K + 1

    if verbose:
        print(f"    Starting constrained K-means with K={K}, n={n}, d={d}")
        print(f"    Cluster size constraints: [{min_size}, {max_size}]")

    # Use k-means-constrained
    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=min_size,
        size_max=max_size,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=3,
        verbose=verbose,
        n_jobs=16
    )

    # Train and get labels
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    print(f"[Time] balanced_kmeans_level_constrained (K={K}): {time.time() - start_time:.2f}s")

    if verbose:
        # Check cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return labels, centroids


def residual_kmeans_constrained(X, K, L, max_iter=300, tol=1e-4, random_state=None, verbose=False):
    """逐层聚类当前残差并减去选中中心，得到多层代码与重建向量。

    Args:
        X (numpy.ndarray): 待聚类或计算距离的浮点向量矩阵，shape [N,D] 或一个批次 [B,D]。
        K (int | list[int]): 每层码本大小；单个整数会重复 L 次，列表长度需为 L。
        L (int): 残差量化层数，每层产生一个整数代码。
        max_iter (int): KMeans 优化的最大迭代次数。
        tol (float): KMeans 收敛容差。
        random_state (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        verbose (bool): 是否打印聚类、编码或统计过程的详细信息。

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray], numpy.ndarray]: codes [L,N]、各层码本 [K_l,D]、重建 [N,D]。
    """
    total_start = time.time()
    n, d = X.shape
    Ks = ([K] * L) if isinstance(K, int) else list(K)
    assert len(Ks) == L

    X = X.astype(np.float32, copy=False)
    R = X.copy()
    codes_all = np.empty((L, n), dtype=np.int32)
    codebooks = []

    for l in range(L):
        level_start = time.time()
        k_l = Ks[l]
        if verbose:
            mse_before = np.mean(R ** 2)
            print(f"\n=== Level {l+1}/{L} | K={k_l} ===")
            print(f"  Residual MSE before clustering: {mse_before:.6f}")

        # Generate random seed for sub-level
        seed_l = None if random_state is None else int(np.random.RandomState(random_state + l).randint(0, 2**31 - 1))

        codes_l, C_l = balanced_kmeans_level_constrained(
            R, k_l, max_iter=max_iter, tol=tol, random_state=seed_l, verbose=verbose
        )

        codes_all[l] = codes_l
        codebooks.append(C_l)

        # Subtract reconstructed part from residual
        R -= C_l[codes_l]

        print(f"[Time] Level {l+1}: {time.time() - level_start:.2f}s")
        if verbose:
            mse_after = np.mean(R ** 2)
            print(f"  Residual MSE after Level {l+1}: {mse_after:.6f}")

    recon = X - R
    print(f"[Time] residual_kmeans_constrained total: {time.time() - total_start:.2f}s")

    if verbose:
        total_mse = np.mean((X - recon) ** 2)
        print(f"\nFinal reconstruction MSE: {total_mse:.6f}")

    return codes_all, codebooks, recon


def deal_with_deduplicate(df):
    """对重复完整代码追加组内序号，使不同商品得到可区分的完整路径。

    Args:
        df (polars.DataFrame): 包含 codes 列的代码表，每行是一个商品的多层整数代码。

    Returns:
        polars.DataFrame: codes 列已追加消歧层的表。
    """
    df_with_index = df.with_row_index()

    result_df = df_with_index.with_columns(
        pl.when(pl.len().over("codes") > 1)
        .then(
            pl.col("codes").list.concat(
                pl.col("index").rank(method="ordinal").over("codes").cast(pl.Int64)
            )
        )
        .otherwise(pl.col("codes"))
        .alias("codes")
    ).drop("index")

    return result_df


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
            print(f"\n{title}")
        print(f"  Total items: {N}")
        for l in range(M):
            unique_count = len(np.unique(codes[:, l]))
            print(f"  Level {l+1}: unique codes = {unique_count}")

        # Check collision rate
        combos = len(set(map(tuple, codes)))
        collision_rate = 1 - combos / N
        print(f"  Unique full-paths: {combos}")
        print(f"  Collision rate: {collision_rate:.4f}")
    return


def parse_args():
    """解析当前脚本的命令行参数，返回后续数据或模型构造配置。

    Args:
        无显式参数。

    Returns:
        argparse.Namespace: 当前入口定义的参数集合。
    """
    parser = argparse.ArgumentParser(description="Constrained RQ-KMeans clustering")
    parser.add_argument('--root', type=str, default="./data/Amazon", help="Root directory for data")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Industrial_and_Scientific)")
    parser.add_argument("--k", type=int, default=256, help="Number of clusters per level")
    parser.add_argument("--l", type=int, default=4, help="Number of levels")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check k-means-constrained availability
    if not HAS_CONSTRAINED:
        print("Error: k-means-constrained is required but not installed")
        print("Install with: pip install k-means-constrained")
        exit(1)

    # Load data
    print("=" * 60)
    print(f"RQ-KMeans Constrained Training")
    print("=" * 60)

    t0 = time.time()
    print("root: ", args.root)
    print("dataset: ", args.dataset)
    data_path = os.path.join(args.root, args.dataset + '.emb-qwen-td.npy')

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        exit(1)

    embeddings = np.load(data_path).astype(np.float32)
    print(f"Loaded embeddings from: {data_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"[Time: {time.time()-t0:.2f}s]\n")

    K_values = [args.k] * args.l

    # Run residual K-means
    t1 = time.time()
    codes_all, codebooks, recon = residual_kmeans_constrained(
        embeddings, K=K_values, L=args.l, random_state=args.seed,
        verbose=args.verbose, max_iter=args.max_iter
    )
    print(f"\n[Time] Total training time: {time.time()-t1:.2f}s")

    # Analyze codes
    analyze_codes(codes_all.T, title="Code Statistics:", verbose=True)

    # Save codebooks
    output_dir = args.root
    os.makedirs(output_dir, exist_ok=True)

    t2 = time.time()
    codebook_path = os.path.join(output_dir, f'{args.dataset}.codebooks_constrained.npz')
    np.savez_compressed(codebook_path,
                       **{f'codebook_{i}': cb for i, cb in enumerate(codebooks)})
    print(f"\n[Time] Saved codebooks: {time.time()-t2:.2f}s")
    print(f"Codebooks saved to: {codebook_path}")

    # Prepare codes (+1 offset for token format)
    codes_plus_one = codes_all.T + 1
    codes_df = pl.DataFrame({'codes': [list(c) for c in codes_plus_one]})

    # Deduplication
    t4 = time.time()
    codes_dedup = deal_with_deduplicate(codes_df)
    print(f"[Time] Deduplication: {time.time()-t4:.2f}s")

    # Save original codes (not deduplicated)
    codes_path = os.path.join(output_dir, f'{args.dataset}.codes_constrained.npy')
    np.save(codes_path, codes_all.T)
    print(f"Codes saved to: {codes_path}")

    # Generate JSON index
    t5 = time.time()
    codes_json = {}
    for id, row in enumerate(codes_dedup.iter_rows(named=True)):
        codes_ = []
        for i, code in enumerate(row['codes']):
            codes_.append(f'<{chr(97+i)}_{code}>')
        codes_json[str(id)] = codes_

    # Save JSON index
    json_path = os.path.join(output_dir, f'{args.dataset}.index.json')
    with open(json_path, 'w') as f:
        json.dump(codes_json, f, indent=2)
    print(f"[Time] JSON index generation: {time.time()-t5:.2f}s")
    print(f"JSON index saved to: {json_path}")

    # Print final statistics
    print("\n" + "=" * 60)
    print("Final Statistics:")
    print("=" * 60)
    print(f"- Original data shape: {embeddings.shape}")
    print(f"- Number of levels: {args.l}")
    print(f"- K values per level: {K_values}")
    print(f"- Final reconstruction error (MSE): {np.mean((embeddings - recon) ** 2):.6f}")

    # Deduplication statistics
    codes_str = codes_df.with_columns(
        pl.col("codes").map_elements(lambda x: ','.join(map(str, x)), return_dtype=pl.Utf8).alias("codes_str")
    )
    duplicates = (codes_str
                  .group_by("codes_str")
                  .count()
                  .filter(pl.col("count") > 1)
                  .sort("count", descending=True))

    if len(duplicates) > 0:
        print(f"\nDeduplication Statistics:")
        print(f"- Number of duplicate groups: {len(duplicates)}")
        print(f"- Total duplicates: {duplicates['count'].sum() - len(duplicates)}")
        print(f"- Largest duplicate group size: {duplicates['count'].max()}")
    else:
        print("\nNo duplicates found!")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
