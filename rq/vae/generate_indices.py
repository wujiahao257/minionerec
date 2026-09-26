"""从 RQ-VAE checkpoint 导出商品的 SID 索引。"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import EmbDataset
from .models.rqvae import RQVAE


"""
检查每件商品的完整 SID 是否唯一。

Args:
    all_indices_str: np.ndarray，完整 SID 的字符串数组。

Returns:
    bool，所有完整 SID 都唯一时为 True。
"""
def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


"""
统计各个完整 SID 对应的商品数。

Args:
    all_indices_str: np.ndarray，完整 SID 的字符串数组。

Returns:
    dict[str, int]，完整 SID 到商品数的映射。
"""
def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


"""
找出共用一个完整 SID 的商品行号。

Args:
    all_indices_str: np.ndarray，完整 SID 的字符串数组。

Returns:
    list[list[int]]，每组相同 SID 对应的商品行号。
"""
def get_collision_item(all_indices_str):
    index2id = {}
    for item, index in enumerate(all_indices_str):
        index2id.setdefault(index, []).append(item)
    return [items for items in index2id.values() if len(items) > 1]


"""
解析索引导出的命令行参数。

Returns:
    argparse.Namespace，checkpoint、商品向量和输出位置等配置。
"""
def parse_args():
    parser = argparse.ArgumentParser(description="从 RQ-VAE checkpoint 导出商品 SID 索引")
    parser.add_argument("--ckpt_path", required=True, help="RQ-VAE checkpoint 文件")
    parser.add_argument("--data_path", help="商品向量文件；默认使用 checkpoint 中的路径")
    parser.add_argument("--dataset", help="输出文件的数据集名；默认取向量文件名第一个点之前的部分")
    parser.add_argument("--output_path", help="输出 JSON 路径；默认写到向量文件所在目录")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


"""
加载 checkpoint、生成 SID 并写出商品编号到 SID 的 JSON。

Args:
    cli_args: argparse.Namespace，命令行解析后的导出配置。
"""
def main(cli_args):
    # 旧训练入口从 rq 目录运行，checkpoint 可能保存了相对于该目录的向量路径。
    rq_dir = Path(__file__).resolve().parents[1]
    ckpt = torch.load(cli_args.ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    data_path = Path(cli_args.data_path or train_args.data_path).expanduser()
    if not data_path.is_absolute() and cli_args.data_path is None:
        data_path = (rq_dir / data_path).resolve()

    dataset = cli_args.dataset or data_path.name.split(".")[0]
    output_path = Path(cli_args.output_path) if cli_args.output_path else data_path.parent / f"{dataset}.index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli_args.device)

    data = EmbDataset(str(data_path))
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_epsilons=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(
        data,
        num_workers=train_args.num_workers,
        batch_size=cli_args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    all_indices = []
    all_indices_str = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            indices = model.get_indices(batch.to(device), use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = [prefix[level].format(int(ind)) for level, ind in enumerate(index)]
                all_indices.append(code)
                all_indices_str.append(str(code))

        all_indices = np.array(all_indices)
        all_indices_str = np.array(all_indices_str)

        for vq in model.rq.vq_layers[:-1]:
            vq.sk_epsilon = 0.0
        if model.rq.vq_layers[-1].sk_epsilon == 0.0:
            model.rq.vq_layers[-1].sk_epsilon = 0.003

        # 与原脚本一致：仅重分配发生冲突的商品，最多尝试 20 轮。
        for _ in range(20):
            if check_collision(all_indices_str):
                break
            collision_item_groups = get_collision_item(all_indices_str)
            print(f"Collision groups: {len(collision_item_groups)}")
            for collision_items in collision_item_groups:
                indices = model.get_indices(data[collision_items].to(device), use_sk=True)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item, index in zip(collision_items, indices):
                    code = [prefix[level].format(int(ind)) for level, ind in enumerate(index)]
                    all_indices[item] = code
                    all_indices_str[item] = str(code)

    print("All indices number:", len(all_indices))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))
    print("Collision Rate:", (len(all_indices_str) - len(set(all_indices_str.tolist()))) / len(all_indices_str))

    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(all_indices_dict, fp)
    print(f"SID index saved to: {output_path}")


if __name__ == "__main__":
    main(parse_args())
