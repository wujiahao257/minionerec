#!/usr/bin/env python3
"""
Convert dataset (Office/Industrial_and_Scientific) to MiniOneRec format with semantic IDs
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
import argparse

def load_dataset(data_dir: str, dataset_name: str) -> Dict[str, Any]:
    """读取商品元数据、SID 索引和存在的 train/valid/test 原子交互文件。

    Args:
        data_dir (str): 数据集目录，内含同名前缀的 item/index JSON 和 train/valid/test inter 文件。
        dataset_name (str): 数据集名称，用来拼接数据文件名或标记结果。

    Returns:
        dict[str, object]: 包含 items、item_to_semantic、splits。
    """
    data = {}
    
    # Load item metadata (id -> {title, description, ...})
    with open(os.path.join(data_dir, f'{dataset_name}.item.json'), 'r') as f:
        data['items'] = json.load(f)
    
    # Load item_id to semantic tokens mapping from index.json
    with open(os.path.join(data_dir, f'{dataset_name}.index.json'), 'r') as f:
        data['item_to_semantic'] = json.load(f)
    
    # Load train/valid/test splits
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_file = os.path.join(data_dir, f'{dataset_name}.{split}.inter')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                splits[split] = [line.strip().split('\t') for line in lines if line.strip()]
    
    data['splits'] = splits
    return data

def semantic_tokens_to_id(tokens: List[str]) -> str:
    """按层顺序直接拼接 SID token，保留尖括号且不插入空格。

    Args:
        tokens (list[str]): 按量化层顺序排列的 SID token 字符串，拼接时保留尖括号。

    Returns:
        str: 一个商品的完整 SID 字符串。
    """
    # Keep brackets and concatenate directly (no spaces)
    return ''.join(tokens)

def create_item_info_file(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    """关联 SID、标题和商品编号，写出供约束解码使用的商品目录。

    Args:
        items (dict[str, dict[str, object]]): 商品编号到元数据字典的映射，包含标题、描述等字段。
        item_to_semantic (dict[str, list[str]]): 商品编号到多层 SID token 列表的映射。
        output_path (str): 当前读写操作使用的文件或目录路径。

    Returns:
        None: 写出三列制表符分隔的 info 文件。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item_id, item_data in items.items():
            # Get semantic ID from index mapping
            if item_id in item_to_semantic:
                semantic_tokens = item_to_semantic[item_id]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                # Get item title, fallback to Item_id if not available
                item_title = item_data.get('title', f'Item_{item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{item_id}\n")

def convert_interactions_to_csv(splits: Dict[str, List], items: Dict[str, Dict], 
                               item_to_semantic: Dict[str, List], output_dir: str, category: str = "Office_Products",
                               max_valid_samples: int = None, max_test_samples: int = None, seed: int = 42,
                               keep_longest_only: bool = True):
    """把原子交互中的商品编号关联到标题与 SID，可抽样并写出训练格式 CSV。

    Args:
        splits (dict[str, list[list[str]]]): train/valid/test 对应的交互行，每行包含用户、历史 ID 序列、目标 ID；GPR 还可含上下文。
        items (dict[str, dict[str, object]]): 商品编号到元数据字典的映射，包含标题、描述等字段。
        item_to_semantic (dict[str, list[str]]): 商品编号到多层 SID token 列表的映射。
        output_dir (str): 输出目录；转换入口写数据，训练入口写 checkpoint 和 tokenizer。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        max_valid_samples (int | None): 相应验证集或测试集的抽样上限；None 表示不设上限。
        max_test_samples (int | None): 相应验证集或测试集的抽样上限；None 表示不设上限。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        keep_longest_only (bool): 训练集是否每用户仅保留最长历史；等长时保留先遇到的样本，不保证是最新窗口。

    Returns:
        None: 写出 CSV；缺失目标 SID 的记录被跳过。
    """
    
    import random
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        rows = []
        user_to_longest = {}  # For train data: keep only longest sequence per user
        
        for line in split_data:
            if len(line) != 3:
                continue
                
            user_id, item_sequence, target_item = line
            
            # Parse item sequence - these are item_ids from the interaction data
            if item_sequence.strip():
                history_item_ids = [int(x) for x in item_sequence.split()]
            else:
                history_item_ids = []
            
            target_item_id = int(target_item)
            
            # Convert item_ids to semantic_ids using index mapping
            history_semantic_ids = []
            for item_id in history_item_ids:
                if str(item_id) in item_to_semantic:
                    semantic_tokens = item_to_semantic[str(item_id)]
                    semantic_id = semantic_tokens_to_id(semantic_tokens)
                    history_semantic_ids.append(semantic_id)
            
            # Get target semantic_id
            target_semantic_id = None
            if str(target_item_id) in item_to_semantic:
                semantic_tokens = item_to_semantic[str(target_item_id)]
                target_semantic_id = semantic_tokens_to_id(semantic_tokens)
            
            if target_semantic_id is None:
                continue  # Skip if no semantic_id found for target
            
            # Get item titles using item_ids (not semantic_ids)
            history_item_titles = []
            for item_id in history_item_ids:
                if str(item_id) in items:
                    title = items[str(item_id)].get('title', f'Item_{item_id}')
                    history_item_titles.append(title)
            
            # Target item title
            target_title = items.get(str(target_item_id), {}).get('title', f'Item_{target_item_id}')
            
            # Create row with required fields
            row = {
                'user_id': f'A{user_id}',  # Format like A2013JDMPUV6D9
                'history_item_title': history_item_titles,
                'item_title': target_title,
                'history_item_id': history_item_ids,  # Original item_ids
                'item_id': target_item_id,  # Original item_id
                'history_item_sid': history_semantic_ids,  # Semantic IDs (with brackets)
                'item_sid': target_semantic_id  # Target semantic ID (with brackets)
            }
            
            # For train data: keep only longest sequence per user (if enabled)
            if split_name == 'train' and keep_longest_only:
                sequence_length = len(history_item_ids)
                if user_id not in user_to_longest or sequence_length > len(user_to_longest[user_id]['history_item_id']):
                    user_to_longest[user_id] = row
            else:
                rows.append(row)
        
        # For train data, add only the longest sequences (if enabled)
        if split_name == 'train' and keep_longest_only:
            rows = list(user_to_longest.values())
        
        # Apply sample limits for valid/test sets
        original_count = len(rows)
        if split_name == 'valid' and max_valid_samples is not None and len(rows) > max_valid_samples:
            rows = random.sample(rows, max_valid_samples)
            print(f"  Sampled {max_valid_samples} from {original_count} validation samples")
        elif split_name == 'test' and max_test_samples is not None and len(rows) > max_test_samples:
            rows = random.sample(rows, max_test_samples)
            print(f"  Sampled {max_test_samples} from {original_count} test samples")
        
        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f'{category}_5_2016-10-2018-11.csv')
            df.to_csv(output_file, index=False)
            print(f"Created {split_name} file: {output_file} with {len(rows)} rows")
            if split_name == 'train' and keep_longest_only:
                print(f"  Kept longest sequences for {len(rows)} unique users")
            elif split_name == 'train':
                print(f"  Kept all sequences for train data")
            if rows:
                print(f"  Sample row:")
                print(f"    user_id: {rows[0]['user_id']}")
                print(f"    history_item_id: {rows[0]['history_item_id']}")
                print(f"    item_id: {rows[0]['item_id']}")
                print(f"    history_item_sid: {rows[0]['history_item_sid']}")
                print(f"    item_sid: {rows[0]['item_sid']}")
                print(f"    item_title: {rows[0]['item_title'][:50]}...")

def main():
    """读取转换参数，关联原子交互、商品元数据和 SID 索引，写出 CSV 与 info。

    Args:
        无显式参数。

    Returns:
        None: 执行对应命令行流程并写出结果。
    """
    parser = argparse.ArgumentParser(description='Convert dataset (Office_Products/Industrial_and_Scientific) to MiniOneRec format with semantic IDs')
    parser.add_argument('--data_dir', type=str, 
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='Industrial_and_Scientific',
                       help='Dataset name (Office_Products, Industrial_and_Scientific)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for MiniOneRec format data')
    parser.add_argument('--category', type=str, default=None,
                       help='Category name for output files (if None, will use dataset_name)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                       help='Maximum number of samples to keep in validation set (None for all)')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='Maximum number of samples to keep in test set (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--keep_longest_only', action='store_true', default=False,
                       help='Keep only longest sequence per user in train data (default: False)')
    
    args = parser.parse_args()
    
    # Default category to dataset name if not specified
    if args.category is None:
        args.category = args.dataset_name
    
    print(f"Loading {args.dataset_name} data from {args.data_dir}")
    dataset_data = load_dataset(args.data_dir, args.dataset_name)
    
    print(f"Found {len(dataset_data['items'])} items")
    print(f"Found {len(dataset_data['item_to_semantic'])} item-to-semantic mappings")
    
    # Sample a few semantic ID conversions
    sample_items = list(dataset_data['item_to_semantic'].items())[:3]
    for item_id, tokens in sample_items:
        semantic_id = semantic_tokens_to_id(tokens)
        print(f"  Item {item_id}: {tokens} -> {semantic_id}")
    
    # Create output directories
    for subdir in ['train', 'valid', 'test', 'info']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # Create item info file
    info_file = os.path.join(args.output_dir, 'info', f'{args.category}_5_2016-10-2018-11.txt')
    create_item_info_file(dataset_data['items'], dataset_data['item_to_semantic'], info_file)
    print(f"Created item info file: {info_file}")
    
    # Convert each split
    for split_name in ['train', 'valid', 'test']:
        if split_name in dataset_data['splits']:
            split_output_dir = os.path.join(args.output_dir, split_name)
            convert_interactions_to_csv(
                {split_name: dataset_data['splits'][split_name]}, 
                dataset_data['items'],
                dataset_data['item_to_semantic'],
                split_output_dir,
                args.category,
                max_valid_samples=args.max_valid_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
                keep_longest_only=args.keep_longest_only
            )
    
    print(f"\nConversion completed! Data saved to {args.output_dir}")
    print(f"You can now use these files with MiniOneRec training scripts by setting category='{args.category}'")
    
    # Show sampling information if limits were applied
    if args.max_valid_samples is not None:
        print(f"Validation set was limited to {args.max_valid_samples} samples")
    if args.max_test_samples is not None:
        print(f"Test set was limited to {args.max_test_samples} samples")

if __name__ == '__main__':
    main()
