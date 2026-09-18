import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import datetime
import torch
from tqdm import tqdm
import numpy as np


def clean_text(text):
    """将元数据转为文本，移除 HTML、解码实体并压缩多余空白。

    Args:
        text (str | list | None): 原始元数据文本；清洗函数先转成字符串并去除 HTML/多余空白。

    Returns:
        str: 清洗后的文本，空输入返回空字符串。
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities
    text = html.unescape(text)
    # Replace quotes
    text = text.replace("&quot;", "\"").replace("&amp;", "&")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def check_path(path):
    """在目录不存在时创建目录，供后续数据或 checkpoint 写入。

    Args:
        path (str): 当前读写操作使用的文件或目录路径。

    Returns:
        None: 在文件系统中创建目录。
    """
    os.makedirs(path, exist_ok=True)


def write_json_file(data, file_path):
    """把数据序列化为 JSON 文件。

    Args:
        data (dict | list): 可 JSON 序列化的评论、商品特征或交互数据。
        file_path (str): 当前读写操作使用的文件或目录路径。

    Returns:
        None: 写入指定文件。
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def write_remap_index(index_map, file_path):
    """逐行写出原始用户或商品标识与连续整数编号的对应关系。

    Args:
        index_map (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。
        file_path (str): 当前读写操作使用的文件或目录路径。

    Returns:
        None: 写入制表符分隔的映射文件。
    """
    with open(file_path, 'w') as f:
        for original, mapped in index_map.items():
            f.write(f"{original}\t{mapped}\n")


# Dataset name mapping
amazon18_dataset2fullname = {
    'Arts': 'Arts_Crafts_and_Sewing',
    'Games': 'Video_Games',
    'Sports': 'Sports_and_Outdoors',
    'Instruments': 'Musical_Instruments',
    'Scientific': 'Scientific'
}


def get_timestamp_start(year, month):
    """把指定月份第一天的本地时间转换为秒级 Unix 时间戳。

    Args:
        year (int): 日历年份或月份；起止月份按该月第一天零点转换成时间边界。
        month (int): 日历年份或月份；起止月份按该月第一天零点转换成时间边界。

    Returns:
        int: 当月第一天零点对应的时间戳。
    """
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())


def load_metadata_json2csv_style(category, metadata_file=None):
    """读取 Amazon18 元数据，过滤异常或过长标题并建立 ASIN 到标题映射。

    Args:
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        metadata_file (str | None): 原始商品元数据或评论 JSON/JSONL 路径；Amazon18 可用 None 选择约定文件名。

    Returns:
        tuple[list[dict], dict[str, str], set[str]]: 原始元数据、合法标题映射、待排除商品集合。
    """
    if metadata_file is None:
        metadata_file = f'../meta_{category}.json'
    
    metadata = []
    try:
        with open(metadata_file) as f:
            metadata = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Metadata file {metadata_file} not found")
        return [], {}, set()
    
    id_title = {}
    remove_items = set()
    
    for meta in tqdm(metadata, desc="Processing metadata"):
        if ('title' not in meta) or (meta['title'].find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue
        
        # Clean title like json2csv
        meta['title'] = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
        
        if len(meta['title']) > 1 and len(meta['title'].split(" ")) <= 20:
            id_title[meta['asin']] = meta['title']
            # Store full metadata for later use
        else:
            remove_items.add(meta['asin'])
    
    return metadata, id_title, remove_items


def load_reviews_json2csv_style(category, reviews_file=None, start_timestamp=None, end_timestamp=None):
    """读取 Amazon18 评论 JSON 行；时间筛选留给后续 K-core 阶段。

    Args:
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        reviews_file (str | None): 原始商品元数据或评论 JSON/JSONL 路径；Amazon18 可用 None 选择约定文件名。
        start_timestamp (int | None): 以秒为单位的时间筛选边界；筛选函数在两端都提供时应用区间过滤。
        end_timestamp (int | None): 以秒为单位的时间筛选边界；筛选函数在两端都提供时应用区间过滤。

    Returns:
        list[dict]: 评论记录，找不到输入文件时返回空列表。
    """
    if reviews_file is None:
        try:
            with open(f'../{category}_5.json') as f:
                reviews = [json.loads(line) for line in f]
        except FileNotFoundError:
            try:
                with open(f'../{category}.json') as f:
                    reviews = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Reviews file not found for category {category}")
                return []
    else:
        try:
            with open(reviews_file) as f:
                reviews = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"Reviews file {reviews_file} not found")
            return []
    
    # NOTE: Don't filter by timestamp here, do it in k-core filtering like json2csv
    return reviews


def k_core_filtering_json2csv_style(reviews, id_title, K=5, start_timestamp=None, end_timestamp=None):
    """按时间和有效标题过滤交互，反复移除不足 K 次交互的用户和商品。

    Args:
        reviews (list[dict[str, object]] | None): 原始评论或商品元数据记录列表；具体字段遵循对应 Amazon 数据版本。
        id_title (dict[str, str]): 原始商品 ASIN/parent_asin 到清洗后标题的映射。
        K (int): 每层聚类中心数量；数据预处理函数中表示用户和商品的最小交互次数。
        start_timestamp (int | None): 以秒为单位的时间筛选边界；筛选函数在两端都提供时应用区间过滤。
        end_timestamp (int | None): 以秒为单位的时间筛选边界；筛选函数在两端都提供时应用区间过滤。

    Returns:
        tuple[list[dict], dict[str, int], dict[str, int]]: 保留评论、用户交互数、商品交互数。
    """
    remove_users = set()
    remove_items = set()
    
    # Remove items without titles (like json2csv)
    for review in reviews:
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])
    
    # Iterative k-core filtering (exactly like json2csv)
    while True:
        new_reviews = []
        flag = False
        total = 0
        user_counts = dict()
        item_counts = dict()
        
        for review in tqdm(reviews, desc="K-core filtering"):
            # Filter by timestamp INSIDE the loop like json2csv
            if start_timestamp and end_timestamp:
                if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                    continue
            
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue
            
            if review['reviewerID'] not in user_counts:
                user_counts[review['reviewerID']] = 0
            user_counts[review['reviewerID']] += 1
            
            if review['asin'] not in item_counts:
                item_counts[review['asin']] = 0
            item_counts[review['asin']] += 1
            
            total += 1
            new_reviews.append(review)
        
        # Mark users/items for removal if below threshold
        for user in user_counts:
            if user_counts[user] < K:
                remove_users.add(user)
                flag = True
        
        for item in item_counts:
            if item_counts[item] < K:
                remove_items.add(item)
                flag = True
        
        print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {total}, Density: {total / (len(user_counts) * len(item_counts)) if len(user_counts) > 0 and len(item_counts) > 0 else 0}")
        
        if not flag:
            break
        
        reviews = new_reviews
    
    return new_reviews, user_counts, item_counts


def convert_inters2dict_amazon18_style(reviews):
    """按用户时间排序并给用户和商品连续编号，保留重复交互。

    Args:
        reviews (list[dict[str, object]] | None): 原始评论或商品元数据记录列表；具体字段遵循对应 Amazon 数据版本。

    Returns:
        tuple[dict, dict[str, int], dict[str, int], list[tuple]]: 用户编号到商品序列、用户映射、商品映射、交互元组。
    """
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    
    # Sort reviews by timestamp for each user
    user_reviews = collections.defaultdict(list)
    for review in reviews:
        user_reviews[review['reviewerID']].append(review)
    
    # Sort each user's reviews by timestamp (don't remove duplicates like json2csv)
    for user in user_reviews:
        user_reviews[user].sort(key=lambda x: int(x['unixReviewTime']))
    
    # Create mappings and interactions
    interactions = []
    for user in user_reviews:
        if user not in user2index:
            user2index[user] = len(user2index)
        
        user_items = []
        for review in user_reviews[user]:
            item = review['asin']
            if item not in item2index:
                item2index[item] = len(item2index)
            
            user_items.append(item)
            interactions.append((
                user, item, 
                float(review['overall']), 
                int(review['unixReviewTime'])
            ))
        
        user2items[user2index[user]] = [item2index[item] for item in user_items]
    
    return user2items, user2index, item2index, interactions


def generate_interaction_list_json2csv_style(reviews, user2index, item2index, id_title):
    """对每个用户构造最多 10 条历史的下一商品样本，再按目标时间全局排序。

    Args:
        reviews (list[dict[str, object]] | None): 原始评论或商品元数据记录列表；具体字段遵循对应 Amazon 数据版本。
        user2index (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。
        item2index (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。
        id_title (dict[str, str]): 原始商品 ASIN/parent_asin 到清洗后标题的映射。

    Returns:
        list[list[object]]: 历史窗口记录；GPR 版本还附加随机上下文 token。
    """
    # Create user interactions similar to json2csv
    interact = dict()
    item2id = {item: idx for item, idx in item2index.items()}
    
    for review in tqdm(reviews, desc="Building interaction list"):
        user = review['reviewerID']
        item = review['asin']
        
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'item_ids': [],
                'titles': []
            }
        
        # Keep all interactions like json2csv (no deduplication)
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
        interact[user]['item_ids'].append(item2id[item])
        interact[user]['titles'].append(id_title[item])
    
    # Sort by timestamp for each user
    interaction_list = []
    for user in tqdm(interact.keys(), desc="Creating interaction sequences"):
        items = interact[user]['items']
        ratings = interact[user]['ratings']
        timestamps = interact[user]['timestamps']
        item_ids = interact[user]['item_ids']
        titles = interact[user]['titles']
        
        # Sort all by timestamp
        all_data = list(zip(items, ratings, timestamps, item_ids, titles))
        all_data.sort(key=lambda x: int(x[2]))
        items, ratings, timestamps, item_ids, titles = zip(*all_data)
        items, ratings, timestamps, item_ids, titles = list(items), list(ratings), list(timestamps), list(item_ids), list(titles)
        
        # Create sequences like json2csv (sliding window with max history of 10)
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([
                user,                    # user_id
                items[st:i],            # item_asins (history)
                items[i],               # item_asin (target)
                item_ids[st:i],         # history_item_id
                item_ids[i],            # item_id (target)
                titles[st:i],           # history_item_title
                titles[i],              # item_title (target)
                ratings[st:i],          # history_rating
                ratings[i],             # rating (target)
                timestamps[st:i],       # history_timestamp
                timestamps[i]           # timestamp (target)
            ])
    
    # Sort by timestamp for chronological split
    interaction_list.sort(key=lambda x: int(x[-1]))
    return interaction_list


def convert_to_atomic_files_json2csv_style(args, interaction_list, user2index):
    """按已排序样本的 80%/10%/10% 切分，写出训练、验证和测试原子交互文件。

    Args:
        args (argparse.Namespace): dataset、output_path、时间范围和数据清洗参数。
        interaction_list (list[list[object]]): 按目标时间排序的历史窗口记录；包含历史、目标、评分和时间，GPR 还追加上下文。
        user2index (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。

    Returns:
        tuple[list, list, list]: train、valid、test 交互记录。
    """
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    
    # Create output directories
    check_path(os.path.join(args.output_path, args.dataset))
    
    # Split 8:1:1 like json2csv
    total_len = len(interaction_list)
    train_end = int(total_len * 0.8)
    valid_end = int(total_len * 0.9)
    
    train_interactions = interaction_list[:train_end]
    valid_interactions = interaction_list[train_end:valid_end]
    test_interactions = interaction_list[valid_end:]
    
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Valid interactions: {len(valid_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    # Write train file
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in train_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original]  
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    # Write valid file  
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in valid_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original]  
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    # Write test file
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for interaction in test_interactions:
            user_id_original = interaction[0]
            user_id = user2index[user_id_original] 
            history_item_ids = [str(x) for x in interaction[3]]  # history item ids  
            target_item_id = str(interaction[4])  # target item id
            
            # Limit history to last 50 items like amazon18
            history_seq = history_item_ids[-50:]
            file.write(f'{user_id}\t{" ".join(history_seq)}\t{target_item_id}\n')
    
    return train_interactions, valid_interactions, test_interactions


def load_review_data_amazon18_style(reviews, user2index, item2index):
    """按用户编号、商品编号和时间构建评论文本字典。

    Args:
        reviews (list[dict[str, object]] | None): 原始评论或商品元数据记录列表；具体字段遵循对应 Amazon 数据版本。
        user2index (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。
        item2index (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。

    Returns:
        dict[str, dict[str, object]]: 以三元组字符串为键的评论与摘要等信息。
    """
    review_data = {}
    
    for review in tqdm(reviews, desc='Load reviews'):
        try:
            user = review['reviewerID']
            item = review['asin']
            
            if user in user2index and item in item2index:
                uid = user2index[user]
                iid = item2index[item]
                
                # Use timestamp to create unique keys for same user-item pairs at different times
                timestamp = review['unixReviewTime']
                unique_key = str((uid, iid, timestamp))
                
            else:
                continue
                
            if 'reviewText' in review:
                review_text = clean_text(review['reviewText'])
            else:
                review_text = ''
                
            if 'summary' in review:
                summary = clean_text(review['summary'])
            else:
                summary = ''
                
            review_data[unique_key] = {"review": review_text, "summary": summary}
            
        except (ValueError, KeyError):
            continue
    
    return review_data


def create_item_features_amazon18_style(metadata, item2index, id_title):
    """把保留商品的元数据映射到连续商品编号；GPR 版本还构造模拟类型和价值。

    Args:
        metadata (list[dict[str, object]] | None): 原始评论或商品元数据记录列表；具体字段遵循对应 Amazon 数据版本。
        item2index (dict[str, int]): 原始用户或商品标识到连续整数编号的映射。
        id_title (dict[str, str]): 原始商品 ASIN/parent_asin 到清洗后标题的映射。

    Returns:
        dict[int, dict[str, object]]: 商品标题、描述、品牌、类别等特征。
    """
    item2feature = collections.defaultdict(dict)
    
    # Create a mapping from asin to metadata
    asin_to_meta = {}
    for meta in metadata:
        asin_to_meta[meta['asin']] = meta
    
    for item_asin, item_id in item2index.items():
        if item_asin in asin_to_meta:
            meta = asin_to_meta[item_asin]
            
            title = id_title.get(item_asin, clean_text(meta.get("title", "")))
            
            descriptions = meta.get("description", "")
            if descriptions:
                descriptions = clean_text(descriptions)
            else:
                descriptions = ""
                
            brand = meta.get("brand", "").replace("by\n", "").strip()
            
            categories = meta.get("categories", [])
            if categories and len(categories) > 0:
                # Handle both list and string formats
                if isinstance(categories[0], list):
                    # Flatten nested categories
                    flat_categories = []
                    for cat_group in categories:
                        flat_categories.extend(cat_group)
                    categories = flat_categories
                
                # Filter out invalid categories
                new_categories = []
                for category in categories:
                    if "</span>" not in str(category):
                        new_categories.append(str(category).strip())
                categories = ",".join(new_categories).strip()
            else:
                categories = ""
            
            item2feature[item_id] = {
                "title": title,
                "description": descriptions,
                "brand": brand,
                "categories": categories
            }
    
    return item2feature


def process_dataset_recursive(args, metadata, reviews, start_timestamp, end_timestamp):
    """加载元数据并过滤交互，商品不足 3000 时逐年向前扩展起始时间。

    Args:
        args (argparse.Namespace): dataset、output_path、时间范围和数据清洗参数。
        metadata (list[dict] | None): 传入的元数据列表；当前函数会重新从文件加载并覆盖此参数。
        reviews (list[dict[str, object]] | None): 原始评论或商品元数据记录列表；具体字段遵循对应 Amazon 数据版本。
        start_timestamp (int | None): 以秒为单位的时间筛选边界；筛选函数在两端都提供时应用区间过滤。
        end_timestamp (int | None): 以秒为单位的时间筛选边界；筛选函数在两端都提供时应用区间过滤。

    Returns:
        tuple[list, dict, dict, list, dict] | None: 保留评论、用户计数、商品计数、元数据、标题映射；无元数据时返回 None。
    """
    
    # Load metadata 
    metadata, id_title, remove_items = load_metadata_json2csv_style(
        args.dataset, args.metadata_file
    )
    
    if not metadata:
        print(f"Error: No metadata found for dataset {args.dataset}")
        return None
    
    print(f"Loaded {len(metadata)} metadata items, {len(id_title)} with valid titles")
    
    # Perform k-core filtering with timestamp filtering inside
    print("Performing k-core filtering...")
    filtered_reviews, user_counts, item_counts = k_core_filtering_json2csv_style(
        reviews, id_title, args.user_k, start_timestamp, end_timestamp
    )
    
    print(f"After filtering: {len(user_counts)} users, {len(item_counts)} items, {len(filtered_reviews)} reviews")
    
    # Check if we need to expand time range (like json2csv)
    if args.st_year > 1996 and len(item_counts) < 3000:
        print(f"Items count {len(item_counts)} < 3000, expanding time range...")
        args.st_year -= 1
        new_start_timestamp = get_timestamp_start(args.st_year, args.st_month)
        print(f"New time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}")
        return process_dataset_recursive(args, metadata, reviews, new_start_timestamp, end_timestamp)
    
    return filtered_reviews, user_counts, item_counts, metadata, id_title


def parse_args():
    """解析当前脚本的命令行参数，返回后续数据或模型构造配置。

    Args:
        无显式参数。

    Returns:
        argparse.Namespace: 当前入口定义的参数集合。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games / Sports')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--st_year', type=int, default=1996, help='start year')
    parser.add_argument('--st_month', type=int, default=10, help='start month')
    parser.add_argument('--ed_year', type=int, default=2018, help='end year')
    parser.add_argument('--ed_month', type=int, default=11, help='end month')
    parser.add_argument('--metadata_file', type=str, default=None, help='metadata file path')
    parser.add_argument('--reviews_file', type=str, default=None, help='reviews file path')
    parser.add_argument('--output_path', type=str, default='./data', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f'Processing dataset: {args.dataset}')
    print(f'Initial time range: {args.st_year}-{args.st_month} to {args.ed_year}-{args.ed_month}')
    print(f'K-core threshold: {args.user_k}')
    
    # Set time range
    start_timestamp = get_timestamp_start(args.st_year, args.st_month)
    end_timestamp = get_timestamp_start(args.ed_year, args.ed_month)
    
    # Load reviews first (without timestamp filtering)
    print("Loading reviews...")
    reviews = load_reviews_json2csv_style(
        args.dataset, args.reviews_file
    )
    
    if not reviews:
        print(f"Error: No reviews found for dataset {args.dataset}")
        print("Please check if the reviews file exists and try again.")
        exit(1)
        
    print(f"Loaded {len(reviews)} total reviews")
    
    # Process dataset with recursive logic
    result = process_dataset_recursive(args, None, reviews, start_timestamp, end_timestamp)
    
    if result is None:
        print("Failed to process dataset")
        exit(1)
    
    filtered_reviews, user_counts, item_counts, metadata, id_title = result
    
    print(f"Final filtering results:")
    print(f"Users: {len(user_counts)}, Items: {len(item_counts)}, Reviews: {len(filtered_reviews)}")
    print(f"Density: {len(filtered_reviews) / (len(user_counts) * len(item_counts)) if len(user_counts) > 0 and len(item_counts) > 0 else 0}")
    
    # Convert to amazon18 style format
    print("Converting to amazon18 format...")
    user2items, user2index, item2index, interactions = convert_inters2dict_amazon18_style(filtered_reviews)
    
    print(f"After amazon18 conversion:")
    print(f"  User2index: {len(user2index)} users")
    print(f"  Item2index: {len(item2index)} items") 
    print(f"  Interactions: {len(interactions)} interactions")
    
    # Generate interaction list for 8:1:1 split like json2csv
    print("Generating interaction list for 8:1:1 split...")
    interaction_list = generate_interaction_list_json2csv_style(
        filtered_reviews, user2index, item2index, id_title
    )
    
    print(f"Generated {len(interaction_list)} interaction sequences")
    
    # Create output directory and split data
    train_interactions, valid_interactions, test_interactions = convert_to_atomic_files_json2csv_style(
        args, interaction_list, user2index
    )
    
    # Generate user2items for compatibility with amazon18 format
    user2items_final = collections.defaultdict(list)
    for user_idx, item_list in user2items.items():
        user2items_final[user_idx] = item_list
    
    # Write interaction files (amazon18 style output)
    write_json_file(user2items_final, os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter.json'))
    
    # Create item features
    print("Creating item features...")
    item2feature = create_item_features_amazon18_style(metadata, item2index, id_title)
    
    # Load review data
    print("Loading review data...")
    review_data = load_review_data_amazon18_style(filtered_reviews, user2index, item2index)
    
    print(f"Final statistics:")
    print(f"Users: {len(user2index)}")
    print(f"Items: {len(item2index)}")
    print(f"Reviews: {len(review_data)}")
    print(f"Total interaction sequences: {len(interaction_list)}")
    print(f"Train sequences: {len(train_interactions)}")
    print(f"Valid sequences: {len(valid_interactions)}")
    print(f"Test sequences: {len(test_interactions)}")
    
    # Write output files (amazon18 style)
    write_json_file(item2feature, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item.json'))
    write_json_file(review_data, os.path.join(args.output_path, args.dataset, f'{args.dataset}.review.json'))
    
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2id'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2id'))
    
    print("Processing completed!")
