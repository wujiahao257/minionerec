"""将 MiniOneRec 的 SID 候选转成 SIM 商品候选；可恢复预测时点之前的长历史。"""

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .data import file_digest


def clean_sid(value):
    """Args: value (str): SID 字符串。Returns: str: 清理外层引号和空白后的 SID。"""
    return str(value).strip().strip('"').strip()


def restore_history(sequence, short_history, target, history_end=None):
    """定位样本在完整行为序列中的位置，只取目标出现前的前缀。

    Args:
        sequence (list[int]): 用户完整时间有序序列，可能包含未来行为。
        short_history (list[int]): CSV 中该样本的历史尾部。
        target (int | None): 该样本的目标，只用来核对离线位置，不加入特征。
        history_end (int | None): 可显式指定预测位置；切片右端，不包含该位置。
    Returns:
        list[int]: sequence[:history_end]；匹配不唯一时拒绝猜测。
    """
    def matches(pos):
        return (len(short_history) <= pos <= len(sequence)
                and (not short_history or sequence[pos-len(short_history):pos] == short_history)
                and (target is None or (pos < len(sequence) and sequence[pos] == target)))
    if history_end is not None:
        if type(history_end) is not int or not matches(history_end):
            raise ValueError('history_end does not match this sample')
        return sequence[:history_end]
    if target is None or not short_history:
        raise ValueError('Long-history alignment requires history_end or history + target')
    positions = [pos for pos in range(len(short_history), len(sequence)) if matches(pos)]
    if len(positions) != 1:
        raise ValueError('Long-history alignment is missing/ambiguous; supply verified history_end')
    return sequence[:positions[0]]


def prepare_records(predictions, index, index_digest, sequences=None, csv_rows=None):
    """展开 SID 碰撞、去重商品，保留召回位置；不注入真实目标。

    Args:
        predictions (list[dict]): evaluate 输出；每条含 predict 和原始样本标识。
        index (dict[str,list[str]]): 商品编号到 SID token 的映射。
        index_digest (str): 原索引文件 SHA256。
        sequences (dict[str,list[int]] | None): 同一编号体系下的 .inter.json。
        csv_rows (list[dict] | None): 兼容旧输出；必须与预测逐行对应且输入/目标可核验。
    Returns:
        tuple[list[dict],dict]: 候选记录和无效 SID、碰撞等统计。
    """
    if csv_rows is not None and len(csv_rows) != len(predictions):
        raise ValueError('CSV and predictions have different lengths')
    reverse = defaultdict(list)
    for item, tokens in index.items():
        item_id = int(item)
        if item_id < 0:
            raise ValueError('Negative item ID in SID index')
        reverse[''.join(tokens)].append(item_id)
    for ids in reverse.values():
        ids.sort()
    stats = {'queries': len(predictions), 'invalid_sids': 0, 'collision_sids': 0,
             'empty_candidates': 0, 'target_recalled': 0, 'max_history_length': 0}
    records = []
    for number, prediction in enumerate(predictions):
        source = prediction if csv_rows is None else csv_rows[number]
        if 'history_item_id' not in source or 'user_id' not in source:
            raise ValueError('Missing sample metadata; regenerate predictions or supply --csv')
        history = source['history_item_id']
        if isinstance(history, str):
            history = ast.literal_eval(history)
        history = [int(i) for i in history]
        target = int(source['item_id']) if source.get('item_id') is not None else None
        user = str(source['user_id'])
        if csv_rows is not None:
            expected = ', '.join(ast.literal_eval(source['history_item_sid']))
            expected_input = ('Can you predict the next possible item the user may expect, '
                              'given the following chronological interaction history: ' + expected)
            if prediction.get('input') != expected_input or clean_sid(prediction['output']) != clean_sid(source['item_sid']):
                raise ValueError('CSV/prediction order or identity mismatch')
        if target is not None:
            target_sid = ''.join(index.get(str(target), []))
            if not target_sid or clean_sid(prediction['output']) != target_sid:
                raise ValueError('Target and SID mapping mismatch')
        short_history = history[:]
        if sequences is not None:
            # convert_dataset 为原数字用户编号加 A 前缀，优先使用原样键。
            key = user if user in sequences else user.removeprefix('A')
            if key not in sequences:
                raise ValueError('User missing from full history: ' + user)
            end = source.get('history_end')
            history = restore_history([int(i) for i in sequences[key]], history, target,
                                      None if end in (None, '') else int(end))
        # ID 只基于历史上下文，不把目标内容编码进可训练特征。
        identity = json.dumps([user, history], separators=(',', ':'))
        query_id = hashlib.sha256(identity.encode()).hexdigest()
        scores = prediction.get('predict_scores', [None] * len(prediction['predict']))
        if len(scores) != len(prediction['predict']):
            raise ValueError('predict_scores length mismatch')
        candidates, ranks, candidate_scores, seen = [], [], [], set()
        for rank, (sid, score) in enumerate(zip(prediction['predict'], scores), 1):
            if score is not None and not math.isfinite(score):
                raise ValueError('Non-finite beam score')
            ids = reverse.get(clean_sid(sid), [])
            if not ids:
                stats['invalid_sids'] += 1
            if len(ids) > 1:
                stats['collision_sids'] += 1
            for item in ids:
                if item not in seen:
                    candidates.append(item)
                    candidate_scores.append(score)
                    ranks.append(rank)
                    seen.add(item)
        stats['empty_candidates'] += int(not candidates)
        stats['target_recalled'] += int(target is not None and target in seen)
        stats['max_history_length'] = max(stats['max_history_length'], len(history))
        row = dict(query_id=query_id, user_id=user, history_ids=history,
                   short_history_ids=short_history, candidate_ids=candidates,
                   recall_scores=candidate_scores, recall_ranks=ranks,
                   index_digest=index_digest,
                   history_source='aligned_full_sequence' if sequences is not None else 'provided_history')
        if target is not None:
            row['target_id'] = target
        records.append(row)
    if len({r['query_id'] for r in records}) != len(records):
        raise ValueError('Repeated user/history queries; deduplicate explicitly before preparation')
    return records, stats


def main():
    """解析路径参数，保存 JSONL 候选及统计；不执行大模型推理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequences', help='与 CSV 商品/用户编号完全一致的 .inter.json')
    parser.add_argument('--csv', help='旧预测文件缺少商品编号时，提供对应 CSV')
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    csv_rows = None
    if args.csv:
        with open(args.csv, encoding='utf-8-sig', newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
    rows, stats = prepare_records(read(args.predictions), read(args.index), file_digest(args.index),
                                 read(args.sequences) if args.sequences else None, csv_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
