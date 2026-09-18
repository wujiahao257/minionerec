# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path):
    """读取预测 SID，统计 Top-K 命中率、NDCG 及扫描到的无效 SID 数。

    Args:
        path (str | list[str]): 一个或多个预测 JSON 路径；当前多文件循环共享累计状态，默认 Shell 使用单文件。
        item_path (str): 商品目录 TXT 路径，首列为 SID，后续列为标题与商品 ID；用于约束或合法性检查。

    Returns:
        None: 打印离线评测指标。
    """
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0
        
    
    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    # item_names = [ _[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_names= [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")
            minID = 1000000
            for i in range(len(sample)):
                
                if sample[i] not in item_dict:
                    CC += 1
                    print(sample[i])
                    print(target_item)
                if sample[i] == target_item:
                    minID = i
                    break
            
            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        print(valid_topk)
        print(f"NDCG:\t{ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR\t{ALLHR / len(text)}")
        print(CC)

if __name__=='__main__':
    fire.Fire(gao)
