
import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data import  EvalD3Dataset, EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
from accelerate import Accelerator
import random
import bitsandbytes as bnb



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    """将 token ID 转成字符串并用连字符连接，作为前缀查表键。

    Args:
        x (list[int] | torch.Tensor): 需要转换为查表键的 token ID 序列。

    Returns:
        str: 确定性的前缀键，不是密码学哈希。
    """
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    """固定 Python、NumPy 或 PyTorch 随机状态，减少重复实验中的随机差异。

    Args:
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。

    Returns:
        None: 修改当前进程的随机状态和相关后端设置。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
):
    """加载模型和 SID 目录，构造评测输入，受约束生成候选并保存预测 JSON。

    Args:
        base_model (str): 模型 checkpoint 目录或模型标识；需要配套的模型配置、权重和 tokenizer。
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        info_file (str): 商品目录 TXT 路径，首列为 SID，后续列为标题与商品 ID；用于约束或合法性检查。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        test_data_path (str): 用于离线生成的测试 CSV 路径。
        result_json_data (str): 输出预测 JSON 的文件路径，每条样本追加 predict 候选列表。
        batch_size (int): 一次处理的样本数量；SFT 入口中表示用于推算累积步数的全局目标 batch。
        K (int): 预留的示例数量参数；当前 SID/标题样本构造不实际使用它。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        length_penalty (float): 生成器的序列长度惩罚系数；0 表示不额外按长度调整 beam 分数。
        max_new_tokens (int): 每条生成序列新增 token 的上限，不包含 prompt 长度。
        num_beams (int): beam 搜索宽度；相应生成配置通常返回同样数量的候选序列。

    Returns:
        None: 执行对应命令行流程并写出结果。
    """
    random.seed(seed)
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    category = category_dict[category]
    print(category)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        # Format for tokenization
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create prefixID for semantic IDs (existing functionality)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    # Build hash_dict for semantic IDs (existing functionality)
    hash_dict = dict()
    # print(f"eos token: {tokenizer.eos_token_id}")
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Build hash_dict_title for item titles (new functionality)
    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Convert sets to lists for both dictionaries
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    # Define prefix constraint functions
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        """根据当前前缀查找允许生成的下一个 token。

        Args:
            batch_id (int): 生成器当前请求在 batch 中的编号；目录约束对所有请求共享同一查表。
            input_ids (list[int]): 当前回答前缀的 token ID，用于查询共享合法后继表。

        Returns:
            list[int]: 合法后继 token ID；找不到前缀时返回空列表。
        """
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
        
    def prefix_allowed_tokens_fn_title(batch_id, input_ids):
        """根据当前前缀查找允许生成的下一个 token。

        Args:
            batch_id (int): 生成器当前请求在 batch 中的编号；目录约束对所有请求共享同一查表。
            input_ids (list[int]): 当前回答前缀的 token ID，用于查询共享合法后继表。

        Returns:
            list[int]: 合法后继 token ID；找不到前缀时返回空列表。
        """
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title:
            return hash_dict_title[hash_number]
        return []

    # Default to semantic constraints (backward compatibility)
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    # encodings = [val_dataset[i] for i in indexes]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=64,
            length_penalty=1.0,
            **kwargs,
    ):
        """左补齐当前批次的 prompt，执行受 SID 约束的确定性 beam 生成并解码。

        Args:
            encodings (list[dict[str, list[int]]]): 当前 batch 的 prompt 编码列表；函数会手动左 padding。
            num_beams (int): beam 搜索宽度；相应生成配置通常返回同样数量的候选序列。
            max_new_tokens (int): 每条生成序列新增 token 的上限，不包含 prompt 长度。
            length_penalty (float): 生成器的序列长度惩罚系数；0 表示不额外按长度调整 beam 分数。
            kwargs (dict[str, object]): 额外传给 GenerationConfig 的生成参数，避免与显式参数重复。

        Returns:
            list[list[str]]: 每条输入对应 num_beams 个候选 SID 字符串。
        """
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L) 
        
        # print(f"num_beams: {num_beams}")
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            do_sample=False,
            top_k=None,
            top_p=None,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                # Passed as a generate() kwarg on purpose. Since transformers
                # 4.50, values in `generation_config` that equal the *global*
                # default are overwritten by the model's own generation_config
                # -- so `do_sample=False` set on the GenerationConfig above is
                # silently replaced by `do_sample=true` for checkpoints that
                # ship it (e.g. Qwen2.5-*-Instruct), turning constrained beam
                # search into sampling and producing invalid SIDs. Only kwargs
                # take priority over the model defaults.
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        
        if base_model.lower().find("llama") > -1:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs
    
    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])

    
    for idx, encodings in enumerate(tqdm(new_encodings)):
        # Use standard evaluation
        output = evaluate(encodings, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty)
        
        outputs = outputs + output
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
  

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)





