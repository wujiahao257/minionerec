import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F
import re

class Tokenizer:
    """包装已有 tokenizer，统一控制文本编码时的 BOS/EOS 与解码。

    Args:
        tokenizer (transformers.PreTrainedTokenizerBase): 实际分词器实例，必须提供 encode/decode 方法和 bos_token_id/eos_token_id 属性。
    """
    def __init__(self, tokenizer):
        """初始化 Tokenizer：包装已有 tokenizer，统一控制文本编码时的 BOS/EOS 与解码。

        Args:
            self (Tokenizer): 当前实例，由 Python 在调用实例方法时自动传入。
            tokenizer (transformers.PreTrainedTokenizerBase): 实际分词器实例，必须提供 encode/decode 方法和 bos_token_id/eos_token_id 属性。

        Returns:
            None: 完成实例初始化。
        """
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """调用外部分词器编码文本，移除已有边界 token 后按开关添加 BOS/EOS。

        Args:
            self (Tokenizer): 当前实例，由 Python 在调用实例方法时自动传入。
            s (str): 待编码或包装为请求模板的文本。
            bos (bool): 是否在相应序列边界追加分词器定义的起始或结束 token。
            eos (bool): 是否在相应序列边界追加分词器定义的起始或结束 token。

        Returns:
            list[int]: 编码后的 token ID。
        """
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """将 token ID 序列交给外部分词器还原文本。

        Args:
            self (Tokenizer): 当前实例，由 Python 在调用实例方法时自动传入。
            t (list[int]): 待解码的 tokenizer token ID 序列。

        Returns:
            str: 解码文本。
        """
        return self.tokenizer.decode(t)

class SidSFTDataset(Dataset):
    """将历史 SID 与下一 SID 包装为因果语言模型监督微调样本。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        K (int): 预留的示例数量参数；当前 SID/标题样本构造不实际使用它。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        """初始化 SidSFTDataset：将历史 SID 与下一 SID 包装为因果语言模型监督微调样本。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            K (int): 预留的示例数量参数；当前 SID/标题样本构造不实际使用它。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.get_inputs()  
    
    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.data)

    def generate_prompt(self, data_point):
        """按当前任务构造 User Input 与 Response 模板，问答子类留下空回答前缀。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 模型输入模板；普通基类按 data_point.output 决定是否含答案。
        """
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = ", ".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        # print("**********************")
        # print("history: ", history)
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
        prompt = self.generate_prompt(history)
        # print("prompt: ", prompt)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        # print("tokens: ", tokens)
        # print("**********************")
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    
    def get_inputs(self):
        """逐个调用 pre 预处理样本，并将结果缓存到 self.inputs。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 更新实例的样本缓存。
        """
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            
        self.inputs = inputs
    
    def get_all(self):
        """逐行提取输入描述与真实目标，供离线预测回填和指标计算。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: 未 token 化的历史和目标记录。
        """
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        """返回预处理后的样本列表，部分子类在没有缓存时即时调用 pre。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: token 样本或 prompt/completion 记录。
        """
        return self.inputs

    def __getitem__(self, idx):
        """按样本位置返回预处理缓存，部分子类无缓存时调用 pre。

        Args:
            self (SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: 当前任务的编码或文本样本；过滤分支可能缓存 None。
        """
        return self.inputs[idx]



class SidTokenFeatDataset(Dataset):
    """构造单个 SID token 与其语义描述之间的双向 SFT 样本。

    Args:
        description_file (str): SID token 语义 JSON 路径；记录包含 token、description，初始化还使用 keywords。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
    """
    def __init__(self, description_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        """初始化 SidTokenFeatDataset：构造单个 SID token 与其语义描述之间的双向 SFT 样本。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            description_file (str): SID token 语义 JSON 路径；记录包含 token、description，初始化还使用 keywords。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。

        Returns:
            None: 完成实例初始化。
        """
        random.seed(seed)
        
        # Load item features and indices
        with open(description_file, 'r') as f:
            self.token_feat = json.load(f)
        
        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None
        self.test = test
        self.max_len = max_len
        self.category = category
        
        # Build sid2title and title2sid mappings
        self.token2description = {}
        self.description2token = {}

        for example in self.token_feat:
            prefix = example['token']
            description = example['description']
            self.token2description[prefix] = description
            self.description2token[description] = prefix
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        for prefix, description in self.token2description.items():
            self.data.append({
                'task': 'token2description',
                'input': prefix,
                'output': description
            })
        
        # Create title2sid samples  
        for description, prefix in self.description2token.items():
            self.data.append({
                'task': 'description2token',
                'input': description,
                'output': prefix
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        if self.tokenizer is not None:
            self.get_inputs()
    
    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.data)
    
    def generate_prompt(self, data_point):
        """按当前任务构造 User Input 与 Response 模板，问答子类留下空回答前缀。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 模型输入模板；普通基类按 data_point.output 决定是否含答案。
        """
        if data_point['task'] == 'token2description':
            prompt = f"What is the typical scope and shared features of items that contain the token: {data_point['input']}?"
            response = data_point['output']
        else:  # description2token
            prompt = f'What token do the items that have the following scope and shared characteristics contain: "{data_point["input"]}"?'
            response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
        

    def pre(self, idx):
        """构造语义 ID 与文本之间的对齐样本；有 tokenizer 时编码并屏蔽 prompt 的监督。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object]: 无 tokenizer 时返回原始任务字典；否则返回 input_ids/attention_mask，训练模式还含 labels。
        """
        if self.tokenizer is None:
            return self.data[idx]
        
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item Semantic ID token identification.
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        data_point = self.data[idx]
        
        prompt = self.generate_prompt(data_point)
        # print("sidfeature prompt: ", prompt)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        target = data_point['output'] + '\n'
        
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            # "prompt": prompt,
        }
    
    def get_inputs(self):
        """逐个调用 pre 预处理样本，并将结果缓存到 self.inputs。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 更新实例的样本缓存。
        """
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
        self.inputs = inputs
    
    def get_inputs_list(self):
        """返回预处理后的样本列表，部分子类在没有缓存时即时调用 pre。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: token 样本或 prompt/completion 记录。
        """
        return self.inputs if hasattr(self, 'inputs') else [self.pre(i) for i in range(len(self))]
    
    def __getitem__(self, idx):
        """按样本位置返回预处理缓存，部分子类无缓存时调用 pre。

        Args:
            self (SidTokenFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: 当前任务的编码或文本样本；过滤分支可能缓存 None。
        """
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)



class SidItemFeatDataset(Dataset):
    """构造商品 SID 与标题之间的双向问答，仅拼接索引的前三层代码。

    Args:
        item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
    """
    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        """初始化 SidItemFeatDataset：构造商品 SID 与标题之间的双向问答，仅拼接索引的前三层代码。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
            index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。

        Returns:
            None: 完成实例初始化。
        """
        random.seed(seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None
        self.test = test
        self.max_len = max_len
        self.category = category
        
        # Build sid2title and title2sid mappings
        self.sid2title = {}
        self.title2sid = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })
        
        # Create title2sid samples  
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        if self.tokenizer is not None:
            self.get_inputs()
    
    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.data)
    
    def generate_prompt(self, data_point):
        """按当前任务构造 User Input 与 Response 模板，问答子类留下空回答前缀。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 模型输入模板；普通基类按 data_point.output 决定是否含答案。
        """
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
            response = data_point['output']
        else:  # sid2title
            prompt = f'What is the title of item "{data_point["input"]}"?'
            response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        """构造语义 ID 与文本之间的对齐样本；有 tokenizer 时编码并屏蔽 prompt 的监督。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object]: 无 tokenizer 时返回原始任务字典；否则返回 input_ids/attention_mask，训练模式还含 labels。
        """
        if self.tokenizer is None:
            return self.data[idx]
        
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        data_point = self.data[idx]
        
        prompt = self.generate_prompt(data_point)
        # print("sidfeature prompt: ", prompt)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        target = data_point['output'] + '\n'
        
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    
    def get_inputs(self):
        """逐个调用 pre 预处理样本，并将结果缓存到 self.inputs。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 更新实例的样本缓存。
        """
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
        self.inputs = inputs
    
    def get_inputs_list(self):
        """返回预处理后的样本列表，部分子类在没有缓存时即时调用 pre。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: token 样本或 prompt/completion 记录。
        """
        return self.inputs if hasattr(self, 'inputs') else [self.pre(i) for i in range(len(self))]
    
    def __getitem__(self, idx):
        """按样本位置返回预处理缓存，部分子类无缓存时调用 pre。

        Args:
            self (SidItemFeatDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: 当前任务的编码或文本样本；过滤分支可能缓存 None。
        """
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)
    


class FusionSeqRecDataset(Dataset):
    """构造历史 SID 到下一商品标题的 SFT 样本；当前 pre 未启用描述任务分支。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        """初始化 FusionSeqRecDataset：构造历史 SID 到下一商品标题的 SFT 样本；当前 pre 未启用描述任务分支。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
            index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                # Process description according to requirements:
                # 1. If description is empty, use title
                # 2. If description is a list, select the longest one
                # 3. If the longest in list is also empty, use title
                processed_description = self._process_description(description, title)
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        # print("self.sid2title: ", self.sid2title)
        # print("self.sid2description: ", self.sid2description)
        self.get_inputs()
    
    def _process_description(self, description, title):
        """选取最长非空描述，缺失或全空时回退到商品标题。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            description (str | list[str] | None): 商品描述；支持普通字符串、列表及列表的字符串形式，空内容回退到标题。
            title (str): 商品标题，作为缺失或空描述的回退文本。

        Returns:
            str: 可用于生成任务的描述文本。
        """
        # Check if description is empty or None
        if not description or description == '':
            return title
        
        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title
        
        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title
    
    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.data)
    
    def generate_prompt_title(self, history):
        """根据按时间排列的历史 SID 构造询问下一商品标题的问题。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            history (str): 已按交互时间排列并拼接好的历史文本，用于组成推荐问题。

        Returns:
            str: 以标题为预期答案的推荐问题。
        """
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item"
    
    def generate_prompt_description(self, history):
        """根据历史 SID 构造询问下一类商品需求的问题；当前主 pre 没有选择该分支。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            history (str): 已按交互时间排列并拼接好的历史文本，用于组成推荐问题。

        Returns:
            str: 以商品描述为预期答案的问题文本。
        """
        return f"Please review the user's historical interactions: {history}, and describe what kind of item he still needs."
    
    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 历史 SID 和目标标题/描述等字段，缺失映射时使用回退文本。
        """
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
            
        if target_sid in self.sid2description:
            target_description = self.sid2description[target_sid]
            # Clean description if it's a string representation of a list
            if isinstance(target_description, str) and target_description.startswith("['") and target_description.endswith("']"):
                try:
                    desc_list = eval(target_description)
                    target_description = desc_list[0] if desc_list else target_description
                except:
                    pass  # Keep original if eval fails
        else:
            target_description = f"An item with semantic ID {target_sid}"
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_description": target_description,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def generate_formatted_prompt(self, prompt, response):
        """把问题放进 User Input 模板，并留下空的 Response 前缀。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            prompt (str): 待编码或包装为请求模板的文本。
            response (str): 预留的答案文本参数；当前格式化函数只输出空回答前缀，不使用它。

        Returns:
            str: 尚未包含正确答案的 prompt。
        """
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""  
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        """if random.random() < 0.5:
            # Title task
            prompt = self.generate_prompt_title(history_data['history_str'])
            target = history_data['target_title'] + '\n'
        else:
            # Description task
            prompt = self.generate_prompt_description(history_data['history_str'])
            target = history_data['target_description'] + '\n'
        """
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_title'] + '\n'
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        tokens = tokens + self.tokenizer.encode(formatted_prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    
    def get_inputs(self):
        """逐个调用 pre 预处理样本，并将结果缓存到 self.inputs。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 更新实例的样本缓存。
        """
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_inputs_list(self):
        """返回预处理后的样本列表，部分子类在没有缓存时即时调用 pre。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: token 样本或 prompt/completion 记录。
        """
        return self.inputs if hasattr(self, 'inputs') else []
    
    def __getitem__(self, idx):
        """按样本位置返回预处理缓存，部分子类无缓存时调用 pre。

        Args:
            self (FusionSeqRecDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: 当前任务的编码或文本样本；过滤分支可能缓存 None。
        """
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)