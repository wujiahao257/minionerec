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

class BaseDataset(Dataset):
    """推荐数据集基类，定义 prompt 模板、预处理缓存和样本读取接口。

    Args:
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
    """
    def __init__(self, tokenizer=None, max_len=2048, test=False, category="", dedup=False, seed=None):
        """初始化 BaseDataset：推荐数据集基类，定义 prompt 模板、预处理缓存和样本读取接口。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__()
        self.data = None
        self.inputs = None
        
        if tokenizer is not None:
            self.tokenizer = Tokenizer(tokenizer)
        if seed is not None:
            random.seed(seed)
        
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

    def __len__(self):
        """返回底层数据记录数量。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            int: 数据集样本数；部分去重缓存长度可能与原始记录数不同。
        """
        return len(self.data)

    def get_inputs(self):
        """逐个调用 pre 预处理样本，并将结果缓存到 self.inputs。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。

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
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。

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
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: token 样本或 prompt/completion 记录。
        """
        return self.inputs

    def __getitem__(self, idx):
        """按样本位置返回预处理缓存，部分子类无缓存时调用 pre。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: 当前任务的编码或文本样本；过滤分支可能缓存 None。
        """
        return self.inputs[idx]

    def pre(self, idx):
        """声明单样本预处理接口，具体任务必须在子类实现。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            无正常返回值：基类抛出 NotImplementedError。
        """
        raise NotImplementedError(None)

    def get_history(self, row):
        """声明子类的历史提取接口；当前基类实现不能直接调用。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            无正常返回值：基类执行 raise，子类必须覆盖。
        """
        raise {}
       
    def generate_prompt(self, data_point):
        """按当前任务构造 User Input 与 Response 模板，问答子类留下空回答前缀。

        Args:
            self (BaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 模型输入模板；普通基类按 data_point.output 决定是否含答案。
        """
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""


class CSVBaseDataset(BaseDataset):    
    """读取交互 CSV 并提供可复现抽样的推荐数据集基类。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
    """
    def __init__(self, train_file, sample=-1, seed=0, max_len=2048, category="", dedup=False, tokenizer=None, test=False):
        """初始化 CSVBaseDataset：读取交互 CSV 并提供可复现抽样的推荐数据集基类。

        Args:
            self (CSVBaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(tokenizer, max_len, test, category, dedup, seed)

        self.data = pd.read_csv(train_file)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)


class JSONBaseDataset(BaseDataset):
    """加载商品元数据与 SID 索引的问答数据集基类。

    Args:
        item_file (str | None): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        index_file (str | None): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
    """
    def __init__(self, item_file=None, index_file=None, tokenizer=None, max_len=2048, test=False, category="", dedup=False, seed=None):
        """初始化 JSONBaseDataset：加载商品元数据与 SID 索引的问答数据集基类。

        Args:
            self (JSONBaseDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            item_file (str | None): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
            index_file (str | None): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
            tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        
        # Load item features and indices if files are provided
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)


class SFTData(CSVBaseDataset):
    """以历史商品标题为输入、下一商品标题为答案的监督微调数据集。

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
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        """初始化 SFTData：以历史商品标题为输入、下一商品标题为答案的监督微调数据集。

        Args:
            self (SFTData): 当前实例，由 Python 在调用实例方法时自动传入。
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
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  


    def generate_example_prompt(self, data_point):
        """把含示例编号的输入和答案包装为示例提示段。

        Args:
            self (SFTData): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 包含 Example 和 Response 标记的文本。
        """
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (SFTData): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (SFTData): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
""" 
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
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
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }


class D3Dataset(CSVBaseDataset):
    """标题序列推荐的 RL 数据集，输出文本 prompt/completion 和答案查询映射。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, train_file, max_len=2048, sample=-1, seed=0, category="", dedup=False):
        """初始化 D3Dataset：标题序列推荐的 RL 数据集，输出文本 prompt/completion 和答案查询映射。

        Args:
            self (D3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False)

        self.prompt2history = {}
        self.history2target = {}
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (D3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        """构造当前记录的 RL prompt/completion，并更新历史到目标的查询映射。

        Args:
            self (D3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, str] | None: prompt 和 completion；启用跳过重复的子类可能返回 None。
        """
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[instruction + prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": instruction + prompt,
            "completion": target_item,
        }


class EvalD3Dataset(CSVBaseDataset):

    """标题序列推荐的评测数据集，test 模式只返回输入编码。

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
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        """初始化 EvalD3Dataset：标题序列推荐的评测数据集，test 模式只返回输入编码。

        Args:
            self (EvalD3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
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
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  

    def generate_example_prompt(self, data_point):
        """把含示例编号的输入和答案包装为示例提示段。

        Args:
            self (EvalD3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 包含 Example 和 Response 标记的文本。
        """
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""
    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (EvalD3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (EvalD3Dataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                
                # "select_index": select_index,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }


class SidDataset(CSVBaseDataset):
    """以历史 SID 预测下一 SID 的 RL 数据集，维护 prompt 到目标的查询映射。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, train_file, max_len=2048, sample=-1, seed=0, category="", dedup=False):
        """初始化 SidDataset：以历史 SID 预测下一 SID 的 RL 数据集，维护 prompt 到目标的查询映射。

        Args:
            self (SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer=None, test=False)

        self.prompt2history = {}
        self.history2target = {}
        self.get_inputs()  

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = "::".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                # Analyze user preferences and then predict the semantic ID of the next item.
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}
    
    def pre(self, idx):
        """构造当前记录的 RL prompt/completion，并更新历史到目标的查询映射。

        Args:
            self (SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, str] | None: prompt 和 completion；启用跳过重复的子类可能返回 None。
        """
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,

        }


class SidSFTDataset(CSVBaseDataset):
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
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.get_inputs()

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
            print(len(tokens))
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }


class SidSFTDataset_GPR(CSVBaseDataset):
    """构造含用户、场景和商品类型 token 的 GPR 样本，并携带模拟价值权重。

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
        """初始化 SidSFTDataset_GPR：构造含用户、场景和商品类型 token 的 GPR 样本，并携带模拟价值权重。

        Args:
            self (SidSFTDataset_GPR): 当前实例，由 Python 在调用实例方法时自动传入。
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
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        # Try to load features from standard location
        try:
            with open(f'data/{category}/{category}.user.json', 'r') as f:
                self.user_features = json.load(f)
        except FileNotFoundError:
            try:
                dataset_dir = os.path.dirname(train_file)
                # Assuming structure data/Amazon/train/Sports... -> data/Sports/Sports.user.json
                with open(f'data/{category}/{category}.user.json', 'r') as f:
                    self.user_features = json.load(f)
            except:
                self.user_features = {}
            
        try:
            with open(f'data/{category}/{category}.item.json', 'r') as f:
                self.item_features = json.load(f)
        except FileNotFoundError:
            self.item_features = {}
            
        self.get_inputs()  

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (SidSFTDataset_GPR): 当前实例，由 Python 在调用实例方法时自动传入。
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
            self (SidSFTDataset_GPR): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row = self.data.iloc[idx]
        
        # Heterogeneous Prompt Construction
        user_id = str(row.get('user_id_original_str', ''))
        u_token = self.user_features.get(user_id, '[USER_UNKNOWN]')
        e_token = row.get('e_token', '[CTX_HOMEPAGE]')
        
        try:
            history_item_ids = eval(str(row['history_item_id']))
            history_sids = eval(str(row['history_item_sid']))
        except:
            history_item_ids = []
            history_sids = []
            
        history_str = ""
        for i, item_id in enumerate(history_item_ids):
            item_type = self.item_features.get(str(item_id), {}).get('item_type', 'O')
            token_prefix = '[O_TOKEN]' if item_type == 'O' else '[I_TOKEN]'
            
            if i > 0: history_str += ", "
            sid = history_sids[i] if i < len(history_sids) else ""
            history_str += f"{token_prefix}{sid}"
            
        target_item_id = str(row['item_id'])
        target_sid = str(row['item_sid'])
        target_type = self.item_features.get(target_item_id, {}).get('item_type', 'O')
        target_prefix = '[O_TOKEN]' if target_type == 'O' else '[I_TOKEN]'
        
        target_item_with_prefix = f"{target_prefix}{target_sid}\n"
        
        # Prompt
        input_text = f"{u_token} {e_token} The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?"
        
        history = {
            "input": input_text,
            "output": target_item_with_prefix
        }
        
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        # Final Value for VAFT
        final_value = self.item_features.get(target_item_id, {}).get('final_value', 0.0)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "final_value": final_value
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "final_value": final_value
        }


class EvalSidDataset(CSVBaseDataset):

    """把测试历史 SID 构造为生成 prompt，同时保留真实答案供离线对比。

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
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        """初始化 EvalSidDataset：把测试历史 SID 构造为生成 prompt，同时保留真实答案供离线对比。

        Args:
            self (EvalSidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
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
        super().__init__(train_file, sample, seed, max_len, category, dedup, tokenizer, test)

        self.get_inputs()  

    def generate_example_prompt(self, data_point):
        """把含示例编号的输入和答案包装为示例提示段。

        Args:
            self (EvalSidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 包含 Example 和 Response 标记的文本。
        """
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (EvalSidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": # f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
                "output": target_item + '\n',
                "dedup": target_item_sid == last_history_item_sid}
    
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (EvalSidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        
                
           
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
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
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            
        }


class SidItemFeatDataset(JSONBaseDataset):
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
        super().__init__(item_file=item_file, index_file=index_file, tokenizer=tokenizer, max_len=max_len, test=test, category=category, dedup=False, seed=seed)
        
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


class RLTitle2SidDataset(JSONBaseDataset):
    """构造标题或描述到 SID 的 RL 问答样本。

    Args:
        item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """初始化 RLTitle2SidDataset：构造标题或描述到 SID 的 RL 问答样本。

        Args:
            self (RLTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
            index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(item_file, index_file, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)

        
        self.prompt2history = {}
        self.history2target = {}
        
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.title2sid = {}
        self.sid2description = {}
        self.description2sid = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                # Handle description format
                if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
                    try:
                        desc_list = eval(description)
                        description = desc_list[0] if desc_list else description
                    except:
                        pass
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
                    self.sid2description[combined_sid] = description
                    self.description2sid[description] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create title2sid samples  
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })
        
        # Create description2sid samples
        for description, sid in self.description2sid.items():
            self.data.append({
                'task': 'description2sid',
                'input': description,
                'output': sid
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()

    
    def generate_prompt(self, data_point):
        """按当前任务构造 User Input 与 Response 模板，问答子类留下空回答前缀。

        Args:
            self (RLTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 模型输入模板；普通基类按 data_point.output 决定是否含答案。
        """
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
            response = data_point['output']
        else:  # description2sid
            prompt = f"An item can be described as follows: \"{data_point['input']}\". Which item is it describing?"
            response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        """构造当前记录的 RL prompt/completion，并更新历史到目标的查询映射。

        Args:
            self (RLTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, str] | None: prompt 和 completion；启用跳过重复的子类可能返回 None。
        """
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point['output'] + "\n"
        
        self.prompt2history[prompt] = data_point['input']
        self.history2target[data_point['input']] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,
 
        }


class RLSeqTitle2SidDataset(CSVBaseDataset):
    """构造历史标题序列到下一 SID 的 RL 样本。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, train_file, sample=-1, seed=0, category="", dedup=False):
        """初始化 RLSeqTitle2SidDataset：构造历史标题序列到下一 SID 的 RL 样本。

        Args:
            self (RLSeqTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(train_file, sample, seed, max_len=1024, category=category, dedup=dedup, tokenizer=None, test=False)

        self.prompt2history = {}
        self.history2target = {}
        
        self.get_inputs()
    
    def generate_prompt(self, inter_titles):
        """把历史标题拼成询问下一商品的问题。

        Args:
            self (RLSeqTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            inter_titles (str): 已按交互时间排列并拼接好的历史文本，用于组成推荐问题。

        Returns:
            str: 推荐问题文本。
        """
        return f"Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"
    
    def get_history(self, row):
        # Parse history_item_title field
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (RLSeqTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 历史标题、目标 SID、历史查询键和重复标记。
        """
        history_item_title = eval(row['history_item_title'])
        
        # Format title sequence for prompt
        inter_titles = ", ".join([f'"{title}"' for title in history_item_title])
        
        target_sid = row['item_sid']
        
        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                target_item_id = row.get('item_id', None)
                last_history_item_id = history_item_id[-1] if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False
        
        return {
            "inter_titles": inter_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate,
            "history_str": "::".join(history_item_title)
        }
    
    def generate_formatted_prompt(self, prompt, response):
        """把问题放进 User Input 模板，并留下空的 Response 前缀。

        Args:
            self (RLSeqTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            prompt (str): 待编码或包装为请求模板的文本。
            response (str): 预留的答案文本参数；当前格式化函数只输出空回答前缀，不使用它。

        Returns:
            str: 尚未包含正确答案的 prompt。
        """
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        """构造当前记录的 RL prompt/completion，并更新历史到目标的查询映射。

        Args:
            self (RLSeqTitle2SidDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, str] | None: prompt 和 completion；启用跳过重复的子类可能返回 None。
        """
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Generate prompt using title sequence
        prompt = self.generate_prompt(history_data['inter_titles'])
        target = history_data['target_sid'] + '\n'
        
        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        
        self.prompt2history[formatted_prompt] = history_data['history_str']
        self.history2target[history_data['history_str']] = target
        
        return {
            "prompt": formatted_prompt,
            "completion": target,

        }


class RLSid2TitleDataset(JSONBaseDataset):
    """构造单个 SID 到商品标题的 RL 问答样本。

    Args:
        item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """初始化 RLSid2TitleDataset：构造单个 SID 到商品标题的 RL 问答样本。

        Args:
            self (RLSid2TitleDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
            index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        super().__init__(item_file, index_file, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)

        self.prompt2history = {}
        self.history2target = {}
        
        # Build sid2title mapping
        self.sid2title = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()
    
    def generate_prompt(self, data_point):
        """按当前任务构造 User Input 与 Response 模板，问答子类留下空回答前缀。

        Args:
            self (RLSid2TitleDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            data_point (dict[str, object]): 当前任务记录；普通模板使用 input/output，问答模板还使用 task 等字段。

        Returns:
            str: 模型输入模板；普通基类按 data_point.output 决定是否含答案。
        """
        prompt = f'What is the title of item "{data_point["input"]}"?'
        response = data_point['output']
        
        return f"""### User Input: 
{prompt}

### Response:\n"""
    
    def pre(self, idx):
        """构造当前记录的 RL prompt/completion，并更新历史到目标的查询映射。

        Args:
            self (RLSid2TitleDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, str] | None: prompt 和 completion；启用跳过重复的子类可能返回 None。
        """
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point['output'] + "\n"
        
        self.prompt2history[prompt] = data_point['input']
        self.history2target[data_point['input']] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,

        }


class RLSidhis2TitleDataset(BaseDataset):
    """构造历史 SID 到下一商品标题的 RL 样本。

    Args:
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, train_file, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """初始化 RLSidhis2TitleDataset：构造历史 SID 到下一商品标题的 RL 样本。

        Args:
            self (RLSidhis2TitleDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
            item_file (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
            index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
            sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
            seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
            category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
            dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。

        Returns:
            None: 完成实例初始化。
        """
        BaseDataset.__init__(self, tokenizer=None, max_len=1024, test=False, category=category, dedup=dedup, seed=seed)

        # Initialize CSV part
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Initialize JSON part
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.prompt2history = {}
        self.history2target = {}
        
        # Build item_id to title mapping
        self.id2title = {}
        for item_id, features in self.item_feat.items():
            self.id2title[item_id] = features['title']
        
        self.get_inputs()

    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (RLSidhis2TitleDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = "::".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        
        # Get target item title from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2title:
            target_title = self.id2title[target_item_id]
        else:
            target_title = f"Unknown item {target_item_id}"
        
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        
        return {
            "input": f"The user has interacted with items {history} in chronological order. Can you predict the title of the next item that the user may expect? Analyze user preferences and then predict the title of the next item.",
            "output": target_title + "\n",
            "history_str": history_str,
            "dedup": target_item_sid == last_history_item_sid
        }
    
    def pre(self, idx):
        """构造当前记录的 RL prompt/completion，并更新历史到目标的查询映射。

        Args:
            self (RLSidhis2TitleDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, str] | None: prompt 和 completion；启用跳过重复的子类可能返回 None。
        """
        history = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history['dedup']:
            return None
        
        target_item = history['output']
        history['output'] = ''
           
        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item
        
        return {
            "prompt": prompt,
            "completion": target_item,

        }


class FusionSeqRecDataset(BaseDataset):
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
        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        
        # Initialize CSV part
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Initialize JSON part
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

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


class TitleHistory2SidSFTDataset(BaseDataset):
    """构造历史标题到下一 SID 的监督样本。

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
        """初始化 TitleHistory2SidSFTDataset：构造历史标题到下一 SID 的监督样本。

        Args:
            self (TitleHistory2SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
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
        BaseDataset.__init__(self, tokenizer, max_len, test, category, dedup, seed)
        # Initialize CSV part
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Initialize JSON part
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        
        # Build item_id to semantic ID mapping
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3:
                combined_sid = sids[0] + sids[1] + sids[2]
                self.id2sid[item_id] = combined_sid
        
        self.get_inputs()
    
    def get_history(self, row):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (TitleHistory2SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row (pandas.Series): 一条 CSV 记录，包含历史和目标字段；历史列表在 CSV 中以字符串保存。

        Returns:
            dict[str, object]: 用于 prompt 的 input、正确 output，以及类中构造的历史键或重复标记。
        """
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])
        
        # Format title sequence for prompt
        history_titles = ", ".join([f'"{title}"' for title in history_item_title])
        
        # Get target item's semantic ID from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2sid:
            target_sid = self.id2sid[target_item_id]
        else:
            target_sid = target_item_id  # Fallback to item_id if no semantic ID found
        
        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                last_history_item_id = str(history_item_id[-1]) if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False
        
        return {
            "input": f"The user has interacted with the following {self.category} items in chronological order: {history_titles}. Can you predict the next item the user may expect?",
            "output": target_sid + "\n",
            "history_titles": history_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (TitleHistory2SidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        target_output = history_data['output']
        history_data['output'] = ''
        
        prompt = self.generate_prompt(history_data)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target_output, bos=False, eos=True)
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


class PreferenceSFTDataset(BaseDataset):
    """用历史监督生成偏好解释与目标 SID，消费外部准备好的偏好文本。

    Args:
        user_preference_file (str): 偏好 JSON/JSONL 文件路径；记录包含 user、split、user_preference 和 context。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, user_preference_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        """初始化 PreferenceSFTDataset：用历史监督生成偏好解释与目标 SID，消费外部准备好的偏好文本。

        Args:
            self (PreferenceSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            user_preference_file (str): 偏好 JSON/JSONL 文件路径；记录包含 user、split、user_preference 和 context。
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
        super().__init__(tokenizer, max_len, test, category, dedup, seed)
        # Load user preferences - handle both JSON and JSONL formats
        with open(user_preference_file, 'r') as f:
            try:
                preference_data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format (multiple JSON objects, one per line)
                f.seek(0)
                preference_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        preference_data.append(json.loads(line))
        
        # Handle new flat structure: each item is a separate training sample
        self.training_samples = []
        
        for item in preference_data:
            if item.get('split') == 'train':  # Only process train data
                user_id = item['user']
                preference_text = item.get('user_preference', '')
                context = item.get('context', {})
                history_items = context.get('history_items', [])
                target_item = context.get('target_item')
                
                # Create interaction history by combining history_items and target_item
                interaction_history = history_items + ([target_item] if target_item is not None else [])
                
                # Each item becomes a separate training sample
                self.training_samples.append({
                    'user_id': user_id,
                    'preference_text': preference_text,
                    'interaction_history': interaction_history
                })
        
        # Load index mapping
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        # Find users with preferences and prepare data
        self.data = self._prepare_preference_data()
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()
    
    def _prepare_preference_data(self):
        """从偏好记录里提取至少两次交互，以最后商品为目标、其余为历史。

        Args:
            self (PreferenceSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: 含 user_id、user_preference、input_history、target_item_id 的样本。
        """
        matched_data = []
        
        for sample in self.training_samples:
            interaction_history = sample['interaction_history']
            # Skip samples without sufficient interaction history (need at least 2 items)
            if not interaction_history or len(interaction_history) < 2:
                continue
            
            # Use all items except the last one as input history
            # Use the last item as the target to predict
            input_history = interaction_history[:-1]  # All but last item
            target_item = interaction_history[-1]     # Last item as target
            
            # Create data point from each training sample
            row_dict = {
                'user_id': sample['user_id'],
                'user_preference': sample['preference_text'],
                'input_history': input_history,
                'target_item_id': target_item
            }
            
            matched_data.append(row_dict)
        
        return matched_data
    
    def _convert_to_semantic_ids(self, item_ids):
        """把商品编号查成前三层 SID；缺失索引或层数不足时回退到编号字符串。

        Args:
            self (PreferenceSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            item_ids (list[int | str]): 待查表的商品编号序列，不是 tokenizer 的词表编号。

        Returns:
            list[str]: 与输入顺序一致的 SID 或回退编号。
        """
        semantic_ids = []
        
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.indices:
                sids = self.indices[item_id_str]
                if len(sids) >= 3:
                    # Combine the three semantic IDs
                    combined_sid = sids[0] + sids[1] + sids[2]
                    semantic_ids.append(combined_sid)
                else:
                    semantic_ids.append(item_id_str)
            else:
                semantic_ids.append(item_id_str)
        
        return semantic_ids
    
    def get_history(self, row_data):
        """从当前记录提取时间有序历史、监督目标和必要的查询/重复标记。

        Args:
            self (PreferenceSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row_data (dict[str, object]): 已整理的偏好样本，包含 input_history、target_item_id 和 user_preference。

        Returns:
            dict[str, object]: 历史 SID 输入和包含偏好解释及 SID 的答案。
        """
        # Get input history item IDs (all but last item) and convert to semantic IDs
        input_history_ids = row_data['input_history']
        history_semantic_ids = self._convert_to_semantic_ids(input_history_ids)
        
        # Format semantic IDs as a comma-separated string
        history_str = ", ".join(history_semantic_ids)
        
        # Get user preference
        user_preference = row_data['user_preference']
        
        # Get target item semantic ID
        target_item_id = row_data['target_item_id']
        target_semantic_ids = self._convert_to_semantic_ids([target_item_id])
        target_sid = target_semantic_ids[0] if target_semantic_ids else str(target_item_id)
        
        result = {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you analyze the user's preferences and predict the next item?",
            "output": f"### Reasoning:\n{user_preference}\n### Response:\n{target_sid}",
            "history_semantic_ids": history_semantic_ids,
            "user_preference": user_preference,
            "target_sid": target_sid
        }
        # print(result)
        return result
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (PreferenceSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Analyze the user's interaction history, provide insights about their preferences, and predict the next item's semantic ID.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row_data = self.data[idx]
        history_and_pref = self.get_history(row_data)
        
        # Skip empty histories or missing targets
        if not history_and_pref['history_semantic_ids'] or not history_and_pref['target_sid']:
            return None
        
        target_output = history_and_pref['output']
        history_and_pref['output'] = ''
        
        prompt = self.generate_prompt(history_and_pref)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target_output, bos=False, eos=True)
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


class UserPreference2sidSFTDataset(BaseDataset):
    """将已有偏好解释与历史一起作为输入，监督生成下一商品 SID。

    Args:
        user_preference_file (str): 偏好 JSON/JSONL 文件路径；记录包含 user、split、user_preference 和 context。
        index_file (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        tokenizer (transformers.PreTrainedTokenizerBase | None): 分词器，负责文本与 token ID 的转换；部分元数据类允许 None 以返回原始任务记录。
        max_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        test (bool): 是否只构造推理输入；为 True 时不把答案 token 与 labels 加入返回样本。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        dedup (bool): 重复目标处理开关；仅部分子类据此跳过目标等于最后历史商品的样本，并非统一过滤。
    """
    def __init__(self, user_preference_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        """初始化 UserPreference2sidSFTDataset：将已有偏好解释与历史一起作为输入，监督生成下一商品 SID。

        Args:
            self (UserPreference2sidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            user_preference_file (str): 偏好 JSON/JSONL 文件路径；记录包含 user、split、user_preference 和 context。
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
        super().__init__(tokenizer, max_len, test, category, dedup, seed)

        # Load user preferences - handle both JSON and JSONL formats
        with open(user_preference_file, 'r') as f:
            try:
                preference_data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format (multiple JSON objects, one per line)
                f.seek(0)
                preference_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        preference_data.append(json.loads(line))
        
        # Handle new flat structure: each item is a separate training sample
        self.training_samples = []
        
        for item in preference_data:
            if item.get('split') == 'train':  # Only process train data
                user_id = item['user']
                preference_text = item.get('user_preference', '')
                context = item.get('context', {})
                history_items = context.get('history_items', [])
                target_item = context.get('target_item')
                
                # Create interaction history by combining history_items and target_item
                interaction_history = history_items + ([target_item] if target_item is not None else [])
                
                # Each item becomes a separate training sample
                self.training_samples.append({
                    'user_id': user_id,
                    'preference_text': preference_text,
                    'interaction_history': interaction_history
                })
        
        # Load index mapping
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        # Prepare training data from preference file interaction histories
        self.data = self._prepare_sequence_data()
        
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        
        self.get_inputs()
    
    def _prepare_sequence_data(self):
        """从偏好记录里提取至少两次交互，以最后商品为目标、其余为历史。

        Args:
            self (UserPreference2sidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[dict[str, object]]: 含 user_id、user_preference、input_history、target_item_id 的样本。
        """
        matched_data = []
        
        for sample in self.training_samples:
            interaction_history = sample['interaction_history']
            
            # Skip samples without sufficient interaction history (need at least 2 items)
            if not interaction_history or len(interaction_history) < 2:
                continue
            
            # Use all items except the last one as input history
            # Use the last item as the target to predict
            input_history = interaction_history[:-1]  # All but last item
            target_item = interaction_history[-1]     # Last item as target
            
            row_dict = {
                'user_id': sample['user_id'],
                'user_preference': sample['preference_text'],
                'input_history': input_history,
                'target_item_id': target_item
            }
            
            matched_data.append(row_dict)
        
        return matched_data
    
    def _convert_to_semantic_ids(self, item_ids):
        """把商品编号查成前三层 SID；缺失索引或层数不足时回退到编号字符串。

        Args:
            self (UserPreference2sidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            item_ids (list[int | str]): 待查表的商品编号序列，不是 tokenizer 的词表编号。

        Returns:
            list[str]: 与输入顺序一致的 SID 或回退编号。
        """
        semantic_ids = []
        
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.indices:
                sids = self.indices[item_id_str]
                if len(sids) >= 3:
                    # Combine the three semantic IDs
                    combined_sid = sids[0] + sids[1] + sids[2]
                    semantic_ids.append(combined_sid)
                else:
                    semantic_ids.append(item_id_str)
            else:
                semantic_ids.append(item_id_str)
        
        return semantic_ids
    
    def get_input_and_target(self, row_data):
        """把历史和已有偏好解释放入 prompt，并提取下一商品 SID 答案。

        Args:
            self (UserPreference2sidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            row_data (dict[str, object]): 已整理的偏好样本，包含 input_history、target_item_id 和 user_preference。

        Returns:
            dict[str, object]: 含 input、output、history_semantic_ids、user_preference、target_sid。
        """
        # Get input history item IDs and convert to semantic IDs
        input_history_ids = row_data['input_history']
        history_semantic_ids = self._convert_to_semantic_ids(input_history_ids)
        
        # Format semantic IDs as a comma-separated string
        history_str = ", ".join(history_semantic_ids)
        
        # Get user preference
        user_preference = row_data['user_preference']
        
        # Get target item semantic ID
        target_item_id = row_data['target_item_id']
        target_semantic_ids = self._convert_to_semantic_ids([target_item_id])
        target_sid = target_semantic_ids[0] if target_semantic_ids else str(target_item_id)
        
        return {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you analyze the user's preferences?\n### Reasoning:\n{user_preference}\nCan you predict the next possible item that the user may expect?",
            "output": target_sid,
            "history_semantic_ids": history_semantic_ids,
            "user_preference": user_preference,
            "target_sid": target_sid
        }
    
    def pre(self, idx):
        """构造当前任务的 prompt，编码答案，并用 -100 屏蔽非答案位置的监督。

        Args:
            self (UserPreference2sidSFTDataset): 当前实例，由 Python 在调用实例方法时自动传入。
            idx (int): 样本在当前数据集中的位置索引，从 0 开始。

        Returns:
            dict[str, object] | None: input_ids/attention_mask，训练时含 labels；GPR 还含 final_value，部分跳过分支返回 None。
        """
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user interaction history and preference analysis, predict the next item's semantic ID.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row_data = self.data[idx]
        input_and_target = self.get_input_and_target(row_data)
        
        # Skip empty histories or missing targets
        if not input_and_target['history_semantic_ids'] or not input_and_target['target_sid']:
            return None
        
        target_output = input_and_target['output']
        input_and_target['output'] = ''
        
        prompt = self.generate_prompt(input_and_target)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }
        
        golden_tokens = self.tokenizer.encode(target_output + '\n', bos=False, eos=True)
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
