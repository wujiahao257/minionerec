import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class TokenExtender:
    """从商品 SID 索引收集去重且排序后的新增词表 token。

    Args:
        data_path (str): SID 索引文件所在目录。
        dataset (str): 数据集名称，用来拼接数据文件名或标记结果。
        index_file (str): 索引文件后缀，与 data_path 和 dataset 拼接；默认 .index.json。
    """
    def __init__(self, data_path, dataset, index_file=".index.json"):
        """初始化 TokenExtender：从商品 SID 索引收集去重且排序后的新增词表 token。

        Args:
            self (TokenExtender): 当前实例，由 Python 在调用实例方法时自动传入。
            data_path (str): SID 索引文件所在目录。
            dataset (str): 数据集名称，用来拼接数据文件名或标记结果。
            index_file (str): 索引文件后缀，与 data_path 和 dataset 拼接；默认 .index.json。

        Returns:
            None: 完成实例初始化。
        """
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        """读取数据集 SID 索引并缓存到 self.indices。

        Args:
            self (TokenExtender): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 更新索引缓存。
        """
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        """遍历商品 SID 的各层 token，去重排序并缓存结果。

        Args:
            self (TokenExtender): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            list[str]: 要添加到语言模型 tokenizer 的代码 token。
        """
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    """计算带预热且倍率不低于 0.1 的余弦学习率倍率。

    Args:
        current_step (int): 当前学习率调度步数。
        num_warmup_steps (int): 学习率预热阶段的优化步数。
        num_training_steps (int): 计划执行的总优化步数，用于计算预热后的进度。
        num_cycles (float): 预热后余弦变化的周期数；0.5 对应半个余弦周期。

    Returns:
        float: 乘到优化器基础学习率上的倍率。
    """
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    """用自定义预热余弦倍率创建 LambdaLR；主 SFT 默认未把它传入 Trainer。

    Args:
        optimizer (torch.optim.Optimizer): 需要应用学习率调度的优化器。
        num_warmup_steps (int): 学习率预热阶段的优化步数。
        num_training_steps (int): 计划执行的总优化步数，用于计算预热后的进度。
        num_cycles (float): 预热后余弦变化的周期数；0.5 对应半个余弦周期。
        last_epoch (int): LambdaLR 恢复调度时的步数位置；-1 表示新建调度器。

    Returns:
        torch.optim.lr_scheduler.LambdaLR: 学习率调度器。
    """
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



class VAFT_Trainer(transformers.Trainer):
    """有 final_value 时按模拟价值加权序列 CE，否则使用父类训练 loss。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get final_value
        """按有效答案 token 平均交叉熵，并乘 log1p(final_value)；缺失权重时回退父类。

        Args:
            self (VAFT_Trainer): 当前实例，由 Python 在调用实例方法时自动传入。
            model (transformers.PreTrainedModel): 当前因果语言模型，输入 token 和 mask，输出词表 logits；loss 路径需要梯度。
            inputs (dict[str, torch.Tensor]): SFT batch：input_ids、attention_mask、labels 为 [B,L]，可选 final_value 为 [B]；权重字段会被 pop。
            return_outputs (bool): 是否连同 loss 返回模型输出；ReReTrainer 不支持 True，VAFT 支持该模式。
            num_items_in_batch (int | torch.Tensor | None): 父类 Trainer 传入的批内计数兼容参数；当前自定义 loss 未用它归一化。

        Returns:
            torch.Tensor | tuple[torch.Tensor, object]: loss 标量；return_outputs=True 时附带模型输出。
        """
        final_values = inputs.pop("final_value", None)
        
        if final_values is None:
             # Fallback to normal loss if final_value is missing
             return super().compute_loss(model, inputs, return_outputs)

        final_values = final_values.to(self.args.device)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # Calculate loss per token
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Reshape to (batch, seq_len)
        loss_per_seq = loss_per_token.view(shift_labels.shape[0], -1)

        # Average loss per sequence (ignoring padding)
        valid_tokens_mask = (shift_labels != -100)
        # Avoid division by zero
        sum_loss = (loss_per_seq * valid_tokens_mask).sum(dim=1)
        num_valid = valid_tokens_mask.sum(dim=1)
        seq_loss = sum_loss / (num_valid + 1e-9)

        # Value Weighting
        # Ensure final_values are positive and broadcastable
        value_weights = torch.log1p(final_values.to(seq_loss.dtype))
        
        # Apply weighted loss
        weighted_loss = (seq_loss * value_weights).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str="",
    eval_file: str="",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str="",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
    """加载因果语言模型、扩充 SID 词表并混合推荐/对齐任务，执行 SFT 后保存模型和 tokenizer。

    Args:
        base_model (str): 模型 checkpoint 目录或模型标识；需要配套的模型配置、权重和 tokenizer。
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        eval_file (str): 验证交互 CSV 路径，用于训练过程中的验证。
        output_dir (str): 输出目录；转换入口写数据，训练入口写 checkpoint 和 tokenizer。
        sample (int): 请求抽取的样本数；小于等于 0 使用全部样本，CSV 基类不会自动限制到数据总数。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        batch_size (int): 一次处理的样本数量；SFT 入口中表示用于推算累积步数的全局目标 batch。
        micro_batch_size (int): SFT 每设备每次 forward 的样本数。
        num_epochs (int | float): 训练遍历数据集的轮数，传给相应训练器。
        learning_rate (float): 优化器初始学习率。
        cutoff_len (int): 训练 token 序列保留的最大长度，超长时保留尾部；test 分支提前返回时不执行此截断。
        group_by_length (bool): 是否让 Trainer 按 token 序列长度组织训练样本以减少 padding。
        freeze_LLM (bool): 是否冻结模型主体，仅打开输入 Embedding 并屏蔽旧词表行的梯度。
        wandb_project (str): Weights & Biases 项目名，写入训练日志环境配置。
        wandb_run_name (str): 训练运行名，用于 TrainingArguments 和日志记录。
        resume_from_checkpoint (str | None): 恢复训练的 checkpoint 路径；None 表示从本次加载的模型开始。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        train_from_scratch (bool): 为 True 时按基础配置随机初始化模型；否则加载预训练权重。
        sid_index_path (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        item_meta_path (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。

    Returns:
        None: 在 output_dir 及 final_checkpoint 中保存训练产物。
    """
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    original_vocab_size = len(tokenizer)
    
    # Add Special Tokens
    new_special_tokens = ['[USER_HIGH_RATING]', '[USER_MID_RATING]', '[USER_LOW_RATING]', '[USER_UNKNOWN]',
                          '[CTX_BROWSE]', '[CTX_SEARCH]', '[CTX_HOMEPAGE]',
                          '[O_TOKEN]', '[I_TOKEN]']
    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Resize embeddings for special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    """将原始词表行的梯度原地置零，保留新增 SID 行的梯度。

                    Args:
                        grad (torch.Tensor): Embedding 权重梯度，shape 为 [词表大小, 隐藏维度]；旧词表行会被原地置零。

                    Returns:
                        torch.Tensor: 修改后的同一个梯度 Tensor。
                    """
                    grad[:original_vocab_size].zero_()
                    return grad
                
                embedding_layer.weight.register_hook(mask_grad)

                print(f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})")

        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params     = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    train_datasets = []
    # train_data1 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_data1 = SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data1)
    train_data2 = SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data2)
    train_data3 = FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    train_datasets.append(train_data3)
    train_data4 = SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    train_datasets.append(train_data4)
    train_data5 = TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
    train_datasets.append(train_data5)
    
    # Add UserPreference2sidSFTDataset for "Thinking" simulation
    pref_file = os.path.join(os.path.dirname(train_file), f"{category}.preference.json")
    if not os.path.exists(pref_file):
         pref_file = f"data/{category}/{category}.preference.json"
    
    if os.path.exists(pref_file):
        print(f"Loading preference data from {pref_file}")
        train_data_pref = UserPreference2sidSFTDataset(user_preference_file=pref_file, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category)
        train_datasets.append(train_data_pref)
    else:
        print(f"Warning: Preference file {pref_file} not found. Skipping Thinking simulation data.")
        
    train_data = ConcatDataset(train_datasets)
    val_data = SidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category)
    # val_data = SFTData(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    
    # Safe creation of HFDataset handling missing keys (like final_value)
    # Assuming train_data[0] has the superset of keys (SidSFTDataset has final_value)
    keys = train_data[0].keys()
    hf_train_dataset = HFDataset.from_dict({k: [v.get(k, 1.0 if k == 'final_value' else None) for v in train_data] for k in keys})
    hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
    
    val_keys = val_data[0].keys()
    hf_val_dataset = HFDataset.from_dict({k: [v.get(k, 1.0 if k == 'final_value' else None) for v in val_data] for k in val_keys}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=42)

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 0.05
    trainer = VAFT_Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step, 
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
