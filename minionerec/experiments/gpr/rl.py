from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
import torch
from minionerec.datasets.recommendation import D3Dataset, SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset, RLSid2TitleDataset, RLSidhis2TitleDataset
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from minionerec.training.trainer import ReReTrainer
from minionerec.models.sasrec import SASRec
from fire import Fire
import pickle
import math
import json
from sklearn.metrics import ndcg_score
import re

os.environ['WANDB_MODE'] = 'disabled'

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

def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    
    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,
):
    """构造混合 RL 样本与奖励函数，加载推荐策略并启动 ReReTrainer，保存最终模型。

    Args:
        model_path (str): 模型 checkpoint 目录或模型标识；需要配套的模型配置、权重和 tokenizer。
        seed (int | None): 随机种子；用于固定抽样、初始化或打乱顺序，None 表示不显式固定。
        train_file (str): 交互 CSV 路径；包含历史商品和目标商品字段，实际任务决定使用标题还是 SID。
        eval_file (str): 验证交互 CSV 路径，用于训练过程中的验证。
        info_file (str): 商品目录 TXT 路径，首列为 SID，后续列为标题与商品 ID；用于约束或合法性检查。
        category (str): 商品领域名称，用于数据路径、输出命名或任务提示语，具体由当前入口决定。
        wandb_project (str): Weights & Biases 项目名，写入训练日志环境配置。
        wandb_run_name (str): 训练运行名，用于 TrainingArguments 和日志记录。
        output_dir (str): 输出目录；转换入口写数据，训练入口写 checkpoint 和 tokenizer。
        train_batch_size (int): 对应阶段每设备的候选样本行数；同一 prompt 会重复 num_generations 次。
        eval_batch_size (int): 对应阶段每设备的候选样本行数；同一 prompt 会重复 num_generations 次。
        gradient_accumulation_steps (int): 累积多少个 micro batch 的梯度后执行一次优化更新。
        temperature (float): 生成分布的温度；用于调整候选采样时的概率集中程度，需大于 0。
        add_gt (bool): 是否用真实答案替换部分生成候选；替换周期由当前实现的目标分组逻辑决定。
        eval_step (float | int): 验证间隔；小于 1 的比例值由 Trainer 按计划训练步数解释。
        num_generations (int): 每个原始样本重复的次数；RL 中对应同一 prompt 的候选组大小。
        num_train_epochs (int | float): 训练遍历数据集的轮数，传给相应训练器。
        learning_rate (float): 优化器初始学习率。
        beta (float): 参考策略 KL 惩罚系数。
        beam_search (bool): 是否每组取一个 prompt 用多 beam 生成候选；当前训练配置仍开启 do_sample。
        test_during_training (bool): 是否在准备 RL batch 时额外生成候选并统计 HR/NDCG；不是独立离线评测入口。
        dynamic_sampling (bool): 是否在非 beam 分支多生成约 1.5 倍候选，再选择目标和较多样的结果。
        mask_all_zero (bool): 预留的全零奖励屏蔽开关；当前入口没有把它接入训练器。
        sync_ref_model (bool): 是否注册参考模型同步回调，使 reference 随训练按配置更新。
        test_beam (int): beam 搜索宽度；相应生成配置通常返回同样数量的候选序列。
        reward_type (str): 奖励选项：rule、ranking、ranking_only、semantic 或 sasrec，决定传给 Trainer 的评分函数。
        sample_train (bool): 若启用且模型路径含 sft，则保留打乱后训练数据的后 80%。
        ada_path (str): semantic 奖励使用的商品向量 pickle 路径，行序必须与 info 商品编号一致。
        cf_path (str): SASRec 奖励模型 state_dict 的路径，结构和商品编号需与当前目录匹配。
        sid_index_path (str): 商品 ID 到各层 SID token 列表的 JSON 文件路径。
        item_meta_path (str): 商品元数据 JSON 路径，键为商品 ID 字符串，值含 title、description 等字段。
        dapo (bool): 是否按所有有效 completion token 的总数归一化 RL loss。
        gspo (bool): 是否使用序列平均 log-ratio 的损失变体；dapo=True 时优先执行 DAPO 分支。

    Returns:
        None: 在 output_dir 及 final_checkpoint 中保存训练产物。
    """
    torch.backends.cuda.enable_flash_sdp(False)  
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)
    
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    print(category)
    
    
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Extract semantic_id (first column) from the format: semantic_id \t item_title \t item_id
        item_name = [_.split('\t')[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    # Parse semantic IDs for HEPO
    def parse_sid(sid):
        """按方括号正则拆分 SID 片段；当前主数据的尖括号 SID 会得到空列表。

        Args:
            sid (str): 待拆分的 SID 字符串；此 GPR 实现按方括号匹配，不匹配主数据的尖括号格式。

        Returns:
            list[str]: 匹配到的方括号片段。
        """
        return re.findall(r'\[.*?\]', sid)

    item2id_parts = {}
    for name in item_name:
        parts = parse_sid(name)
        item2id_parts[name] = tuple(parts)

    def hepo_reward(prompts, completions):
        """按候选与目标连续匹配的前三层代码给予 0、0.2、0.5 或 1.0 的层级奖励。

        Args:
            prompts (list[str]): 候选对应的输入文本；相同 prompt 连续重复以组成奖励比较组。
            completions (list[str]): 模型生成的候选答案文本，与 prompts 或当前选择组顺序一致。

        Returns:
            list[float]: 每条候选的分层奖励，受 parse_sid 格式约束。
        """
        history = [prompt2history.get(prompt, "") for prompt in prompts]
        targets_full_sid = [history2target.get(elm, "") for elm in history]
    
        rewards = []
        for i, comp_full_sid in enumerate(completions):
            comp_full_sid = comp_full_sid.strip(" \n\"")
            target_full_sid = targets_full_sid[i].strip(" \n\"")
    
            if target_full_sid not in item2id_parts:
                rewards.append(0.0)
                continue
    
            target_parts = item2id_parts[target_full_sid]
            comp_parts = parse_sid(comp_full_sid)
            
            reward = 0.0
            
            # Hierarchical Reward
            if len(comp_parts) > 0 and len(target_parts) > 0 and comp_parts[0] == target_parts[0]:
                reward = 0.2
                if len(comp_parts) > 1 and len(target_parts) > 1 and comp_parts[1] == target_parts[1]:
                    reward = 0.5
                    if len(comp_parts) > 2 and len(target_parts) > 2 and comp_parts[2] == target_parts[2]:
                        reward = 1.0
            
            rewards.append(reward)
        return rewards

    sample = -1
    train_datasets = []
    # train_data = D3Dataset(train_file, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data)
    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)
    train_data2 = RLTitle2SidDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    train_datasets.append(train_data2)
    train_data3 = RLSeqTitle2SidDataset(train_file, category=category_dict[category], sample=10000)
    train_datasets.append(train_data3)
    # train_data4 = RLSid2TitleDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data4)
    # train_data5 = RLSidhis2TitleDataset(train_file, item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data5)
    # train_data6 = RLTitle2Sid_1LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data6)
    # train_data7 = RLTitle2Sid_2LayerDataset(item_file=item_meta_path, index_file=sid_index_path, category=category_dict[category], sample=sample)
    # train_datasets.append(train_data7)
    train_data = ConcatDataset(train_datasets)
    # eval_data = D3Dataset(eval_file, category=category_dict[category], sample=sample)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict({k : [elm[k] for elm in train_data] for k in train_data[0].keys()})
    train_dataset = train_dataset.shuffle(seed=seed) 
    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(range(int(0.2 * len(train_dataset)), len(train_dataset)))
    eval_dataset = Dataset.from_dict({k : [elm[k] for elm in eval_data] for k in eval_data[0].keys()})
    eval_dataset = eval_dataset.shuffle(seed=seed)
    

    # prompt2history = {**train_data.prompt2history, **eval_data.prompt2history}
    # history2target = {**train_data.history2target, **eval_data.history2target}

    prompt2history = {}
    history2target = {}
    
    # Collect prompt2history and history2target from all train datasets
    for dataset in train_datasets:
        if hasattr(dataset, 'prompt2history'):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, 'history2target'):
            history2target.update(dataset.history2target)
    
    # Add eval_data mappings
    if hasattr(eval_data, 'prompt2history'):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, 'history2target'):
        history2target.update(eval_data.history2target)

    print("train_dataset: ", train_dataset)
    print("eval_dataset: ", eval_dataset)

    llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    device = llm_model.device
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    if reward_type == "sasrec":
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path))
        model.eval()
    if reward_type == "semantic":
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(llm_model.device)

    print("Load item_ada_embd successfully.")

    ndcg_rewards = [-1.0/math.log2(i+2) for i in range(num_generations)]
    ndcg_rewards = [-elm/sum(ndcg_rewards) for elm in ndcg_rewards]


    def ndcg_rule_reward(prompts, completions):
        """在至少一次命中的候选组内按位置惩罚错误项；全错组给零奖励。

        Args:
            prompts (list[str]): 候选对应的输入文本；相同 prompt 连续重复以组成奖励比较组。
            completions (list[str]): 模型生成的候选答案文本，与 prompts 或当前选择组顺序一致。

        Returns:
            list[float]: 命中项为 0，错误项为负排名权重或 0。
        """
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        repeat = num_generations
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\"") == targets[i].strip("\n\""):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i%num_generations])
            
            if (i+1)%num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        
        return rewards

    def rule_reward(prompts, completions):
        """比较每条候选与查表得到的真实 SID，命中记 1，未命中记 0。

        Args:
            prompts (list[str]): 候选对应的输入文本；相同 prompt 连续重复以组成奖励比较组。
            completions (list[str]): 模型生成的候选答案文本，与 prompts 或当前选择组顺序一致。

        Returns:
            list[float]: 与候选逐一对应的准确匹配奖励。
        """
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        rewards = []

        for i, completion in enumerate(completions):

            if completion.strip("\n\" ") == targets[i].strip("\n\" "):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def semantic_reward(prompts, completions):
        """查出真实与生成商品的向量，计算逐样本余弦相似度奖励。

        Args:
            prompts (list[str]): 候选对应的输入文本；相同 prompt 连续重复以组成奖励比较组。
            completions (list[str]): 模型生成的候选答案文本，与 prompts 或当前选择组顺序一致。

        Returns:
            torch.Tensor: shape [B] 的相似度，商品必须能在目录中查到。
        """
        history = [prompt2history[prompt] for prompt in prompts]
        targets = [history2target[elm] for elm in history]
        target_ids = [item2id[elm.strip("\"\n")] for elm in targets]
        completions = [elm.strip("\"\n") for elm in completions]
        for i, completion in enumerate(completions):
            if completion not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {completion}")
                print("==============================")
        completion_ids = [item2id[elm] for elm in completions]
        rewards =  torch.cosine_similarity(item_ada_embd[target_ids], item_ada_embd[completion_ids], dim=-1)
        print(rewards)
        return rewards

    def cf_reward(prompts, completions):
        """用 SASRec 读取历史并取生成商品的预测分数作为奖励。

        Args:
            prompts (list[str]): 候选对应的输入文本；相同 prompt 连续重复以组成奖励比较组。
            completions (list[str]): 模型生成的候选答案文本，与 prompts 或当前选择组顺序一致。

        Returns:
            torch.Tensor: shape [B] 的候选分数；历史和商品编号需与基线一致。
        """
        history = [prompt2history[prompt] for prompt in prompts]
        history_list = [elm.split("::") for elm in history]
        pred_ids = []
        for i, elm in enumerate(completions):
            elm = elm.strip("\n\"")
            if elm not in item_name:
                # print("========Invalid Item========")
                # print(f"Invalid item: {elm}")
                # print(f"Prompt: {prompts[i]}")
                # print("============================")
                pred_ids.append(random.randint(0, item_num-1))
            else:
                pred_ids.append(item2id[elm])
        
        len_lis = []
        history_ids = []
        for his in history_list:
            # Filter out items not in item2id mapping
            his = [item2id[elm] for elm in his if elm in item2id]
            # If all items filtered out, use a default item
            if len(his) == 0:
                his = [item_num]  # padding item
            len_lis.append(len(his))
            if len(his) < len_seq: 
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his[:len_seq])  # truncate if too long
        
        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)    
        
        with torch.no_grad():
            predictions = model.forward_eval(seq, torch.tensor(np.array(len_lis)).to(device))
            scores = torch.gather(predictions, 1,  pred.view(-1, 1)).view(-1)
        return scores
    


    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward
    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = [cf_reward, hepo_reward] # Combine rewards
    
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    training_args = GRPOConfig(output_dir=output_dir,
                                save_steps=0.1,
                                save_total_limit=20,
                                eval_strategy="steps",
                                max_completion_length=128,
                                num_generations=num_generations,
                                temperature=temperature,
                                sync_ref_model=sync_ref_model,
                                per_device_eval_batch_size=eval_batch_size,
                                per_device_train_batch_size=train_batch_size,
                                gradient_accumulation_steps=gradient_accumulation_steps,  
                                eval_steps=eval_step, 
                                logging_steps=1, 
                                learning_rate=learning_rate,
                                beta=beta,
                                warmup_ratio=0.03,
                                max_grad_norm= 0.3,
                                num_train_epochs=num_train_epochs,
                                bf16=True,
                                optim="paged_adamw_32bit",
                                lr_scheduler_type="cosine", 
                                save_strategy="steps",
                                report_to="wandb",
                                run_name=wandb_run_name,
                            )
    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
if __name__ == "__main__":
    Fire(train)
