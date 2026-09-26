> **目录已整理**：主代码在 `minionerec/`，推荐启动脚本在 `scripts/`。请从项目根目录用 `python -m minionerec.…` 运行；`rq/` 保持独立运行方式。旧路径对照、运行说明和遗留代码检查见 [迁移说明](docs/09_目录迁移与遗留代码检查.md)，源码阅读从 [中文文档](docs/README.md) 开始。

<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**面向规模化生成式推荐的
开源框架**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 技术报告</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>
</div>

**MiniOneRec** 是一个完全开源的**生成式推荐**框架，提供涵盖 **SID 构建**、**监督微调（SFT）**和面向推荐的**强化学习（RL）**的端到端流程。

---

## 📢 更新公告

- 2026-05-13 — 新增 TS-Rec 代码，实现参考论文 [Fine-grained Semantics Integration for Large Language Model-based Recommendation（面向大语言模型推荐的细粒度语义融合）](https://arxiv.org/pdf/2602.22632) 中提出的方法。感谢贡献者为此次更新付出的努力与支持。

- 2026-01-04 — 如果使用 Instruct 模型复现的结果与项目报告指标有差异，请检查评测日志中的 CC 指标是否非零（参见 minionerec/evaluation/metrics.py）。非零表示评测扫描到了无效商品 SID，需要检查约束解码是否生效。原项目团队当时怀疑问题可能与 Transformers 等依赖版本有关，并在排查通用解决方案；当时建议尝试切换到 Qwen2.5-base 等基础模型。

- 2025-12-04 — 新增脚本，支持处理 Amazon23 数据集。

- 2025-12-01 — 修复 minionerec/datasets/recommendation.py 中可能导致 SID–商品对齐任务提前看到答案的问题。该问题源于此前尝试用部分生成轨迹引导完整 SID–商品生成；原项目公告说明其不影响模型性能。

- 2025-11-20 — 更新 **RQ-Kmeans+** 的 SID 构建方法（该方法最早由 **GPR** 提出，原项目将本实现介绍为首次开源复现）。

- 2025-11-19 — 实现基于 Accelerate 的多 GPU 并行文本向量生成，相比原版本提高处理效率，代码位于 rq/text2emb/amazon_text2emb.py。

- 2025-11-19 — 更新 **constrained-RQ-Kmeans** 的 SID 构建方法。

- 2025-11-07 — 感谢大家提交问题反馈！项目已根据反馈发布新的实现。如运行代码时遇到问题，请先更新并查阅**最新版本**。
  
- 2025-11-07 — SFT 阶段新增冻结 LLM 的选项，可仅对新增 SID 词表的 Embedding 行保留训练梯度。

- 2025-10-31 — 现在可以直接下载 MiniOneRec 模型的 **checkpoint（检查点）**。

- 2025-10-31 — 更新 **RQ-Kmeans** 的 SID 构建方法。

---

## 🛠️ 核心技术
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID 构建：MiniOneRec 首先将每件商品转换为紧凑且具有语义信息的 token 序列。** 将商品标题与描述拼接后，输入冻结的文本编码器获取向量，再通过三层 RQ-VAE 将连续向量量化为离散的语义 ID（Semantic ID，简称 SID）。

- **监督微调（SFT）：商品转换为 SID 后，首先对模型进行有监督训练。** 将按时间排序的用户历史行为表示为 token 序列，通过预测下一个 token，学习生成用户下一件可能交互商品的 SID。同时联合训练自然语言与 SID 之间的对齐任务，让模型学习两种表示之间的对应关系，将大语言模型的语言知识关联到离散商品编码。

- **面向推荐的强化学习（RL）：SFT 之后，采用 GRPO 风格的强化学习进一步优化推荐策略。** 针对每个输入生成多个推荐候选，通过组内奖励归一化计算优势，并用 KL 惩罚约束策略相对于参考模型的偏移。由于输出空间是商品 SID 目录，系统使用受约束的 beam 搜索限制合法前缀，以提高候选生成效率；实际有效性仍需结合目录、分词器和生成配置检查。奖励可结合是否命中的二值信号与排名信号，对排名靠前但错误的候选施加更大惩罚，也提供基于协同过滤分数的奖励选项。这些步骤将语言模型与离散商品表示连接起来，构成轻量的生成式推荐流程。

---

## 📊 评测结果

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

---

## 🗂️ 仓库结构

| 文件 / 目录               | 说明                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `scripts/sft.sh`                  | 启动监督微调（SFT）的 Shell 脚本 |
| `minionerec/training/sft.py`                  | SFT 训练入口，负责模型、数据和 Trainer 的配置 |
| `minionerec/experiments/gpr/sft.py`              | GPR 实验分支：定义价值感知微调（VAFT）加权损失，当前默认数据集尚未提供价值权重字段 |
| `scripts/rl.sh`                   | 启动强化学习（RL）的 Shell 脚本 |
| `minionerec/training/rl.py`                   | RL 训练入口，负责数据、奖励与训练器的配置 |
| `minionerec/experiments/gpr/rl.py`               | GPR 实验分支：包含层次增强策略优化（HEPO）奖励实现 |
| `minionerec/training/trainer.py`   | 面向生成式推荐的 GRPO 风格训练器 |
| `config/`                | YAML 配置文件 |
| `scripts/evaluate.sh`     | 离线 Top-K 评测调度脚本，运行前需配置模型路径 |
| `minionerec/evaluation/evaluate.py`     | 离线推荐候选生成；HR@K、NDCG@K 由同目录 metrics.py 计算 |
| `minionerec/evaluation/logits_processor.py`                | 受约束解码的 logits 处理器 |
| `minionerec/datasets/recommendation.py`                | SFT、RL 和评测所用的数据集与样本构造 |
| `minionerec/preprocessing/convert_dataset.py`                | 将量化产生的 SID 索引与交互数据转换为 SFT、RL 使用的格式 |
| `minionerec/experiments/gpr/convert_dataset.py`           | GPR 实验分支的数据转换器，处理模拟上下文等字段 |
| `scripts/amazon18_data_process.sh`                | Amazon18 数据过滤与预处理启动脚本 |
| `minionerec/preprocessing/amazon18.py`                | Amazon18 数据预处理实现 |
| `minionerec/experiments/gpr/amazon18.py`            | GPR 实验分支：Amazon18 预处理与模拟异构特征构造 |
| `scripts/amazon23_data_process.sh`                | Amazon23 数据过滤与预处理启动脚本 |
| `minionerec/preprocessing/amazon23.py`                | Amazon23 数据预处理实现 |
| `rq/scripts/amazon_text2emb.sh`                | 通过文本编码模型为 Amazon 商品标题与描述生成向量的启动脚本 |
| `rq/text2emb/amazon_text2emb.py`                | 商品文本向量生成实现 |
| `rq/text2emb/amazon_text2emb_gpr.py`           | GPR 实验分支的商品文本向量生成实现 |
| `rq/models/generate_indices.py`                | 训练 RQ-VAE 后导出商品 SID 索引 |
| `rq/scripts/rqvae.sh`                | 使用 Amazon 商品向量训练 RQ-VAE 的启动脚本 |
| `rq/rqvae.py`                | RQ-VAE 训练入口 |
| `rq/rqkmeans_faiss.py`                | 基于 FAISS 的 RQ-Kmeans 实现 |
| `rq/rqkmeans_constrained.py`                | 带聚类容量约束的 RQ-Kmeans 实现 |
| `rq/scripts/rqkmeans_constrained.sh`                | 使用 Amazon 商品向量训练带约束 RQ-Kmeans 的启动脚本 |
| `rq/rqkmeans_plus.py`                | RQ-Kmeans+ 训练实现 |
| `rq/scripts/rqkmeans_plus.sh`                | 使用 Amazon 商品向量训练 RQ-Kmeans+ 的启动脚本 |
| `rq/models/generate_indices_plus.py`                | 训练 RQ-Kmeans+ 后导出商品 SID 索引 |
| `rq/scripts/generate_indices_plus.sh`                | RQ-Kmeans+ 的 SID 索引导出启动脚本 |
| `requirements.txt`        | Python 依赖列表 |

---

## 🚀 快速开始

可以直接使用项目提供的 Industrial / Office 商品 SID，快速开始训练！
原项目给出的复现资源参考为 4–8 张 A100/H100 GPU；运行前请配置脚本中的模型、数据与输出路径。

### 1. 创建独立的 Python 环境

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. 安装所需依赖

```bash
pip install -r requirements.txt
```

### 3. 监督微调（SFT）

```bash
bash scripts/sft.sh
```

### 4. 面向推荐的强化学习（RL）

```bash
bash scripts/rl.sh
```

### 5. 运行评测脚本

```bash
bash scripts/evaluate.sh
```

---

## 📜 完整流程说明

### 0. 前置条件
- GPU：例如 4–8 张 A100/H100 80 GB，或相当配置
- Python: 3.11

### 1. 环境配置
- **1.1 克隆仓库**
```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```
- **1.2 创建并激活 conda 环境**
```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```
- **1.3 安装依赖**
```
pip install -r requirements.txt
```

### 2. 数据准备

- **2.1 下载原始数据集（可选）**
  可从以下官方页面获取：
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), 
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), 
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
  说明：Industrial 和 Office 数据包含在 Amazon 2018 中。处理 Amazon 2023 时可使用 minionerec/preprocessing/amazon23.py；其他版本需要按原始字段格式调整预处理逻辑。
- **2.2 数据过滤与预处理**
```
python -m minionerec.preprocessing.amazon18 \
     --dataset  your_dataset_type \
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```
- **2.3 将商品文本编码为向量**
```
accelerate launch --num_processes 8 rq/text2emb/amazon_text2emb.py \
     --dataset your_dataset_type \
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID 构建

从 3.1.1、3.1.2、3.1.3、3.1.4 中选择一种构建方案。

- **3.1.1 使用商品向量训练 RQ-VAE**
```
python rq/rqvae.py \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 使用商品向量训练 RQ-Kmeans**

```
conda install faiss-gpu
python rq/rqkmeans_faiss.py --dataset Industrial_and_Scientific # 基于语义向量的 RQ-Kmeans 方法碰撞率相对较高。
```

- **3.1.3 使用商品向量训练带约束的 RQ-Kmeans**
对完整编码冲突的商品，追加一层编号以区分不同商品；同时对聚类大小施加平衡约束，改善码本使用分布。
```
pip install k_means_constrained
pip install polars
bash rq/scripts/rqkmeans_constrained.sh
```

- **3.1.4 使用商品向量训练 RQ-Kmeans+**
```
pip install k_means_constrained
pip install polars
bash rq/scripts/rqkmeans_constrained.sh
bash rq/scripts/rqkmeans_plus.sh
```

- **3.2 生成索引（仅 RQ-VAE 和 RQ-Kmeans+ 需要）**
```
python -m rq.models.generate_indices --ckpt_path /path/to/best_collision_model.pth
# 或者
bash rq/scripts/generate_indices_plus.sh /path/to/best_collision_model.pth
```

- **3.3 转换数据集格式**
```
python -m minionerec.preprocessing.convert_dataset \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/ourput_dir \

```

### 4. 监督微调（SFT）

```
bash scripts/sft.sh \
     --base_model your_model_path \
     --output_dir your_ourput_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

### 5. 面向推荐的强化学习（RL）
> （可选）对于生产规模的数据集，考虑强化学习的成本和边际收益递减，可以只使用数万条规模的较小子集进行 RL 训练。
```
bash scripts/rl.sh \
     --model_path your_model_path \
     --output_dir output_dir \
```

### 6. 离线评测

```
# 先在 scripts/evaluate.sh 中设置 exp_name。
bash scripts/evaluate.sh
```

---

## 🤖 支持的 LLM 服务商

MiniOneRec 的文本辅助工具支持多个 LLM 服务商，可用于用户偏好、商品特征提取等文本增强任务。在 `api_info` 字典中配置服务商：

| 服务商 | `provider` 取值 | 默认 API 地址 | 模型示例 |
|----------|-----------------|------------------|----------------|
| OpenAI | `"openai"` | — | `text-davinci-003` |
| DeepSeek | `"deepseek"` | `https://api.deepseek.com` | `deepseek-chat` |
| [MiniMax](https://www.minimaxi.com) | `"minimax"` | `https://api.minimax.io/v1` | `MiniMax-M2.7`, `MiniMax-M2.5` |

**示例：使用 MiniMax**

```python
api_info = {
    "provider": "minimax",
    "api_key_list": ["your-minimax-api-key"],
    "base_url": "https://api.minimax.io/v1",  # 可选，当前值即为默认地址
}
get_res_batch("MiniMax-M2.7", prompt_list, max_tokens=512, api_info=api_info)
```

---

## 📝 后续计划

原项目计划继续扩展 MiniOneRec 的能力，以下为其路线图；计划不代表当前仓库已经实现：
* ⏱️ **更多 SID 构建算法**：原路线图列出 R-VQ、RQ-Kmeans、RQ-OPQ 和 RQ-VAE-v2（PLUM）；其中 RQ-Kmeans 已在当前仓库提供实现。
* ⚙️ **MiniOneRec-Think**：计划融合对话、推理和个性化推荐，为复杂交互场景提供一体化模块。
* 🔍 **支持更多数据集**：计划增加 Yelp 等常用公开数据集，进一步验证算法的通用性。

---

## 🏫 参与机构  <!-- omit in toc -->

本项目由以下机构参与开发：

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)
 
---

## 🧩 参与贡献

欢迎并感谢所有贡献！如果你有改进 MiniOneRec 的想法，欢迎提交 Pull Request（PR）。

---
## 🙏 致谢

本仓库复用或改编了以下开源项目的部分代码，感谢其作者和贡献者：

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 引用 <!-- omit in toc -->

如果本项目的代码、论文或模型对你有帮助，欢迎引用相关论文 📝，并为项目点一个 Star ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

---

<div align="center">
欢迎社区贡献！🤝
</div>

---

## 📚 按训练时间顺序阅读源码

建议沿着“上一步生成什么文件、下一步如何使用它”的顺序阅读。下面以 **RQ-VAE 构建 SID → SFT → 可选 RL → 离线评测** 为主线；其他量化方案是替代路线，不需要全部依次训练。

| 顺序 | 优先阅读的文件夹 | 关键文件与阅读顺序 | 这一阶段要弄懂什么 |
| --- | --- | --- | --- |
| 1. 原始数据预处理 | [minionerec/preprocessing/](minionerec/preprocessing/) | `amazon18.py`；使用 Amazon23 时改读 `amazon23.py` | 如何过滤用户和商品、按时间排列行为、构造历史与下一商品样本，以及划分训练、验证、测试集。产物包括交互 `.inter` 和商品元数据 `.item.json`。 |
| 2. 商品文本转向量 | [rq/text2emb/](rq/text2emb/) | `amazon_text2emb.py` → `utils.py` 中的文本清洗函数 | 商品标题和描述怎样变成文本编码器输入，再经隐藏状态池化得到商品向量 `.npy`。这里提取向量，不训练推荐 LLM。 |
| 3. 训练 SID 量化模型 | [rq/](rq/) → [rq/models/](rq/models/) | `rqvae.py` → `datasets.py` → `trainer.py` → `models/rqvae.py` → `models/rq.py` → `models/vq.py`；MLP 细节看 `models/layers.py` | 商品向量如何经过 Dataset/DataLoader、编码器、残差量化和解码器；重建损失与量化损失如何驱动训练，并保存 checkpoint。 |
| 4. 导出 SID 并转换数据 | [rq/models/](rq/models/) → [minionerec/preprocessing/](minionerec/preprocessing/) | `rq/models/generate_indices.py` → `minionerec/preprocessing/convert_dataset.py` | 如何从量化 checkpoint 导出商品 SID 索引，再将索引、商品元数据和交互关联，得到推荐训练用的 CSV 与商品目录。运行导出脚本时需传入 checkpoint 路径。 |
| 5. 准备 SFT 样本 | [scripts/](scripts/) → [minionerec/training/](minionerec/training/) → [minionerec/datasets/](minionerec/datasets/) | `scripts/sft.sh` → `training/sft.py::train` → `datasets/recommendation.py` 中的 `SidSFTDataset`、`SidItemFeatDataset`、`FusionSeqRecDataset` | 先看入口如何加载模型、扩充 SID 词表和构造数据集，再追踪历史行为如何变成 prompt、目标商品如何变成 label，以及为什么用 `-100` 屏蔽 prompt 的监督。 |
| 6. 执行 SFT 训练 | [minionerec/training/](minionerec/training/) | 回到 `sft.py`，继续看数据整理器、`TrainingArguments`、`Trainer` 和 `trainer.train()` | 样本怎样组成 batch，模型怎样预测下一个 token、计算 loss 并更新参数。主线 CausalLM 的 Transformer、forward 和语言模型 loss 来自外部 Transformers 模型实现，不在本仓库重新定义。 |
| 7. 推荐强化学习（可选） | [scripts/](scripts/) → [minionerec/training/](minionerec/training/) → [minionerec/datasets/](minionerec/datasets/) | `scripts/rl.sh` → `training/rl.py` → `datasets/recommendation.py` 中的 RL 数据集 → `training/trainer.py::ReReTrainer` | SFT 模型怎样生成一组候选、获得奖励、计算组内优势，再通过策略损失与参考 KL 更新模型。重点读 `_prepare_inputs`、`_get_per_token_logps` 和 `compute_loss`。 |
| 8. 推理与离线评测 | [minionerec/evaluation/](minionerec/evaluation/) | 先看 `scripts/evaluate.sh`，再依次读 `split.py` → `evaluate.py` → `logits_processor.py` → `merge.py` → `metrics.py` | 如何输入用户历史、逐 token 约束 SID 生成、得到候选推荐并计算 HR/NDCG。样本构造还需回看 `datasets/recommendation.py` 中的 `EvalSidDataset`。 |

**第一遍可以暂缓阅读：** `minionerec/experiments/` 是 GPR、TS 等实验分支；传统推荐基线及其 SASRec 奖励分支已移除，主线使用外部因果语言模型。`tests/` 可用于对照输入输出，`config/` 可在阅读分布式启动参数时查阅。

`data/` 和 `ts_rec_data/` 存放数据，可以打开少量样本对照代码，但它们不是训练逻辑目录。如果直接使用仓库已有 SID 与 CSV，可以从第 5 步开始；想理解完整训练流程，则从第 1 步顺读。

更细的函数调用顺序见 [按训练时间顺序读源码](docs/08_按训练时间顺序读源码.md)，新旧文件路径对照见 [目录迁移与遗留代码检查](docs/09_目录迁移与遗留代码检查.md)。


## MiniOneRec + SIM 精排（新增）

MiniOneRec 生成候选商品，SIM 从用户长历史检索相关行为，再通过注意力模型重排候选。模型、数据准备、训练、推理和测试集中在 [minionerec/sim/](minionerec/sim/README.md)。这是新增的学习实现，完整流程、长历史数据要求与论文实现差异见该目录说明。
