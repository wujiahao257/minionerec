# MiniOneRec 候选召回 + SIM 精排

这是在原项目上新增的第二阶段精排模块。全部 SIM 代码、测试和说明集中在当前目录；原推理入口只增加样本编号和候选分数导出。

## 1. 整体流程

```text
最近行为 → MiniOneRec（SFT 或 RL checkpoint）→ beam 候选 SID + 分数
                                                  ↓
                             SID 映射为商品 ID，碰撞展开、商品去重
                                                  ↓
完整的已发生历史 + 候选商品 → GSU 检索相关行为 → ESU 注意力 → 商品分数
                                                  ↓
                                     候选重排 → Top-K 商品 ID
```

MiniOneRec 在这里同时提供候选集合和初步排序，没有额外新增一个独立粗排模型。SIM 只能重排召回到的商品，不能找回完全漏召回的目标。相同 SID 对应多个商品时，先展开全部商品再精排；相同分数按原始候选顺序稳定排序。

## 2. SIM 实现范围

参考：[SIM 原论文](https://arxiv.org/abs/2006.05639)。论文的核心是 GSU 从长行为序列选出相关子序列，再由 ESU 建模候选与该子序列的关系。这里采用这一结构，针对当前仓库做了学习用实现，并非论文完整复现或生产服务。

| 部分 | 当前实现 |
| --- | --- |
| GSU | 使用现有 Qwen 商品向量的余弦相似度，为每个候选检索 Top-K 历史行为 |
| 长历史扫描 | NumPy 分块扫描全部提供的历史；没有预先只保留最近 10 条 |
| 检索成本 | 精确线性扫描，不是 ANN，也没有论文的在线索引与缓存系统 |
| ESU | 候选作为 query，相关行为作为 key/value，使用可训练多头注意力 |
| 历史距离 | 距末尾的行为步数使用 log2 分桶；不是实际时间间隔 |
| 短期兴趣 | 最近 S 条行为向量的平均值 |
| 打分 | 拼接候选、长短期兴趣、逐元素交互、召回特征，通过 MLP 输出一个 logit |
| 参数更新 | 商品文本向量固定；训练投影层、距离 Embedding、注意力和 MLP |
| 未实现 | 可训练 GSU 辅助目标、类目 hard-search、曝光日志 CTR 建模、生产级服务与 ANN |

DIN 同样可以关注历史兴趣；长序列上的计算开销才是使用 SIM 检索阶段的主要原因。当前实现仍要扫描长历史，只减少送进可训练注意力的行为数，不承诺百万行为下的实时性能。

## 3. 文件阅读顺序

| 文件 | 重点 |
| --- | --- |
| [prepare.py](prepare.py) | `prepare_records`：SID 到商品；`restore_history`：安全恢复历史前缀 |
| [data.py](data.py) | `SemanticSearch.search`：GSU；`CandidateCollator`：构造 batch |
| [model.py](model.py) | `SIMRanker.forward`：ESU、短期兴趣与打分 |
| [train.py](train.py) | 负例、BCE loss、验证集选模型 |
| [engine.py](engine.py) | 候选打分、指标、checkpoint 加载 |
| [rerank.py](rerank.py) | 加载模型，输出 Top-K 商品及前后对比 |
| [tests.py](tests.py) | 空历史、早期兴趣、未来泄漏、SID 碰撞和训练推理闭环 |

## 4. 输入、shape 和标签

设商品总数 N、原商品向量维度 D、隐藏维度 H、候选 batch 大小 B、GSU 返回长度 K、短期长度 S。

```text
商品向量表 [N,D]：归一化、固定；增加第 0 行 padding 后为 [N+1,D]
候选 ID [B] → 查表 [B,D] → 投影 [B,H]
GSU 在 CPU 上扫描每个候选对应的完整历史，只返回 K 条相关行为
选中 ID [B,K] → 查表 [B,K,D] → 投影 + 行为距离 Embedding [B,K,H]
候选 query [B,1,H] + 历史 key/value [B,K,H] → 长期兴趣 [B,H]
短期 ID [B,S] → 投影并 mask 平均 → 短期兴趣 [B,H]
拼接特征 [B,5H+3] → MLP → logits [B]
```

商品原编号 0 是有效商品；构造 batch 时全部编号加一，保证 padding 0 不与它混淆。空历史输出零兴趣，避免全部 mask 的注意力产生 NaN。

标签为“该候选是否等于这条行为样本的真实下一件商品”：匹配为 1，其余为 0。采用 `BCEWithLogitsLoss`，没有人工将目标补到召回结果中。未命中目标的候选组也参与训练，全部标签为 0；整个训练集必须同时有正、负候选。

Amazon 数据只有观察到的行为，没有完整曝光未点击日志。因此其余候选只是代理负例，并不意味着用户不喜欢；输出也不能当作已校准的点击概率。长历史中的过去重复购买可以保留，未来行为不会用于剔除负例。

## 5. 数据准备与防止泄漏

### 先准备独立的召回候选

需要分别为训练、验证、测试样本生成 MiniOneRec 候选。原 [evaluate.py](../evaluation/evaluate.py) 现在额外输出：

```json
{
  "user_id": "A22",
  "history_item_id": [117],
  "item_id": 118,
  "output": "<a_104><b_118><c_176>\n",
  "predict": ["<a_104><b_118><c_176>"],
  "predict_scores": [-1.23]
}
```

`predict_scores` 是 beam 的序列分数，受生成长度惩罚影响，不是概率。单候选生成未提供分数时允许 null，SIM 使用缺失标记。沿用旧的 `predict` 字段，因此原指标入口仍能读取结果。

例如在仓库根目录生成候选（替换路径；验证和测试分别更换 CSV 与输出文件）：

```bash
python -m minionerec.evaluation.evaluate --base_model /path/to/minionerec_checkpoint --category Industrial_and_Scientific --info_file /path/to/items.txt --test_data_path /path/to/sim_train.csv --result_json_data runs/sim/train_predictions.json --num_beams 50 --batch_size 1
```

这里 `test_data_path` 是原入口的参数名，也可以传入专门用于精排训练的 CSV。该入口仍沿用原项目的模型依赖和设备设置，没有改造成 CPU 上的大模型生成器。输出目录会自动创建。

更严谨的实验应按时间划分：早期数据训练召回器；后续互不重叠的数据分别用于精排训练、验证、测试。不要用在某条样本上训练过的召回器生成该样本候选后，就声称获得了无偏效果。可以使用时间留出或 OOF；本模块没有替你训练多份召回器。训练器会拒绝训练/验证重复查询，但仅凭这些无时间戳文件无法自动证明全部时间划分正确。

### 转换候选

```bash
python -m minionerec.sim.prepare --predictions runs/sim/train_predictions.json --index /path/to/dataset.index.json --output runs/sim/train.jsonl
python -m minionerec.sim.prepare --predictions runs/sim/valid_predictions.json --index /path/to/dataset.index.json --output runs/sim/valid.jsonl
python -m minionerec.sim.prepare --predictions runs/sim/test_predictions.json --index /path/to/dataset.index.json --output runs/sim/test.jsonl
```

旧预测 JSON 没有商品编号时，额外传 `--csv /path/to/corresponding.csv`。程序会核对每行输入历史文本和目标 SID，顺序不匹配直接报错。不要对不同分片顺序的预测与 CSV 强行配对。

### 真正使用长历史

原来的 CSV 通常只保留最多 10 条历史，仅运行上面的命令仍然只有短历史。想研究长期兴趣，转换时追加：

```text
--sequences /path/to/dataset.inter.json
```

该文件必须来自与 CSV、SID、商品向量**完全相同的编号体系**。程序在用户完整序列中定位“历史尾部 + 当前目标”，只取目标之前的前缀，不把目标本次行为及后续行为放入特征；同一商品的早期真实行为仍可保留。多个位置都匹配时会报错，不会随意选最后一次。此时需从可信的原始事件位置补充 `history_end`（切片右端索引），不能任意填一个索引。

线上无目标时，可以直接提供仅含过去行为的 JSONL，或提供完整序列和已知的 `history_end`。目标只用于离线位置核验和标签，不进入打分模型。完整序列文件包含未来并不意味着整个文件可以作为每条样本的历史。

特别注意：仓库重新处理的 `data/Amazon18/Industrial_and_Scientific/` 与原附带 `data/Amazon/index/Industrial_and_Scientific.*` 商品映射不同，不能混用。需要先对新数据生成配套向量、SID 和 CSV。文件指纹可以防止训练/推理中途换文件，但无法证明用户最初交给程序的不同文件在语义上一定对应。

## 6. 训练与精排

使用已经安装 PyTorch 和 NumPy 的环境，不需要为精排单独加载 Qwen。所有命令在项目根目录执行。默认 CPU，可将 `--device cpu` 改成 `cuda` 或设备支持的 `mps`。

```bash
python -m minionerec.sim.train --train runs/sim/train.jsonl --valid runs/sim/valid.jsonl --embeddings /path/to/dataset.emb-qwen-td.npy --index /path/to/dataset.index.json --output runs/sim/sim.pt --epochs 5 --batch-size 128 --search-k 32 --short-length 10 --hidden-dim 64 --heads 4 --device cpu

python -m minionerec.sim.rerank --candidates runs/sim/test.jsonl --checkpoint runs/sim/sim.pt --embeddings /path/to/dataset.emb-qwen-td.npy --index /path/to/dataset.index.json --output runs/sim/recommendations.json --top-k 10 --device cpu
```

训练保存验证集 NDCG@10 最好的 checkpoint 和 `.metrics.json`。checkpoint 包含固定商品向量、模型配置、训练参数、SID/向量/训练文件指纹，不需要网络下载。初始参数只是学习用默认值，没有在此数据集调优，也没有自动处理极端正负样本不平衡。

推理输出每条查询的 `recommendations`：`item_id`、`sid`、`sim_score`、`recall_rank`。通过 item JSON 的商品编号可查标题和描述。没有标签也能精排；有标签时同时计算原召回顺序和 SIM 顺序的商品级指标。

- `candidate_recall`：真实商品是否在展开后的候选集合内，精排前后不会改变。
- `HR@K`：所有查询中，目标进入前 K 的比例。
- `NDCG@K`：正确商品的位置越靠前，贡献越大。

空候选、没有召回目标的查询都计入分母。指标使用全部候选上的对应 K 截断，独立于输出 JSON 的 `top_k` 截断。SID 碰撞展开后的商品指标与原来的 SID 级指标不是同一口径，不应直接混比。

## 7. 测试和验证边界

```bash
python -m unittest minionerec.sim.tests tests.test_repository_layout -v
```

测试包括长历史早期相关行为检索、目标/未来历史隔离、重复窗口拒绝、SID 碰撞展开、空历史和单候选 batch、梯度有限性、冻结向量、标签不进入特征、候选未命中的指标分母、训练保存与无标签推理、数据指纹检查。

测试使用临时合成数据完成训练和推理，不把随机数据冒充真实 MiniOneRec 召回。尚未对真实大模型候选做完整训练与效果对比，不能宣称精排必然提升指标。完成上面的独立测试集实验后，再判断是否值得部署。
