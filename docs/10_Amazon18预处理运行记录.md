# 10 Amazon18 预处理运行记录

运行日期：2026-09-19。已在本机实际下载、解压并执行两个类别的 `minionerec/preprocessing/amazon18.py`，不是仅生成运行命令。

## 输入数据

使用 Amazon Reviews 2018 官方商品元数据及 5-core 评论子集。5-core 已经经过官方交互次数筛选，不是该类别的完整评论集。

| 类别 | 文件 | 压缩大小（十进制 MB） | 解压大小（十进制 MB） |
| --- | --- | ---: | ---: |
| Industrial | `meta_Industrial_and_Scientific.json.gz` | 83.75 | 637.36 |
| Industrial | `Industrial_and_Scientific_5.json.gz` | 11.00 | 38.88 |
| Office | `meta_Office_Products.json.gz` | 514.24 | 5228.77 |
| Office | `Office_Products_5.json.gz` | 111.79 | 407.59 |

来源：[数据集主页](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)、[Industrial 元数据](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Industrial_and_Scientific.json.gz)、[Industrial 评论](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific_5.json.gz)、[Office 元数据](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz)、[Office 评论](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz)。

文件在本地 `data/raw/`，同时保留压缩包、解压文件、下载清单和日志。四个压缩包的大小均与服务器 Content-Length 一致，完整解压通过 gzip 校验。

## 实际运行结果

统一初始参数：`user_k=5`、`item_k=5`，时间范围 `2016-10` 至 `2018-11`。代码实际用 `user_k` 同时过滤用户和商品；结束边界是 2018 年 11 月 1 日，而不是 11 月底。

| 类别 | 实际起止年月 | 用户 | 商品 | 保留交互 | 训练样本 | 验证样本 | 测试样本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Industrial | 2013-10 → 2018-11 | 6,300 | 3,106 | 43,123 | 29,458 | 3,682 | 3,683 |
| Office | 2016-10 → 2018-11 | 8,328 | 3,459 | 56,984 | 38,924 | 4,866 | 4,866 |

Industrial 起始时间自动回退到 2013 年，是因为脚本会在商品不足 3,000 个时逐年扩大时间范围。每个用户的第一条交互没有历史，因此总推荐样本数等于保留交互数减用户数。

`.review.json` 中 Industrial 有 40,468 条，Office 有 52,712 条。它以用户、商品和时间的组合构造字典键，重复键会覆盖，因此条目数不等于交互数；构造推荐历史时仍保留重复交互。

## 去哪里查看结果

```text
data/
├── raw/                              原始压缩包、解压文件、日志
└── Amazon18/
    ├── Industrial_and_Scientific/    本次工业用品处理结果
    ├── Office_Products/              本次办公用品处理结果
    ├── _verification/               Industrial 内存优化回归结果
    └── run_summary.json              机器可读统计结果
```

每个类别目录有 8 个文件：`.item.json`、`.inter.json`、`.review.json`、`.train.inter`、`.valid.inter`、`.test.inter`、`.user2id`、`.item2id`。先看 `.item.json` 中的商品，再看 `.train.inter` 的历史与目标，最后用 `.item2id` 对照原始 ASIN。

这是清洗与样本构造的结果，还没有生成商品向量或 SID。本次 Industrial 数量不同于仓库原有数据，不能将这些新编号直接配上 `data/Amazon/index/` 的旧 SID。应在同一套新编号上继续生成向量与 SID，再转换 CSV；即使商品数量相同，也应核对编号映射。

现有 `data/Amazon/` 文件未改动。按用户选择，仅将两份解压后的 5-core 评论与 Industrial 元数据通过 Git LFS 提交，合计约 1.08GB；Office 元数据约 5.23GB，仅保留在本地。详见 [原始数据说明](../data/raw/README.md)。压缩包、日志、生成的预处理结果和本地虚拟环境仍忽略，不上传。

## 运行命令

本次使用独立环境 `.venv-preprocess/`，安装 `tqdm`。脚本中未使用的 Torch、NumPy 导入已移除，因此预处理本身不需要安装训练框架或使用 GPU。

在项目根目录的 PowerShell 中运行；显式启用 UTF-8，避免 Windows 默认编码影响原始英文及 Unicode 文本：

```powershell
foreach ($category in @("Industrial_and_Scientific", "Office_Products")) {
    .\.venv-preprocess\Scripts\python.exe -X utf8 -m minionerec.preprocessing.amazon18 `
        --dataset $category `
        --metadata_file "./data/raw/meta_${category}.json" `
        --reviews_file "./data/raw/${category}_5.json" `
        --user_k 5 --item_k 5 `
        --st_year 2016 --st_month 10 --ed_year 2018 --ed_month 11 `
        --output_path ./data/Amazon18
}
```

重复运行会重写对应类别的输出文件。第一次运行前需创建虚拟环境并安装 `tqdm`，原始数据按上面的下载链接解压到 `data/raw/`。

## 内存优化与验证

Office 商品元数据解压后约 5.23GB。原先全部解析并保存在列表中，会占用大量内存；本次将读取改为逐行解析，并通过可选 `item_asins` 参数仅保留本次评论出现过的商品。原始文件没有裁剪，过滤规则与后续样本构造保持不变。独立调用读取函数时，不传这个参数仍会读取全部商品。

先用原逻辑完成 Industrial，再用优化后的逻辑重新运行，8 个输出文件逐字节一致。两类输出均检查了商品和用户编号引用、1～10 条历史长度、总样本数量及 80%/10%/10% 划分。日志保存在 `data/raw/Industrial_and_Scientific.preprocess.log` 和 `data/raw/Office_Products.preprocess.log`。
