# Amazon18 原始数据

本目录通过 Git LFS 保存以下三份原始 JSON，合计约 1.08GB，未提交 `.gz` 压缩包。下载来源和文件大小见 `download_manifest.json`，处理结果见 [运行记录](../../docs/10_Amazon18预处理运行记录.md)。

| 文件 | 内容 |
| --- | --- |
| `Industrial_and_Scientific_5.json` | 工业用品 5-core 评论 |
| `Office_Products_5.json` | 办公用品 5-core 评论 |
| `meta_Industrial_and_Scientific.json` | 工业用品元数据 |

Office 元数据原文件约 5.23GB，按用户选择仅保留在本地，未提交。重新运行 Office 预处理时，需要另行[下载 Office 元数据](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz)，解压为 `data/raw/meta_Office_Products.json`。

安装 Git LFS 后，在项目根目录执行：

```bash
git lfs install
git lfs pull personal
```

如果你是重新克隆本仓库，远程通常名为 `origin`，将上述 `personal` 改成 `origin`。本机已有的原始文件保留不变，Office 元数据分片、压缩包和日志均被 Git 忽略。
