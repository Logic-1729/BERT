# 数据集获取与准备（THUCNews / ChnSentiCorp）

> 本仓库**不直接分发数据集**（版权/授权原因）。本文件提供下载来源建议与统一的本地数据格式要求，方便你用本项目脚本直接训练/评测/解释。

## 统一数据格式：JSONL

请将数据处理成 `*.jsonl`（一行一个样本），每行包含：

```json
{"text": "一段中文文本", "label": 0}
```

并放到如下目录（可在 `configs/*.yaml` 修改）：

- `data/thucnews/{train,valid,test}.jsonl`
- `data/chnsenticorp/{train,valid,test}.jsonl`

字段名默认：
- 文本字段：`text`
- 标签字段：`label`

## THUCNews（中文新闻分类）

常见做法：
1. 下载 THUCNews 数据集（可使用公开的课程镜像/第三方整理版，或自行检索“THUCNews 清华 新闻 分类 数据集”）。
2. 选择若干类别组成子集（例如 10 类），避免样本过大影响训练时间。
3. 将类别映射为整数 label（0..K-1），并输出 JSONL。

建议在预处理中额外产出：
- 类别分布统计（��状图）
- 文本长度分布（直方图）

## ChnSentiCorp（中文情感分类）

常见做法：
1. 获取 ChnSentiCorp 数据集（可在课程提供渠道或公开镜像中获得）。
2. 将“正/负”映射为 label=1/0，并输出 JSONL。

## 复现与检查清单

- [ ] 训练/验证/测试划分互不重叠
- [ ] 标签从 0 开始连续编码
- [ ] 文本为空/极短样本已清理
- [ ] 保存 `label2id.json` 与 `id2label.json`（可选，但建议）

