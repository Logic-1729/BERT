# BERT + SHAP/LIME 文本细粒度分类（项目框架）

本项目面向课程大作业/科研小项目选题：**小模型（BERT 类）结合可解释性工具（SHAP/LIME）的文本细粒度分类**。

- 不依赖大模型（LLM），算力友好（单卡 RTX 4060 级别即可）。
- 技术成熟：BERT 文本分类可快速收敛，易于获得较高准确率。
- 易出彩：通过 SHAP/LIME 生成可解释性可视化（高亮/条形图），报告素材丰富。

## 1. 任务与选题说明

### 1.1 任务定义
给定一段中文文本 \(x\)，预测其细粒度类别 \(y \in \{1,\dots,K\}\)。

可选任务：
- **中文新闻分类（多类别）**：如体育、娱乐、财经、科技等。
- **情感极性分类（二分类/多级）**：正面/负面（可扩展为多级评分）。

### 1.2 为什么可行且效果稳定
- BERT（如 `bert-base-chinese`）在中文分类任务上方案成熟。
- 使用 Hugging Face Transformers，训练/推理流程标准化。
- SHAP 对 Transformers 有现成接口，可直接对 token 贡献度可视化。

## 2. 数据集

### 2.1 THUCNews（中文新闻分类）
- 用于多类别新闻分类。
- 建议：选择其子集（若算力/时间受限）。

### 2.2 ChnSentiCorp（中文情感分类）
- 中文酒店/商品评论情感分析数据集。

### 2.3 数据处理建议
- 划分：train/valid/test（如 8/1/1）
- 统计：类别分布柱状图、样本长度分布
- 预处理：去重、去空、截断/最大长度设置

> 数据集版权与下载方式各有不同，建议在 `docs/DATASETS.md` 中提供获取说明与处理脚本（本仓库不直接分发数据）。

## 3. 模型与方法

### 3.1 Baseline（对照组）
- TF-IDF + Logistic Regression / Linear SVM

### 3.2 主模型：BERT 文本分类
- 预训练模型：`bert-base-chinese`（可替换 TinyBERT/ALBERT 等）
- 分类头：`[CLS]` 表征 + Linear

### 3.3 训练细节（建议写入报告）
- batch size、学习率、epoch、warmup、weight decay
- early stopping
- 固定随机种子（可复现）

## 4. 无可争议的定量评测

- Accuracy、Macro-F1
- 每类 Precision/Recall/F1（`sklearn.metrics.classification_report`）
- 混淆矩阵（Confusion Matrix）可视化（seaborn heatmap）

## 5. 可解释性分析（重点）

### 5.1 SHAP（SHapley Additive exPlanations）
输出：文本 token 对预测类别的贡献度可视化。

报告可写：
- 为什么需要解释：提高可用性/可信度/排错能力
- 解释对象：token（WordPiece）级别贡献度
- 注意 tokenization：如何将子词合并回可读文本

### 5.2 LIME（对照）
- 作为局部可解释性对照方法
- 与 SHAP 的解释差异：稳定性、可重复性、粒度

### 5.3 可视化素材清单（建议最终报告必须包含）
- 训练曲线（loss/acc）
- 混淆矩阵
- 分类报告表格
- SHAP text plot（红蓝高亮）/ summary plot
- LIME explanation 示例

## 6. 拓展创新点（可选加分项）

- **细粒度类别**：更多类别、层级标签（粗->细）
- **模型对比**：BERT vs TinyBERT/DistilBERT/ALBERT
- **解释稳定性**：不同 seed、同义改写下解释一致性
- **误差分析**：对混淆最严重的类别做 SHAP/LIME 对比

## 7. 快速开始

### 7.1 环境安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 7.2 训练（示例）
```bash
python -m src.train --config configs/bert_thucnews.yaml
```

### 7.3 评测并生成混淆矩阵
```bash
python -m src.evaluate --config configs/bert_thucnews.yaml --ckpt outputs/best
```

### 7.4 生成 SHAP 可视化
```bash
python -m src.explain_shap --config configs/bert_thucnews.yaml --ckpt outputs/best --text "差评，态度恶劣，体验很差"
```

### 7.5 生成 LIME 可视化
```bash
python -m src.explain_lime --config configs/bert_thucnews.yaml --ckpt outputs/best --text "差评，态度恶劣，体验很差"
```

## 8. 目录结构（计划）

```
.
├── configs/
├── docs/
├── src/
│   ├── data/
│   ├── utils/
│   ├── train.py
│   ├── evaluate.py
│   ├── explain_shap.py
│   └── explain_lime.py
├── assets/              # 输出图表（混淆矩阵、SHAP/LIME 可视化）
├── outputs/             # 模型权重与日志
└── requirements.txt
```

---

如果你是从课程/视频学习 BERT，可参考：BERT 机制、应用方式、以及模型可解释性（SHAP/Attention Pattern）等内容，在报告中作为理论背景与相关工作。
