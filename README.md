# BERT + 可解释性（SHAP / LIME / Attention）文本细粒度分类

本项目面向课程大作业/科研小项目选题：**小模型（BERT 类）结合多种可解释性方法的文本细粒度分类**。

- 不依赖大模型（LLM），算力友好（单卡 RTX 4060 级别即可）。
- 技术成熟：MacBERT（hfl/chinese-macbert-base）文本分类可快速收敛，易于获得较高准确率。
- 易出彩：通过 SHAP / LIME / Attention Pattern 三种方法生成可解释性可视化，报告素材丰富。

## 1. 任务与选题说明

### 1.1 任务定义
给定一段中文文本 \(x\)，预测其细粒度类别 \(y \in \{1,\dots,K\}\)。

可选任务：
- **中文新闻分类（多类别）**：如体育、娱乐、财经、科技等。
- **情感极性分类（二分类/多级）**：正面/负面（可扩展为多级评分）。

### 1.2 为什么可行且效果稳定
- MacBERT（`hfl/chinese-macbert-base`）在中文分类任务上表现优于原生 BERT，采用全词掩码 + 纠错预训练，语义理解更强。
- 使用 Hugging Face Transformers，训练/推理流程标准化。
- SHAP / Attention Pattern 对 Transformers 有现成接口，可直接对 token 贡献度可视化。

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
- 预训练模型：`hfl/chinese-macbert-base`（可替换 TinyBERT/ALBERT 等）
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

### 5.3 Attention Pattern（基于 Self-Attention）
- 直接可视化 Transformer 内部的注意力权重矩阵
- 展示每层每个 head 的 token-to-token 注意力分布
- 对比 [CLS] token 在不同层对输入 token 的关注变化

输出：
- 全局平均注意力热力图（所有层和 head 平均）
- 逐层注意力网格图
- 特定层的 per-head 注意力细节
- [CLS] token 的跨层注意力演化

报告可写：
- 注意力模式如何反映模型的推理过程
- 与 SHAP/LIME 的区别：注意力是模型内部的、前向的权重分布，SHAP/LIME 是后验的贡献度估计
- 三种方法的互补性：Attention 看"模型关注了什么"，SHAP 看"每个词贡献了多少"，LIME 看"局部决策边界"

### 5.4 可视化素材清单（建议最终报告必须包含）
- 训练曲线（loss/acc）
- 混淆矩阵
- 分类报告表格
- SHAP text plot（红蓝高亮）/ summary plot
- LIME explanation 示例
- Attention 热力图（全局平均 / 逐层 / 逐 head / CLS 跨层）

## 6. 拓展创新点（可选加分项）

- **多种可解释性方法对比**：SHAP vs LIME vs Attention Pattern 在本项目中已内置
- **细粒度类别**：更多类别、层级标签（粗->细）
- **模型对比**：MacBERT vs TinyBERT/DistilBERT/ALBERT
- **解释稳定性**：不同 seed、同义改写下解释一致性
- **误差分析**：对混淆最严重的类别做 SHAP/LIME/Attention 对比

## 7. 快速开始

### 7.1 环境安装

**方式一：一键脚本（推荐）**
```bash
bash scripts/setup_env.sh
```
自动创建 conda 环境 `bert-interp`（Python 3.10），安装 PyTorch CUDA 12.1 + 全部依赖。

**方式二：手动安装**
```bash
conda create -y -n bert-interp python=3.10
conda activate bert-interp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 7.2 配置镜像（国内用户）

模型加载默认优先走 ModelScope（阿里魔搭，国内无需代理），下载失败时自动回退到 HuggingFace。

```bash
# 可选：设置 HF 镜像作为后备
export HF_ENDPOINT=https://hf-mirror.com
```

### 7.3 准备数据集

**方式一：自动下载（国内网络可能不稳定）**
```bash
python scripts/prepare_thucnews.py
```
脚本使用 `wget` 断点续传从 hf-mirror 下载数据（约 2.2 GB），支持中断后续传。下载完成后自动按 80/10/10 划分为 train/valid/test JSONL。

采样选项：
```bash
# 默认：每类 5000 条训练样本（共 5 万条），4060 约 1.5 小时完成训练
python scripts/prepare_thucnews.py

# 快速验证：每类 2000 条
python scripts/prepare_thucnews.py --max_per_class 2000

# 完整数据集
python scripts/prepare_thucnews.py --full
```

**方式二：手动下载（推荐，更可靠）**

如果自动下载因网络原因失败，脚本会提示手动下载步骤：

1. 用浏览器或下载工具打开以下链接，下载 `THUCNews.jsonl`：
   ```
   https://hf-mirror.com/datasets/SirlyDreamer/THUCNews/resolve/main/THUCNews.jsonl
   ```
2. 将下载的文件放到 `data/thucnews/THUCNews.jsonl`
3. 运行脚本处理：
   ```bash
   python scripts/prepare_thucnews.py --local data/thucnews/THUCNews.jsonl
   ```

数据处理后存入 `data/thucnews/{train,valid,test}.jsonl` + `label2id.json`。

### 7.4 训练（训练完自动输出测试集指标）
```bash
python -m src.train --config configs/bert_thucnews.yaml
```
训练结束后自动在测试集上计算 Accuracy、Macro-F1 和每类 Precision/Recall/F1。
模型保存至 `outputs/bert_thucnews/`，预测结果保存至 `outputs/bert_thucnews/test_predictions.jsonl`。

### 7.5 生成混淆矩阵（详细评测）
```bash
python -m src.evaluate --config configs/bert_thucnews.yaml --ckpt outputs/bert_thucnews --assets_dir assets
```
输出：
- `assets/confusion_matrix_thucnews.png` — 归一化混淆矩阵热力图
- `assets/classification_report_thucnews.json` — 每类 Precision/Recall/F1

### 7.6 可解释性分析

确认测试集指标达标后，对具体样本进行可解释性分析：

```bash
CKPT=outputs/bert_thucnews
CONFIG=configs/bert_thucnews.yaml
TEXT="这是一条示例中文新闻文本"

# SHAP：token 级别 Shapley 贡献度（红蓝高亮）
python -m src.explain_shap --config $CONFIG --ckpt $CKPT --text "$TEXT"

# LIME：局部线性近似解释
python -m src.explain_lime --config $CONFIG --ckpt $CKPT --text "$TEXT" --num_samples 1000

# Attention：自注意力权重热力图
python -m src.explain_attention --config $CONFIG --ckpt $CKPT --text "$TEXT"

# 查看最后一层每个 head 的注意力细节
python -m src.explain_attention --config $CONFIG --ckpt $CKPT --text "$TEXT" --layer -1
```

产物汇总：
| 方法 | 产物 | 用途 |
|------|------|------|
| SHAP | `assets/shap/shap_explanation.html` | Token 贡献度红蓝高亮 |
| LIME | `assets/lime/lime_explanation.html` | 特征权重排序 |
| Attention | `assets/attention/attention_*.png` | 注意力热力图（4 张） |

### 7.7 创新拓展：注意力机制深度分析

在基础可解释性之上，本项目提供三个创新拓展方向，仅需少量代码即可产出具有学术深度的实验结果。

#### 创新一：词性引导的注意力裁剪（POS-Guided Attention Mask Pruning）

核心论点：模型将大量注意力浪费在 `[CLS]`、标点和虚词上。通过词性标注动态修改 `attention_mask`，将模型注意力强制引导至名词、动词等实义词，验证分类效果是否提升。

```bash
python -m src.explain_attention_prune \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "今天下午在北京国家会议中心举办科技创新大会，多位企业家和科学家出席演讲。"
```

输出：
- `assets/attention_prune/attention_prune_comparison.png` — 原始 vs 词性引导的注意力热力图对比
- `assets/attention_prune/attention_prune_cls_shift.png` — [CLS] token 注意力权重迁移柱状图
- 终端输出每个 token 的词性标注结果（蓝色加粗 = 内容词，普通 = 功能词）

#### 创新二：基于注意力权重的关键词抽取（Attention-based Keyword Extraction）

核心论点：最后一层 [CLS] token 对输入 token 的注意力权重，本身就是模型视角的"词汇重要性打分"。将其作为关键词抽取器，与 TF-IDF 基线对比。

```bash
python -m src.explain_attention_keywords \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "苹果公司今天发布了最新的iPhone 15系列手机，采用了全新的3纳米芯片技术。" \
  --top_k 5
```

输出：
- `assets/attention_keywords/attention_keywords_comparison.png` — Attention vs TF-IDF 关键词对比图
- `assets/attention_keywords/attention_keywords_result.json` — 结构化结果（含两方法关键词及得分）

#### 创新三：注意力头/层剪枝消融实验（Head & Layer Pruning Ablation）

核心论点：BERT 12 层 × 12 头 = 144 个注意力头存在巨大信息冗余。系统性地剪掉不同层、不同比例的头，测量模型准确率的退化曲线，揭示哪些组件是关键、哪些是冗余。

```bash
# 快速验证（50 条样本，约 2 分钟）
python -m src.explain_attention_ablation \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --max_samples 50

# 完整实验（建议 500-1000 条样本，约 15-30 分钟）
python -m src.explain_attention_ablation \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --max_samples 500
```

实验设计（自动完成所有对比组）：

| 实验组 | 操作 | 预期发现 |
|--------|------|---------|
| Baseline | 完整模型 | 准确率基线 |
| 按层组剪枝 | 依次剪掉 Bottom L1-L4 / Middle L5-L8 / Top L9-L12 全部 12 个头 | 底层是关键（断崖下跌），顶层可裁剪 |
| [CLS]-Heavy 剪枝 | 按 [CLS] 自注意力权重从高到低，剪掉 10%→50% 的头 | [CLS] 重头承载关键信息 |
| 随机剪枝 | 随机剪掉 10%→50% 的头（对照组） | 随机剪枝破坏性更小 |

输出：
- `assets/attention_ablation/ablation_layers.png` — 消融实验结果柱状图
- `assets/attention_ablation/ablation_results.json` — 完整实验数据

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
│   ├── explain_lime.py
│   ├── explain_attention.py
│   ├── explain_attention_prune.py    # 创新一：词性引导注意力裁剪
│   ├── explain_attention_keywords.py # 创新二：注意力关键词抽取
│   └── explain_attention_ablation.py # 创新三：头/层剪枝消融实验
├── assets/              # 输出图表（混淆矩阵、SHAP/LIME/Attention 可视化、消融曲线）
├── outputs/             # 模型权重与日志
└── requirements.txt
```

---

如果你是从课程/视频学习 BERT，可参考：BERT 机制、应用方式、以及模型可解释性（SHAP/Attention Pattern）等内容，在报告中作为理论背景与相关工作。
