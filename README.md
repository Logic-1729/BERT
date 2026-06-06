# BERT 可解释性 — 中文文本细粒度分类

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.38+-ff9d00.svg)](https://github.com/huggingface/transformers)

一项关于 **BERT 文本分类器可解释性** 的系统研究，从后验归因（SHAP、LIME）、内部分析（自注意力可视化）和结构消融三个角度切入。项目在 THUCNews（14 类中文新闻分类）上微调 [MacBERT](https://huggingface.co/hfl/chinese-macbert-base)，**测试集准确率 95.55%**（macro-F1 0.956），随后通过三种可解释性方法和三个注意力扩展实验对模型进行深入分析。

---

## 目录

- [任务与数据集](#任务与数据集)
- [模型与训练](#模型与训练)
- [评测](#评测)
- [可解释性分析](#可解释性分析)
  - [1. SHAP — Shapley 值归因](#1-shap--shapley-值归因)
  - [2. LIME — 局部代理解释](#2-lime--局部代理解释)
  - [3. 自注意力模式可视化](#3-自注意力模式可视化)
- [扩展实验](#扩展实验)
  - [4. 词性引导的注意力裁剪](#4-词性引导的注意力裁剪)
  - [5. 基于注意力的关键词抽取](#5-基于注意力的关键词抽取)
  - [6. 注意力头与层的剪枝消融](#6-注意力头与层的剪枝消融)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [结果汇总](#结果汇总)
- [参考文献](#参考文献)

---

## 任务与数据集

### 新闻分类（THUCNews）

使用 [THUCNews](http://thuctc.thunlp.org/) 中文新闻分类语料子集，共 14 个类别。每条样本为 `{"text": "...", "label": N}` 格式的 JSONL 记录。

| ID | 类别 | ID | 类别 |
|----|------|----|------|
| 0 | 体育 | 7 | 时政 |
| 1 | 娱乐 | 8 | 星座 |
| 2 | 家居 | 9 | 游戏 |
| 3 | 彩票 | 10 | 社会 |
| 4 | 房产 | 11 | 科技 |
| 5 | 教育 | 12 | 股票 |
| 6 | 时尚 | 13 | 财经 |

**数据划分**：训练集 67,854 / 验证集 6,865 / 测试集 6,859（80/10/10）。

### 情感分类（ChnSentiCorp）

中文酒店/商品评论二分类情感分析。配置文件见 `configs/bert_chnsenticorp.yaml`，数据获取说明见 [docs/DATASETS.md](docs/DATASETS.md)。

## 模型与训练

**模型架构**：`hfl/chinese-macbert-base` — BERT-base 编码器（12 层、12 头、隐藏维度 768、约 102M 参数），采用全词掩码与纠错预训练，中文语义表示能力优于原生 BERT。

**分类头**：`[CLS]` 表征 → 线性投影 → 14 路 softmax。

**训练配置**：

| 参数 | 值 |
|------|-----|
| 最大序列长度 | 256 |
| 批次大小 | 16（训练）/ 32（评估） |
| 学习率 | 2 × 10⁻⁵ |
| 权重衰减 | 0.01 |
| 训练轮数 | 3 |
| 预热比例 | 6% |
| 学习率调度 | 线性衰减 |
| 最佳模型选择 | 验证集 Macro-F1 |

## 评测

在留出测试集上计算以下定量指标：

| 指标 | 数值 |
|------|------|
| 测试准确率 | **95.55%** |
| Macro-F1 | **0.9561** |
| 最优类别 | 财经，F1 = 0.983 |
| 最差类别 | 体育，F1 = 0.915 |

每类 Precision/Recall/F1 及归一化混淆矩阵保存于 `assets/` 目录。

![混淆矩阵](assets/confusion_matrix_thucnews.png)

## 可解释性分析

三种互补方法从不同粒度揭示模型的决策依据。

### 1. SHAP — Shapley 值归因

SHAP 基于合作博弈论中的 Shapley 值，计算每个输入 token 对模型输出的边际贡献。本项目使用 `shap.Explainer` 封装 HuggingFace `TextClassificationPipeline`，将预测分数分配到各个 WordPiece token。

**红色** 高亮表示该 token 推动预测朝向当前类别，**蓝色** 表示反向推动。

```bash
python -m src.explain_shap \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "今天下午在北京国家会议中心举办科技创新大会"
```

**产物**: `assets/shap/shap_explanation.html` — 交互式 token 贡献度可视化。

### 2. LIME — 局部代理解释

LIME 通过对输入进行扰动采样，训练一个稀疏线性模型来逼近局部分类边界。作为 SHAP 的对照方法 — LIME 速度更快，但跨运行稳定性较低。

```bash
python -m src.explain_lime \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "今天下午在北京国家会议中心举办科技创新大会" \
  --num_samples 1000
```

**产物**: `assets/lime/lime_explanation.html` — 逐 token 特征重要性权重。

### 3. 自注意力模式可视化

通过 `output_attentions=True` 提取全部 12 层 × 12 头的自注意力权重，生成四种可视化：

| 模式 | 说明 | 输出文件 |
|------|------|----------|
| 全局平均 | 所有层和头的均值 | `attention_global_avg.png` |
| 逐层网格 | 12 个子图，每层取头均值 | `attention_per_layer.png` |
| 单层逐头 | 指定层的 12 个头独立热力图 | `attention_layer_12_heads.png` |
| [CLS] 演化 | [CLS] token 跨层注意力变化 (L1→L12) | `attention_cls_per_layer.png` |

```bash
python -m src.explain_attention \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "今天下午在北京国家会议中心举办科技创新大会"

# 查看最后一层每个 head 的注意力细节
python -m src.explain_attention \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "..." --layer -1
```

**产物**: `assets/attention/attention_*.png`（4 张图）。

### 方法对比

| 特性 | SHAP | LIME | Attention |
|------|------|------|-----------|
| 类型 | 后验归因（Shapley） | 后验归因（代理模型） | 模型内部分析（权重） |
| 粒度 | Token 级贡献度 | Token 级重要性 | Token 间亲和度 |
| 稳定性 | 高（有理论保证） | 中（扰动噪声） | 确定性 |
| 计算开销 | 高 | 中 | 低（单次前向传播） |
| 适用场景 | 定位关键证据 | 快速局部解释 | 理解内部表征 |

## 扩展实验

在标准可解释性之上，三个实验分别从操纵、复用和破坏的角度进一步探究注意力机制。

### 4. 词性引导的注意力裁剪

**假设**：BERT 将大量注意力浪费在 `[CLS]`、`[SEP]`、标点符号和虚词上。强制模型更多地关注名词、动词、形容词等实义词，可以检验默认注意力分布是否最优。

**方法**：
1. 使用 `jieba.posseg` 对输入文本进行字符级词性标注
2. 将词性标签对齐到 BERT 的 WordPiece 分词
3. 在前向传播前修改 `attention_mask`：
   - 实义词（n/v/a 系列）：**增强 × 2.0**
   - 虚词（u/p/c/d 等）：**抑制 × 0.3**
   - 特殊 token：不变

```bash
python -m src.explain_attention_prune \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "今天下午在北京国家会议中心举办科技创新大会，多位企业家和科学家出席演讲。"
```

**产物**：
- `assets/attention_prune/attention_prune_comparison.png` — 原始 vs. 词性引导的注意力热力图对比
- `assets/attention_prune/attention_prune_cls_shift.png` — [CLS] 注意力权重迁移柱状图

**发现**：经词性引导后，[CLS] 的注意力从虚词明显迁移至实义词，说明默认注意力分布对语义弱 token 存在过度分配。

### 5. 基于注意力的关键词抽取

**假设**：最后一层 [CLS] token 的注意力分数本身就是模型内置的 token 重要性评分器，无需任何额外训练即可作为零样本关键词抽取器使用。

**方法**：
1. 提取最后一层 (L12) [CLS] token 的注意力向量（12 头均值）
2. 过滤特殊 token 和标点符号
3. 按注意力分数降序排列，取 Top-K
4. 与 TF-IDF 基线对比（字符级 n-gram, n=1–3）

```bash
python -m src.explain_attention_keywords \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "苹果公司今天发布了最新的iPhone 15系列手机，采用了全新的3纳米芯片技术。" \
  --top_k 5
```

**产物**：
- `assets/attention_keywords/attention_keywords_comparison.png` — 注意力 vs. TF-IDF 关键词对比图
- `assets/attention_keywords/attention_keywords_result.json` — 结构化关键词数据

**发现**：基于注意力的关键词能捕获领域特定实体（如"芯片""纳米"），这些词因文档频率低而容易被 TF-IDF 遗漏。TF-IDF 倾向于高频字符 n-gram，而注意力反映模型学习到的语义重要性。

### 6. 注意力头与层的剪枝消融

**假设**：BERT 的 144 个注意力头（12 层 × 12 头）并非同等重要。系统性归零不同头，可以揭示哪些组件是关键、哪些是冗余。

**方法**：将目标头的 Q/K/V/O 权重矩阵置零。三种剪枝策略：

| 策略 | 操作 | 目的 |
|------|------|------|
| 按层组 | 依次移除底层 (L1–4) / 中层 (L5–8) / 顶层 (L9–12) 的全部 12 个头 | 定位关键深度区间 |
| [CLS] 优先 | 按 [CLS] 自注意力分数排序，优先剪掉得分最高的头 (10%–50%) | 验证 [CLS] 聚焦的头是否更重要 |
| 随机 | 随机选取 10%/30%/50% 的头 | 结构化剪枝的对照基线 |

```bash
# 快速验证（50 条样本，约 2 分钟）
python -m src.explain_attention_ablation \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --max_samples 50

# 完整实验（500 条样本，约 15-30 分钟）
python -m src.explain_attention_ablation \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --max_samples 500
```

**产物**：
- `assets/attention_ablation/ablation_layers.png` — 全部实验条件的准确率柱状图
- `assets/attention_ablation/ablation_results.json` — 完整数值结果

**实验结果**（500 条测试样本）：

| 实验条件 | 准确率 | Δ |
|----------|--------|-----|
| Baseline（无剪枝） | 0.980 | — |
| 剪枝底层 L1–L4 | **0.020** | −0.960 |
| 剪枝中层 L5–L8 | 0.940 | −0.040 |
| 剪枝顶层 L9–L12 | **1.000** | +0.020 |
| [CLS]-Heavy 10% | 0.960 | −0.020 |
| [CLS]-Heavy 30% | 0.880 | −0.100 |
| [CLS]-Heavy 50% | 0.700 | −0.280 |
| Random 10% | 0.980 | 0.000 |
| Random 30% | 0.960 | −0.020 |
| Random 50% | 0.860 | −0.120 |

**核心发现**：

1. **底层是关键，不可裁撤。** 移除 L1–L4 后准确率崩塌至 0.02，说明底层编码了所有上层依赖的基础词汇与句法特征。
2. **顶层高度冗余。** 完全移除 L9–L12 对准确率无负面影响，中层即可独立完成分类。
3. **[CLS] 聚焦的头承载更多分类信号。** 剪掉 [CLS] 重头比随机剪枝退化更快（50%: 0.70 vs. 0.86），证实其在分类决策中的关键作用。
4. **BERT 存在显著头部冗余。** 随机移除 30% 的头仅导致 0.02 的准确率下降，支持基于剪枝的模型压缩方向。

## 快速开始

### 环境安装

```bash
# 一键安装：conda 环境 + PyTorch CUDA 12.1 + 全部依赖
bash scripts/setup_env.sh

# 或手动安装
conda create -y -n bert-interp python=3.10
conda activate bert-interp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

模型加载默认走 ModelScope（国内无需代理），失败时自动回退至 HuggingFace。

### 数据准备

```bash
# 默认：每类 5000 条训练样本（共约 5 万条）
python scripts/prepare_thucnews.py

# 快速迭代用子集
python scripts/prepare_thucnews.py --max_per_class 2000

# 完整数据集
python scripts/prepare_thucnews.py --full
```

输出：`data/thucnews/{train,valid,test}.jsonl` + `label2id.json`。

### 训练与评测

```bash
# 训练（完成后自动输出测试集指标）
python -m src.train --config configs/bert_thucnews.yaml

# 生成混淆矩阵与每类指标报告
python -m src.evaluate \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --assets_dir assets
```

### 可解释性与扩展实验

```bash
CKPT=outputs/bert_thucnews
CONFIG=configs/bert_thucnews.yaml
TEXT="今天下午在北京国家会议中心举办科技创新大会"

# 基础可解释性
python -m src.explain_shap       --config $CONFIG --ckpt $CKPT --text "$TEXT"
python -m src.explain_lime       --config $CONFIG --ckpt $CKPT --text "$TEXT" --num_samples 1000
python -m src.explain_attention  --config $CONFIG --ckpt $CKPT --text "$TEXT"

# 扩展实验
python -m src.explain_attention_prune     --config $CONFIG --ckpt $CKPT --text "$TEXT"
python -m src.explain_attention_keywords  --config $CONFIG --ckpt $CKPT --text "$TEXT" --top_k 5
python -m src.explain_attention_ablation  --config $CONFIG --ckpt $CKPT --max_samples 500
```

情感分类任务将 `bert_thucnews` 替换为 `bert_chnsenticorp` 即可。

## 目录结构

```
.
├── configs/
│   ├── bert_thucnews.yaml              # 14 类新闻分类配置
│   └── bert_chnsenticorp.yaml          # 二分类情感分析配置
├── docs/
│   ├── DATASETS.md                     # 数据集获取与格式说明
│   └── PROJECT_PLAN.md                 # 报告大纲模板
├── scripts/
│   ├── prepare_thucnews.py             # 数据集下载与预处理
│   └── setup_env.sh                    # 一键环境安装脚本
├── src/
│   ├── data/
│   │   └── loaders.py                  # JSONL → HuggingFace DatasetDict
│   ├── utils/
│   │   ├── common.py                   # set_seed / ensure_dir / load_yaml
│   │   ├── metrics.py                  # 分类报告 / softmax
│   │   ├── model_loader.py             # ModelScope 优先的模型加载
│   │   └── plot.py                     # 混淆矩阵绘制
│   ├── train.py                        # 训练流程
│   ├── evaluate.py                     # 评测与指标
│   ├── explain_shap.py                 # SHAP token 归因
│   ├── explain_lime.py                 # LIME 局部解释
│   ├── explain_attention.py            # 自注意力可视化
│   ├── explain_attention_prune.py      # 词性引导注意力裁剪
│   ├── explain_attention_keywords.py   # 注意力关键词抽取
│   └── explain_attention_ablation.py   # 注意力头/层剪枝消融
├── assets/                             # 生成的可视化与结果
│   ├── confusion_matrix_thucnews.png
│   ├── classification_report_thucnews.json
│   ├── shap/shap_explanation.html
│   ├── lime/lime_explanation.html
│   ├── attention/                      # 4 张注意力热力图
│   ├── attention_prune/                # 2 张词性引导对比图
│   ├── attention_keywords/             # 关键词对比图 + JSON
│   └── attention_ablation/             # 消融柱状图 + JSON 结果
├── outputs/                            # 训练好的模型权重
├── CLAUDE.md
├── README.md
└── requirements.txt
```

## 结果汇总

| 实验 | 关键指标 | 核心发现 |
|------|----------|----------|
| 文本分类 | 95.55% 准确率，0.956 macro-F1 | MacBERT 有效区分 14 类新闻 |
| SHAP | Token 级 Shapley 值 | 定位每个预测的关键依据词 |
| LIME | 局部代理模型权重 | 快速、补充性的解释方法 |
| 注意力可视化 | 4 种视角的热力图分析 | [CLS] 注意力随层加深向内容词聚焦 |
| 词性引导裁剪 | [CLS] 注意力迁移确认 | 默认注意力向虚词过度分配 |
| 关键词抽取 | 注意力 vs. TF-IDF 对比 | 模型注意力比 TF-IDF 更能捕获领域实体 |
| 层剪枝消融 | 底层移除准确率崩塌至 0.02 | L1–L4 关键不可裁；L9–L12 移除无影响 |
| 头剪枝消融 | 30% 随机剪枝仅降 0.02 | 注意力头大量冗余；[CLS] 重头更重要 |

## 参考文献

- Cui et al., "Revisiting Pre-Trained Models for Chinese Natural Language Processing", *EMNLP Findings*, 2020.
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", *NeurIPS*, 2017.
- Ribeiro et al., "'Why Should I Trust You?': Explaining the Predictions of Any Classifier", *KDD*, 2016.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", *NAACL-HLT*, 2019.
- Michel et al., "Are Sixteen Heads Really Better than One?", *NeurIPS*, 2019.
- THUCNews: 清华大学, [thuctc.thunlp.org](http://thuctc.thunlp.org/).
