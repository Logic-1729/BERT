# 项目报告框架与实验清单

> 本文件为你的最终项目报告（或课程实验报告）提供一个参考框架与实验清单。根据你的具体要求自由调整章节与深度。

## 1. 摘要 (Abstract)

简述本项目的核心贡献：
- 用 BERT 在中文文本分类任务上进行微调
- 产出混淆矩阵、分类报告（Precision/Recall/F1）
- 用 SHAP 与 LIME 两种方法进行可解释性分析
- 对比两种解释方法的优劣

---

## 2. 引言与背景 (Introduction)

- **问题陈述**：为何需要 BERT 微调？为何需要可解释性？
- **相关工作**：BERT、Transformer 的基本原理（1-2段）
- **本项目的创新点或特色**：多任务对比？可解释性对比？

---

## 3. 方法 (Methods)

### 3.1 数据集与预处理
- THUCNews 与 ChnSentiCorp 的数据来源、规模、划分
- 数据预处理步骤（文本清理、分词、标签映射等）
- **截图或表格**：类别分布、文本长度分布

### 3.2 模型架构
- BERT 基本架构（可参考论文或简述）
- 微调策略（学习率、epoch、batch size 等）
- **表格**：`configs/*.yaml` 中的超参设置

### 3.3 评测指标
- Accuracy、Precision、Recall、F1（macro）
- 混淆矩阵的含义与获取方法

### 3.4 可解释性方法
- **SHAP**：基于 Shapley 值，token 级别的贡献度分析
- **LIME**：局部线性近似，扰动采样方法
- 两者的对比优劣

---

## 4. 实验结果 (Results)

### 4.1 训练与评测
- **表格/截图**：各数据集的最终 Accuracy、Precision、Recall、F1
- **混淆矩阵图**：
  - `assets/confusion_matrix_thucnews.png`
  - `assets/confusion_matrix_chnsenticorp.png`
- **分类报告 JSON**：
  - `assets/classification_report_thucnews.json`
  - `assets/classification_report_chnsenticorp.json`

### 4.2 可解释性分析
- **SHAP 可视化**：
  - 截图或嵌入 `assets/shap/shap_explanation.html`
  - 示例文本的 token 贡献度（红/蓝高亮）
- **LIME 可视化**：
  - 截图或嵌入 `assets/lime/lime_explanation.html`
  - 特征权重排序
- **对比分析**：SHAP 与 LIME 对同一样本的解释异同

---

## 5. 分析与讨论 (Discussion)

- 模型表现的强弱点？哪些类别容易混淆？
- 可解释性分析的启示：模型是否"看对了"关键词？
- SHAP 与 LIME 的方法论差异与实用性差异
- 可能的改进方向（数据增强、模型优化等）

---

## 6. 结论 (Conclusion)

总结本项目的关键发现与意义。

---

## 7. 参考文献 (References)

- BERT 论文
- SHAP 论文
- LIME 论文
- Transformer 综述
- 数据集来源等

---

## 实验执行清单

按以下步骤复现本项目的全部实验（假设已按 `docs/DATASETS.md` 准备好数据）：

### 步骤 1：环境与依赖
```bash
pip install -r requirements.txt
```

### 步骤 2：THUCNews 训练与评测

#### 2.1 训练
```bash
python -m src.train --config configs/bert_thucnews.yaml
```
输出：
- 训练日志到 `outputs/bert_thucnews/`
- 最佳模型与 tokenizer 保存到 `outputs/bert_thucnews/`
- 测试预测概率到 `outputs/bert_thucnews/test_predictions.jsonl`

#### 2.2 评测与生成混淆矩阵、分类报告
```bash
python -m src.evaluate \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --assets_dir assets
```
输出：
- `assets/confusion_matrix_thucnews.png`
- `assets/classification_report_thucnews.json`

#### 2.3 SHAP 解释示例
```bash
python -m src.explain_shap \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "一段示例中文文本" \
  --out_dir assets/shap
```
输出：
- `assets/shap/shap_explanation.html`（在浏览器中打开）
- 截图放入报告

#### 2.4 LIME 解释示例
```bash
python -m src.explain_lime \
  --config configs/bert_thucnews.yaml \
  --ckpt outputs/bert_thucnews \
  --text "一段示例中文文本" \
  --out_dir assets/lime \
  --num_samples 1000
```
输出：
- `assets/lime/lime_explanation.html`（在浏览器中打开）
- 截图放入报告

### 步骤 3：ChnSentiCorp 训练与评测

重复步骤 2 的流程，将 `bert_thucnews` 改为 `bert_chnsenticorp`。

### 步骤 4：可解释性对比分析

在相同的测试样本上分别运行 SHAP 和 LIME，对比它们的解释结果。

### 步骤 5：生成报告

按照"项目报告框架"章节，将上述所有产物（表格、图表、截图、JSON 报告）整合进最终报告文档。

---

## 产物清单（用于报告）

提交最终报告时，应包含以下文件/截图：

| 产物 | 文件 | 用途 |
|------|------|------|
| 混淆矩阵（THUCNews） | `assets/confusion_matrix_thucnews.png` | 结果部分 |
| 混淆矩阵（ChnSentiCorp） | `assets/confusion_matrix_chnsenticorp.png` | 结果部分 |
| 分类报告（THUCNews） | `assets/classification_report_thucnews.json` | 数据附录或结果部分 |
| 分类报告（ChnSentiCorp） | `assets/classification_report_chnsenticorp.json` | 数据附录或结果部分 |
| SHAP 可视化 | `assets/shap/shap_explanation.html` 截图 | 可解释性分析 |
| LIME 可视化 | `assets/lime/lime_explanation.html` 截图 | 可解释性分析 |
| 训练曲线（可选） | Hugging Face Trainer 的 eval loss 与 accuracy 曲线 | 训练细节 |
| 数据分布分析（可选） | 类别分布、文本长度分布 | 背景/方法部分 |

---

## 常见问题与建议

- **Q**：如何扩展到其他任务？
  - **A**：修改 `configs/` 中的任务参数（类别数、文本字段名、标签字段名）即可。
  
- **Q**：如何调整超参数？
  - **A**：编辑 `configs/*.yaml` 中的 `train:` 段（学习率、epoch、batch size 等）。

- **Q**：SHAP/LIME 很慢怎么办？
  - **A**：可减少 LIME 的 `--num_samples` 参数（默认 1000），或选择短文本测试。

- **Q**：报告里应该写多少内容？
  - **A**：根据你的课程要求调整。一般而言：
    - 本科课程设计：3-5 页
    - 硕士实验：10-15 页
    - 学位论文：30+ 页

---

**预祝你的项目报告圆满完成！** 如有任何问题，欢迎提 Issue 或 PR 改进本项目框架。

