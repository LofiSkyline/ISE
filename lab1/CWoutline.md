# Intelligent Software Engineering Coursework - Report Outline

**Project Type**: Tool Building Project (Option 1)
**Chosen Problem**: Lab 1 - Bug Report Classification
**Proposed Approach**: Lightweight Semantic Embeddings (Sentence-Transformers) + Advanced ML Classifiers (e.g., Random Forest/SVM), emphasizing cost-effectiveness and local deployment over API-based LLMs.

---

## 1. Introduction (引言)
*   **Problem Context**:
    *   软件工程领域中，快速识别并优先处理性能缺陷 (Performance Bugs) 至关重要。
    *   随着 GitHub 上 Bug Report 数量的爆炸式增长，手动分类是不现实的。
*   **The Baseline's Flaw**: 传统的基于词频的文本挖掘方法（如 TF-IDF）忽略了上下文和深层语义，在面对混杂着自然语言、术语和代码片段的 Bug Report 时表现挣扎。
*   **The Problem with "Overkill" Solutions**:
    *   *高分论点引入*：虽然大型语言模型（LLMs）具备强大的零样本分类能力，但在生产环境中，针对海量日常 Bug 报告调用庞大的 LLM API 存在**推理成本极高 (Cost-prohibitive)** 和 **数据隐私风险 (Data Privacy)**。
*   **Proposed Intelligent Solution**: 
    *   我们提出一种既智能又经济的方法：使用**轻量级本地预训练语义编码器**（如 `Sentence-Transformers`）捕获深层文本语义，结合**强大的传统分类器**（如 Random Forest），在保证极高准确率的同时，维持较低的计算开销。

## 2. Related Work (相关工作)
*   **Traditional Information Retrieval in SE**: 简述过去使用 TF-IDF 和朴素贝叶斯进行软件工程工件分析的文献，指出其局限性（稀疏表示、忽略语序）。
*   **Deep Learning for Text Classification**: 提及基于深度学习（CNN, RNN）的文本分类，但指出这些模型通常需要从头开始训练，且对数据量要求高。
*   **Pre-trained Language Models (PLMs)**: 讨论 BERT 时代的到来，以及如何通过句子级别的嵌入（Sentence Embeddings）在下游任务中取得 SOTA 表现。强调轻量级模型（如 MiniLM）在工业界的价值。

## 3. Solution (解决方案与设计理念)
*   *(这一节是拿高分的关键，重点解释 Design Rationales)*
*   **Phase 1: Semantic Vectorization (语义向量化)**
    *   放弃 TF-IDF，改用预训练的 `Sentence-Transformer` (例如 `all-MiniLM-L6-v2`)。
    *   *Rationale*: 解释为什么稠密向量 (Dense Vectors) 优于稀疏矩阵。它能理解 "slow execution" 和 "high latency" 在语义上是相似的，而 TF-IDF 会认为这是四个完全不同的词。且本地模型只需极小的内存即可运行。
*   **Phase 2: Handling Imbalanced Data & Classification (处理不平衡数据与分类)**
    *   *Rationale*: 引用 Table 1 的数据指出严重的数据不平衡（Positive 仅占 16.4%）。Naive Bayes 对此处理很差。
    *   我们选择的分类器（如 Random Forest 或 SVM）不仅更适合处理高维连续的向量特征，还可以通过设置类权重（`class_weight='balanced'`）来惩罚对少数类的误判，从而提升关键指标 Recall 和 F1-Score。

## 4. Experimental Setup (实验设置)
*   **Datasets**: 使用提供的 5 个深度学习项目数据集（TensorFlow, PyTorch 等，共 3712 条）。
*   **Procedure (严格遵循规范)**: 
    *   70% Train, 30% Test split.
    *   **Repeat 30 times** with random seeds to avoid stochastic bias.
*   **Baselines for Comparison**:
    *   Baseline: TF-IDF + Naive Bayes (Lab 提供的标准靶子)。
    *   Our Approach: Sentence-Transformers + Random Forest / SVM。
*   **Evaluation Metrics**: Precision, Recall, 和 F1-Score。强调 F1-Score 在处理不平衡数据时的重要性。
*   **Statistical Analysis**: 声明将使用例如 Wilcoxon signed-rank test 或 T-test 来验证结果的显著性 ($p < 0.05$)。

## 5. Experiments & Results (实验结果与分析)
*   *(预留放图表的位置)*
*   **Table/Figure 1: Overall Performance Comparison**: 展示我们的方法与 Baseline 在 30 次实验后的平均 Precision, Recall, F1 柱状图。
*   **Table 2: Project-wise Breakdown**: 展示在 5 个不同项目上的细分表现。
*   **Discussion (讨论)**:
    *   分析为什么我们的方法在 F1-Score（尤其是 Recall）上有显著提升（归功于语义理解和不平衡权重处理）。
    *   举一个具体例子（如果可能）：找一条不含明显“性能触发词”但被成功分类的短文本，证明语义向量的优越性。

## 6. Reflection (反思与限制)
*   **Limitations**:
    *   轻量级模型虽然比 TF-IDF 强，但其上下文窗口（如 512 tokens）可能无法完全覆盖极端超长的 Bug 报告。截断文本可能丢失关键报错堆栈信息。
    *   模型是在通用语料上预训练的，可能对某些极度特定的小众软件工程术语缺乏深刻理解。
*   **Future Work**:
    *   探索对轻量级模型进行领域适应 (Domain-adaptive Pretraining)微调。
    *   结合代码解析器专门处理 Bug Report 中的代码块。

## 7. Conclusion (结论)
*   总结实验发现，重申在真实的软件工程自动化中，寻找“效果与成本的平衡（Semantic Embeddings + ML）”比盲目追求巨型 API 更有价值，并证明了该方法成功击败了规定的 Baseline。

## 8. References (参考文献)
*   (将使用 IEEE/Chicago 格式在此处列出)