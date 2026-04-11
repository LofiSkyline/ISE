import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

# 导入句向量提取模型
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 数据加载与预处理 (Data Loading & Preprocessing)
# ==========================================
def load_data(data_dir):
    \"\"\"
    加载 datasets 目录下所有的 csv 文件并合并。
    同时提取文本特征（Title + Body）和标签（class）。
    \"\"\"
    all_files = glob.glob(os.path.join(data_dir, \"*.csv\"))
    df_list = []
    
    for file in all_files:
        df = pd.read_csv(file)
        # 记录来源项目，方便后续如果需要做 Project-wise Breakdown
        df['Project'] = os.path.basename(file).replace('.csv', '') 
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 填补空值，防止向量化时报错
    full_df['Title'] = full_df['Title'].fillna('')
    full_df['Body'] = full_df['Body'].fillna('')
    
    # 特征工程：将 Title 和 Body 合并作为完整的自然语言输入
    full_df['Text'] = full_df['Title'] + \" . \" + full_df['Body']
    
    X_text = full_df['Text'].values
    y = full_df['class'].values
    
    return X_text, y, full_df

# ==========================================
# 2. 语义嵌入提取 (Semantic Embedding - Proposed Method)
# ==========================================
def get_semantic_embeddings(X_text, model_name='all-MiniLM-L6-v2'):
    \"\"\"
    使用 HuggingFace 的预训练轻量级模型提取语义向量。
    这是一个核心的性能提升点：将稀疏的词频转化为富含语义的稠密向量。
    \"\"\"
    print(f\"Loading Sentence-Transformer model: {model_name}...\")
    model = SentenceTransformer(model_name)
    
    print(\"Encoding texts to dense vectors (this might take a few minutes)...\")
    # show_progress_bar=True 可以直观看到提取进度
    X_embeddings = model.encode(X_text, show_progress_bar=True)
    
    return X_embeddings

# ==========================================
# 3. 评估辅助函数
# ==========================================
def evaluate_model(y_true, y_pred):
    \"\"\"计算 Precision, Recall 和 F1-Score\"\"\"
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f1

# ==========================================
# 4. 主实验流程 (包含讲义要求的 30 次重复)
# ==========================================
def run_experiments():
    # 假设脚本在 src/ 目录下运行，数据集在上一层的 datasets/ 中
    data_dir = \"../datasets\"
    
    print(\"--- Step 1: Loading Data ---\")
    X_text, y, _ = load_data(data_dir)
    print(f\"Total samples: {len(X_text)}, Positive ratio: {np.mean(y)*100:.2f}%\")
    
    print(\"\\n--- Step 2: Semantic Vectorization ---\")
    # 强烈建议在 30 次循环外进行 Embedding，否则会极大地浪费时间！
    X_embeddings = get_semantic_embeddings(X_text)
    
    print(\"\\n--- Step 3: Running 30 Repetitions ---\")
    n_repeats = 30
    
    # 记录结果的字典
    baseline_metrics = {'P': [], 'R': [], 'F1': []}
    proposed_metrics = {'P': [], 'R': [], 'F1': []}

    for seed in tqdm(range(n_repeats), desc=\"Experiment repeats\"):
        
        # 数据划分：70% 训练，30% 测试。
        # 极其重要：使用 stratify=y 保证每次划分的类别不平衡比例一致！
        X_text_train, X_text_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_text, y, range(len(y)), test_size=0.3, random_state=seed, stratify=y
        )
        
        # 对应获取 embedding 的拆分
        X_emb_train = X_embeddings[idx_train]
        X_emb_test  = X_embeddings[idx_test]
        
        # ----------------------------------------------------
        # 模型 1 (Baseline): TF-IDF + Naive Bayes (遵循讲义设定)
        # ----------------------------------------------------
        baseline_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        baseline_pipeline.fit(X_text_train, y_train)
        y_pred_base = baseline_pipeline.predict(X_text_test)
        
        p_base, r_base, f1_base = evaluate_model(y_test, y_pred_base)
        baseline_metrics['P'].append(p_base)
        baseline_metrics['R'].append(r_base)
        baseline_metrics['F1'].append(f1_base)
        
        # ----------------------------------------------------
        # 模型 2 (Proposed): Semantic Embeddings + Random Forest
        # ----------------------------------------------------
        # 注意: 这里的 class_weight='balanced' 是解决数据不平衡的核心，它会增加少数类的权重
        proposed_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1)
        proposed_clf.fit(X_emb_train, y_train)
        y_pred_prop = proposed_clf.predict(X_emb_test)
        
        p_prop, r_prop, f1_prop = evaluate_model(y_test, y_pred_prop)
        proposed_metrics['P'].append(p_prop)
        proposed_metrics['R'].append(r_prop)
        proposed_metrics['F1'].append(f1_prop)

    # ==========================================
    # 5. 结果聚合与统计检验 (Result Aggregation & Statistical Test)
    # ==========================================
    print(\"\\n--- Step 4: Final Results & Statistical Test ---\")
    
    print(\"[Baseline (TF-IDF + NB)]\")
    print(f\"Mean Precision: {np.mean(baseline_metrics['P']):.4f}, Mean Recall: {np.mean(baseline_metrics['R']):.4f}, Mean F1: {np.mean(baseline_metrics['F1']):.4f}\")
    
    print(\"\\n[Proposed (MiniLM + RF)]\")
    print(f\"Mean Precision: {np.mean(proposed_metrics['P']):.4f}, Mean Recall: {np.mean(proposed_metrics['R']):.4f}, Mean F1: {np.mean(proposed_metrics['F1']):.4f}\")
    
    # 使用 Wilcoxon Signed-Rank Test (非参数检验)，比较两个模型在 30 次实验中的 F1 表现
    stat, p_value = wilcoxon(proposed_metrics['F1'], baseline_metrics['F1'], alternative='greater')
    print(f\"\\nWilcoxon Signed-Rank Test p-value (F1): {p_value:.4e}\")
    
    if p_value < 0.05:
        print(\"=> SUCCESS: The proposed approach is significantly BETTER than the baseline on F1-score (p < 0.05)!\")
    else:
        print(\"=> NO SIGNIFICANT DIFFERENCE or Baseline is better.\")

if __name__ == '__main__':
    run_experiments()
