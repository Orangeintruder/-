# evaluate.py
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, confusion_matrix)

import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.preprocessor import DomainPreprocessor

# 设置中文字体（以防乱码，Windows下可用SimHei，Mac用Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)


def load_keras_model_and_preprocessor(model_name, level):
    """加载 Keras 模型和对应的预处理器"""
    # 加载模型
    model_path = f'models/checkpoints/{model_name}_{level}.h5'
    model = load_model(model_path)

    # 加载 tokenizer
    tokenizer_path = f'models/checkpoints/tokenizer_{level}.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # 重建预处理器
    max_len = 75 if level == 'char' else 20
    prep = DomainPreprocessor(max_len=max_len, level=level)
    prep.tokenizer = tokenizer
    prep.vocab_size = len(tokenizer.word_index) + 1

    return model, prep


def evaluate_all():
    # 读取测试集
    test_df = pd.read_csv('data/processed/test.csv')
    X_test_domains = test_df['domain'].values
    y_true = test_df['label'].values

    results = []

    # --- 深度学习模型评估 ---
    dl_configs = [
        ('cnn', 'char'),
        ('lstm', 'char'),
        ('cnn_lstm', 'char'),
        ('cnn', 'word'),
        ('lstm', 'word'),
        ('cnn_lstm', 'word')
    ]

    for model_name, level in dl_configs:
        try:
            print(f"评估深度学习模型: {model_name} ({level})")
            model, prep = load_keras_model_and_preprocessor(model_name, level)
            X_encoded = prep.encode(X_test_domains)
            y_pred_prob = model.predict(X_encoded, verbose=0).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            results.append({
                '模型': f'DL-{model_name.upper()}-{level}',
                '准确率': acc,
                '精确率': prec,
                '召回率': rec,
                'F1值': f1,
                '预测概率': y_pred_prob
            })
            print(f"  {model_name} ({level}): Acc={acc:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"  {model_name} ({level}) 评估失败: {e}")

    # --- 传统机器学习模型评估 ---
    ml_configs = [
        ('rf_baseline.pkl', 'Random Forest'),
        ('svm_baseline.pkl', 'SVM')
    ]
    from utils.feature_engineer import build_feature_matrix
    X_ml = build_feature_matrix(test_df['domain'])

    for filename, label in ml_configs:
        try:
            print(f"评估传统模型: {label}")
            model_path = f'models/checkpoints/{filename}'
            clf = joblib.load(model_path)
            y_pred = clf.predict(X_ml)
            y_pred_prob = clf.predict_proba(X_ml)[:, 1] if hasattr(clf, 'predict_proba') else None

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            results.append({
                '模型': f'ML-{label}',
                '准确率': acc,
                '精确率': prec,
                '召回率': rec,
                'F1值': f1,
                '预测概率': y_pred_prob
            })
            print(f"  {label}: Acc={acc:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"  {label} 评估失败: {e}")

    # --- 生成对比表格 ---
    df_results = pd.DataFrame(results)[['模型', '准确率', '精确率', '召回率', 'F1值']]
    df_results.to_csv('figures/all_models_results.csv', index=False, encoding='utf-8-sig')
    print("\n=== 模型性能对比总表 ===")
    print(df_results.to_string(index=False))

    # --- 绘制准确率/F1 对比柱状图 ---
    plot_metric_bar(df_results)

    # --- 绘制 ROC 曲线 ---
    plot_roc_curves(results, y_true)

    # --- 选择最佳模型绘制混淆矩阵 ---
    best = df_results.loc[df_results['F1值'].idxmax()]
    print(f"\n最佳模型: {best['模型']} (F1={best['F1值']:.4f})")
    # 找出 best 对应的结果行
    best_result = [r for r in results if r['模型'] == best['模型']][0]
    plot_confusion_matrix_best(best_result, y_true)

    print("\n所有图表已保存至 figures/ 文件夹。")


def plot_metric_bar(df):
    """准确率与 F1 柱状图"""
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width/2 for i in x], df['准确率'], width, label='准确率')
    plt.bar([i + width/2 for i in x], df['F1值'], width, label='F1值')
    plt.xticks(x, df['模型'], rotation=45, ha='right')
    plt.ylim(0.8, 1.0)
    plt.ylabel('分数')
    plt.title('模型性能对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/metrics_bar.png', dpi=300)
    plt.close()


def plot_roc_curves(results, y_true):
    """多模型 ROC 曲线"""
    plt.figure(figsize=(10, 8))
    for res in results:
        if res['预测概率'] is not None:
            fpr, tpr, _ = roc_curve(y_true, res['预测概率'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{res['模型']} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC 曲线')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('figures/roc_curves.png', dpi=300)
    plt.close()


def plot_confusion_matrix_best(best_result, y_true):
    """最佳模型的混淆矩阵"""
    if best_result['预测概率'] is None:
        return
    y_pred = (best_result['预测概率'] > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '恶意'], yticklabels=['正常', '恶意'])
    plt.title(f"{best_result['模型']} 混淆矩阵")
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    evaluate_all()