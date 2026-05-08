# train_ml.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.feature_engineer import build_feature_matrix

os.makedirs('models/checkpoints', exist_ok=True)

def main():
    # 加载数据
    print("加载数据集...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df   = pd.read_csv('data/processed/val.csv')   # 暂未直接用于传统模型，可合并训练
    test_df  = pd.read_csv('data/processed/test.csv')

    # 将训练和验证合并作为最终的训练集（传统模型通常不需要验证集调超参，可统一训练）
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"训练样本: {len(full_train_df)}, 测试样本: {len(test_df)}")

    #提取手工特征
    print("提取手工特征...")
    X_train = build_feature_matrix(full_train_df['domain'])
    y_train = full_train_df['label']
    X_test  = build_feature_matrix(test_df['domain'])
    y_test  = test_df['label']

    print(f"特征维度: {X_train.shape[1]}")

    #训练随机森林
    print("\n训练随机森林...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    print(f"随机森林 - Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")

    #训练 SVM
    print("\n训练 SVM...")
    clf_svm = SVC(kernel='rbf', probability=True, random_state=42)
    clf_svm.fit(X_train, y_train)
    y_pred_svm = clf_svm.predict(X_test)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm)
    rec_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)

    print(f"SVM - Accuracy: {acc_svm:.4f}, Precision: {prec_svm:.4f}, Recall: {rec_svm:.4f}, F1: {f1_svm:.4f}")

    # 保存模型
    joblib.dump(clf_rf, 'models/checkpoints/rf_baseline.pkl')
    joblib.dump(clf_svm, 'models/checkpoints/svm_baseline.pkl')
    print("\n模型已保存至 models/checkpoints/rf_baseline.pkl 和 svm_baseline.pkl")

    #保存评估指标到 CSV（便于后续汇总）
    results = pd.DataFrame({
        'Model': ['Random Forest', 'SVM'],
        'Accuracy': [acc_rf, acc_svm],
        'Precision': [prec_rf, prec_svm],
        'Recall': [rec_rf, rec_svm],
        'F1': [f1_rf, f1_svm]
    })
    results.to_csv('models/checkpoints/traditional_ml_results.csv', index=False)
    print("评估结果已保存至 models/checkpoints/traditional_ml_results.csv")

if __name__ == '__main__':
    main()