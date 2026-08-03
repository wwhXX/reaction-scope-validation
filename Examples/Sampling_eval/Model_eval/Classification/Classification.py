import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def cvt_sampling_df(data, k, not_feature_columns, max_iters=500, tol=1e-4):
    """
    CVT采样算法（支持DataFrame输入，保留非数值列）

    参数:
        data: DataFrame，包含特征列和非数值列
        k: 需要采样的中心点数量
        not_feature_columns: list，非特征列名
        max_iters: 最大迭代次数
        tol: 中心点变化的收敛阈值

    返回:
        centers: DataFrame，采样到的中心点
        unselected_points: DataFrame，未被采样的点
    """
    X = data.drop(not_feature_columns, axis=1).values

    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()

    for _ in range(max_iters):
        distances = pairwise_distances(X, centers_X)
        labels = np.argmin(distances, axis=1)

        new_centers_X = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT算法收敛, 迭代次数: {_}")
            break
        centers_X = new_centers_X

    final_distances = pairwise_distances(X, centers_X)
    selected_indices = np.argmin(final_distances, axis=0)

    centers = data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), selected_indices)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def evaluate_performance(model, X, y, set_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\nPerformance on {set_name} set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return acc, recall, f1

# Optuna目标函数（使用验证集）
def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 6, 50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': 23
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)
    return f1

def plot_confusion_matrix(y_true, y_pred, set_name, train_size, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix ({set_name} Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'./xgb_classify/confusion_matrix_{set_name}_binary_70_{train_size}train.png')
    plt.show()


def prepare_labels(data, y_column):
    """将标签转换为二分类"""
    y = pd.DataFrame([float(y.split('%')[0])/100.0 for y in data[y_column]], index=None)
    return y[0].apply(lambda x: 1 if x > 0.7 else 0)

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.read_csv('1700_final_norepeat.csv')

    not_feature_cols = ['reactant_aldehyde', 'conv']
    sample_sizes = [20, 40, 60, 80, 100]

    # 存储所有划分的结果
    all_results = {k: {'f1': []} for k in sample_sizes}

    # 5次不同的8:1:1划分
    for split_idx in range(5):
        print(f"\n{'='*60}")
        print(f"第 {split_idx + 1} 次划分 (8:1:1)")
        print(f"{'='*60}")

        # 第一次划分：训练80%，临时保存
        X_temp = data.drop(not_feature_cols + ['conv'], axis=1)
        y_all = prepare_labels(data, 'conv')

        X_train_full, X_temp2, y_train_full, y_temp2, idx_train, idx_temp = train_test_split(
            X_temp.values, y_all.values, np.arange(len(data)),
            test_size=0.2, random_state=split_idx
        )

        # 第二次划分：验证10%，测试10%
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp2, y_temp2, test_size=0.5, random_state=split_idx
        )

        # 获取训练集DataFrame（用于CVT采样）
        train_data = data.iloc[idx_train].reset_index(drop=True)

        print(f"训练集大小: {len(train_data)}, 验证集大小: {len(y_valid)}, 测试集大小: {len(y_test)}")

        # 在训练集上进行不同采样数的CVT采样
        for k in sample_sizes:
            print(f"\n--- 采样数: {k} ---")

            # CVT采样（使用固定随机种子确保同一次划分中20-100采样使用相同随机数）
            np.random.seed(42)
            sampled_points, unsampled_points = cvt_sampling_df(
                data=train_data,
                k=k,
                not_feature_columns=not_feature_cols
            )

            # 保存采样点
            sampled_points.to_csv(f'./xgb_classify/split{split_idx}_k{k}_samples.csv', index=False)

            # 准备训练数据
            X_train = sampled_points.drop(not_feature_cols, axis=1).values
            y_train = prepare_labels(sampled_points, 'conv')

            # Optuna超参优化
            study = optuna.create_study(direction='maximize')
            study.optimize(
                partial(objective, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid),
                n_trials=800
            )

            print(f"\n=== Split {split_idx + 1}, k={k} Optuna Results ===")
            trial = study.best_trial
            print(f"  Validation F1: {trial.value:.4f}")

            # 使用最佳参数训练最终模型
            best_params = trial.params
            best_params.update({'random_state': 42, 'n_jobs': -1})
            best_model = XGBClassifier(**best_params)
            best_model.fit(X_train, y_train)

            # 测试集评估
            print("\n=== Test Set Evaluation ===")
            acc_test, recall_test, f1_test = evaluate_performance(best_model, X_test, y_test, "Test")

            # 记录结果
            all_results[k]['f1'].append(f1_test)

    # 计算均值和标准差
    print("\n" + "="*60)
    print("最终结果 (Mean ± Std)")
    print("="*60)

    f1_means = []
    f1_stds = []

    results_summary = []
    for k in sample_sizes:
        f1_mean = np.mean(all_results[k]['f1'])
        f1_std = np.std(all_results[k]['f1'])

        f1_means.append(f1_mean)
        f1_stds.append(f1_std)

        print(f"Sample Size {k}: F1 = {f1_mean:.4f} ± {f1_std:.4f}")

        results_summary.append({
            'Sample_Size': k,
            'F1_Mean': f1_mean,
            'F1_Std': f1_std,
        })

    # 保存结果
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('./xgb_classify/f1_results.csv', index=False)
    print("\n=== Results saved to f1_results.csv ===")
    print(results_df)
