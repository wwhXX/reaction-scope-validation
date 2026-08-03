# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
from functools import partial
import sampling

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """加载和预处理数据"""
    data = pd.read_csv('1700_final_norepeat.csv')
    
    # 特征列（排除smiles和conv）
    feature_columns = [col for col in data.columns if col not in ['smiles', 'conv']]
    X = data[feature_columns]
    y = data['conv']
    
    # 非特征列
    not_feature_columns = ['smiles', 'conv']
    
    return data, X, y, not_feature_columns

def split_data_5fold(X, y, random_seed=42):
    """
    按照8:1:1比例划分5组训练集、验证集和测试集
    
    返回:
        splits: list of tuples, 每个元素为 (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    np.random.seed(random_seed)
    splits = []
    
    for fold in range(5):
        # 为每一折设置不同的随机种子，确保可复现但每折不同
        fold_seed = random_seed + fold
        
        # 首先分离出20%作为验证+测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=fold_seed, stratify=None
        )
        
        # 将20%均分为验证集和测试集（各10%）
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=fold_seed, stratify=None
        )
        
        splits.append((X_train, X_val, X_test, y_train, y_val, y_test))
        print(f"Fold {fold+1}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return splits

def perform_sampling(train_data, method_name, not_feature_columns, k=20, random_seed=42):
    """
    使用指定方法进行采样
    
    参数:
        train_data: DataFrame, 训练数据
        method_name: str, 采样方法名称
        not_feature_columns: list, 非特征列名
        k: int, 采样数量
        random_seed: int, 随机种子
    
    返回:
        sampled_data: DataFrame, 采样得到的数据
    """
    np.random.seed(random_seed)
    
    sampling_methods = {
        'weighted_itr_cvt': sampling.weighted_itr_cvt_sampling_df_norepeat,
        'lhs': sampling.lhs_sampling_df_norepeat,
        'sobol': sampling.sobol_sampling_df_norepeat,
        'ward_clustering': sampling.ward_clustering_df_norepeat,
        'kennard_stone': sampling.kennard_stone_sampling_df_norepeat
    }
    
    if method_name not in sampling_methods:
        raise ValueError(f"Unknown sampling method: {method_name}")
    
    sampling_func = sampling_methods[method_name]
    
    try:
        if method_name == 'weighted_itr_cvt':
            # weighted_itr_cvt_sampling_df_norepeat 有额外参数
            sampled_centers, _ = sampling_func(
                data=train_data,
                k=k,
                not_feature_columns=not_feature_columns,
                sampled_data=None,
                max_iters=500,
                repulsion_strength=1.0,
                cvt_weight=1.0,
                learning_rate=0.01
            )
        else:
            # 其他采样方法
            sampled_centers, _ = sampling_func(
                data=train_data,
                k=k,
                not_feature_columns=not_feature_columns,
                sampled_data=None
            )
        
        return sampled_centers
    
    except Exception as e:
        print(f"Error in {method_name} sampling: {str(e)}")
        # 如果采样失败，返回None表示失败
        return None

def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100):
    """
    使用Optuna优化XGBoost超参数（与test.py保持一致）
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征  
        y_val: 验证标签
        n_trials: int, Optuna试验次数
    
    返回:
        best_params: dict, 最佳参数
    """
    def objective(trial, X_train, y_train, X_val, y_val):
        # 定义搜索空间（与test.py保持一致）
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 6, 50), 
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse
    
    # 禁用Optuna日志输出
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val), 
        n_trials=n_trials, 
        show_progress_bar=False
    )
    
    best_params = study.best_trial.params
    best_params.update({'random_state': 42, 'n_jobs': -1})
    
    return best_params

def train_and_evaluate_model(sampled_data, validation_data, test_data, not_feature_columns, n_trials=100):
    """
    训练模型并在测试集上评估RMSE（包含超参优化）
    
    参数:
        sampled_data: DataFrame, 采样的训练数据
        validation_data: tuple, (X_val, y_val) 用于超参优化
        test_data: tuple, (X_test, y_test) 用于最终评估
        not_feature_columns: list, 非特征列名
        n_trials: int, Optuna优化次数
    
    返回:
        rmse: float, 测试集RMSE
        best_params: dict, 最佳超参数
    """
    # 准备采样训练数据
    X_train_sampled = sampled_data.drop(not_feature_columns, axis=1)
    y_train_sampled = sampled_data['conv']
    
    # 准备验证和测试数据
    X_val, y_val = validation_data
    X_test, y_test = test_data
    
    # 超参数优化
    best_params = optimize_hyperparameters(
        X_train_sampled, y_train_sampled, 
        X_val, y_val, 
        n_trials=n_trials
    )
    
    # 使用最佳参数训练最终模型
    model = XGBRegressor(**best_params)
    model.fit(X_train_sampled, y_train_sampled)
    
    # 在测试集上评估
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse, best_params

def main():
    """主函数"""
    print("=== 模型RMSE测试开始（含超参优化） ===")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data, X, y, not_feature_columns = load_and_prepare_data()
    print(f"数据形状: {data.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {len(data)}")
    
    # 2. 创建5折数据划分
    print("\n2. 创建5折数据划分...")
    splits = split_data_5fold(X, y, random_seed=42)
    
    # 3. 定义采样方法
    sampling_methods = [
        'weighted_itr_cvt',
        'lhs', 
        'sobol',
        'ward_clustering',
        'kennard_stone'
    ]
    
    # 4. 存储结果
    results = {method: [] for method in sampling_methods}
    all_best_params = {method: [] for method in sampling_methods}
    
    # 5. 对每一折进行实验
    print("\n3. 开始5折交叉验证实验（含Optuna超参优化）...")
    
    for fold_idx, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1} ===")
        
        # 重构训练数据DataFrame（包含所有列）
        train_indices = X_train.index
        train_data = data.loc[train_indices].copy().reset_index(drop=True)
        
        # 对每种采样方法进行测试
        for method_idx, method_name in enumerate(sampling_methods):
            print(f"  测试采样方法: {method_name} ({method_idx+1}/{len(sampling_methods)})")
            
            try:
                # 执行采样
                sampled_data = perform_sampling(
                    train_data=train_data,
                    method_name=method_name,
                    not_feature_columns=not_feature_columns,
                    k=20,
                    random_seed=42 + fold_idx
                )
                
                # 检查采样是否成功
                if sampled_data is None:
                    print(f"    采样失败，记录为N/A")
                    results[method_name].append(np.nan)
                    all_best_params[method_name].append({})
                    continue
                
                print(f"    采样完成，获得 {len(sampled_data)} 个样本")
                print(f"    开始超参数优化...")
                
                # 训练模型并评估（包含超参优化）
                rmse, best_params = train_and_evaluate_model(
                    sampled_data=sampled_data,
                    validation_data=(X_val, y_val),
                    test_data=(X_test, y_test),
                    not_feature_columns=not_feature_columns,
                    n_trials=100  # 与test.py中的200相比，减少一些以提高速度
                )
                
                results[method_name].append(rmse)
                all_best_params[method_name].append(best_params)
                print(f"    RMSE: {rmse:.4f}")
                print(f"    最佳参数: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.4f}")
                
            except Exception as e:
                print(f"    错误: {str(e)}")
                # 记录错误的RMSE为NaN
                results[method_name].append(np.nan)
                all_best_params[method_name].append({})
    
    # 6. 计算统计结果
    print("\n=== 最终结果 ===")
    print("\n采样方法性能对比:")
    print("Method".ljust(20) + "Mean RMSE".ljust(12) + "Std RMSE".ljust(12) + "All RMSEs")
    print("-" * 70)
    
    final_results = {}
    
    for method_name in sampling_methods:
        rmse_values = [x for x in results[method_name] if not np.isnan(x)]
        
        if len(rmse_values) > 0:
            mean_rmse = np.mean(rmse_values)
            std_rmse = np.std(rmse_values)
            final_results[method_name] = {
                'mean': mean_rmse,
                'std': std_rmse,
                'values': rmse_values
            }
        else:
            mean_rmse = np.nan
            std_rmse = np.nan
            final_results[method_name] = {
                'mean': mean_rmse,
                'std': std_rmse,
                'values': []
            }
        
        # 格式化输出
        rmse_str = ", ".join([f"{x:.4f}" for x in rmse_values]) if rmse_values else "No valid results"
        print(f"{method_name}".ljust(20) + 
              f"{mean_rmse:.4f}".ljust(12) + 
              f"{std_rmse:.4f}".ljust(12) + 
              f"{rmse_str}")
    
    # 7. 保存详细结果到CSV
    results_df_data = []
    for method_name in sampling_methods:
        for fold_idx, rmse in enumerate(results[method_name]):
            results_df_data.append({
                'method': method_name,
                'fold': fold_idx + 1,
                'rmse': rmse
            })
    
    results_df = pd.DataFrame(results_df_data)
    results_df.to_csv('sampling_rmse_results.csv', index=False, encoding='utf-8')
    print(f"\n详细结果已保存至: sampling_rmse_results.csv")
    
    # 8. 保存汇总统计
    summary_data = []
    for method_name in sampling_methods:
        summary_data.append({
            'method': method_name,
            'mean_rmse': final_results[method_name]['mean'],
            'std_rmse': final_results[method_name]['std'],
            'valid_folds': len(final_results[method_name]['values'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('sampling_rmse_summary.csv', index=False, encoding='utf-8')
    print(f"汇总统计已保存至: sampling_rmse_summary.csv")
    
    # 9. 保存超参数信息
    params_data = []
    for method_name in sampling_methods:
        for fold_idx, params in enumerate(all_best_params[method_name]):
            if params:  # 如果有有效参数
                params_data.append({
                    'method': method_name,
                    'fold': fold_idx + 1,
                    'n_estimators': params.get('n_estimators', np.nan),
                    'max_depth': params.get('max_depth', np.nan),
                    'learning_rate': params.get('learning_rate', np.nan),
                    'reg_alpha': params.get('reg_alpha', np.nan),
                    'reg_lambda': params.get('reg_lambda', np.nan),
                    'min_child_weight': params.get('min_child_weight', np.nan)
                })
    
    if params_data:
        params_df = pd.DataFrame(params_data)
        params_df.to_csv('sampling_best_params.csv', index=False, encoding='utf-8')
        print(f"最佳超参数已保存至: sampling_best_params.csv")
    
    print("\n=== 实验完成 ===")

if __name__ == "__main__":
    main()