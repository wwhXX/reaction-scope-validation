# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 从utils.py导入计算函数
from utils import calc_entropy, calc_mean_squared_distance

# 从sampling.py导入采样方法
from sampling import (
    cvt_sampling_df_norepeat, 
    cvt_sampling_gold_df_norepeat, 
    fps_sampling_df_norepeat, 
    lhs_sampling_df_norepeat, 
    sobol_sampling_df_norepeat,
    rand_sampling_df_no_repeat,
    weighted_itr_cvt_sampling_df_norepeat,
    ward_clustering_df_norepeat,
    kennard_stone_sampling_df_norepeat
)


def _compute_single_entropy_dual_space(args):
    """
    用于在原始空间和降维空间中并行计算熵的辅助函数。
    基于降维空间进行采样，然后在两个空间中计算熵。
    
    参数:
    -----------
    args : tuple
        (original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
         reduced_not_feature_col, sampling_function, distance_metric, seed)
    
    返回:
    --------
    tuple
        (original_space_entropy, reduced_space_entropy)
    """
    (original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
     reduced_not_feature_col, sampling_function, distance_metric, seed) = args
    
    # 设置随机种子以确保可重现性
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    
    # 在降维空间中进行采样
    try:
        sample, _ = sampling_function(reduced_complete_df, sample_size, reduced_not_feature_col, distance_metric=distance_metric)
    except TypeError:
        sample, _ = sampling_function(reduced_complete_df, sample_size, reduced_not_feature_col)
    
    # 获取样本索引
    sample_indices = sample.index
    
    # 使用采样索引在原始空间中计算熵
    original_entropy = calc_entropy(original_complete_df, original_complete_df.loc[sample_indices], original_not_feature_col)
    
    # 在降维空间中计算熵
    reduced_entropy = calc_entropy(reduced_complete_df, sample, reduced_not_feature_col)
    
    return original_entropy, reduced_entropy


def _compute_single_msd_dual_space(args):
    """
    用于在原始空间和降维空间中并行计算平均距离的辅助函数。
    基于降维空间进行采样，然后在两个空间中计算平均距离。
    
    参数:
    -----------
    args : tuple
        (original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
         reduced_not_feature_col, sampling_function, distance_metric, seed)
    
    返回:
    --------
    tuple
        (original_space_msd, reduced_space_msd)
    """
    (original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
     reduced_not_feature_col, sampling_function, distance_metric, seed) = args
    
    # 设置随机种子以确保可重现性
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    
    # 在降维空间中进行采样
    try:
        sample, _ = sampling_function(reduced_complete_df, sample_size, reduced_not_feature_col, distance_metric=distance_metric)
    except TypeError:
        sample, _ = sampling_function(reduced_complete_df, sample_size, reduced_not_feature_col)
    
    # 获取样本索引
    sample_indices = sample.index
    
    # 使用采样索引在原始空间中计算平均距离
    if distance_metric == 'manhattan':
        # 对于曼哈顿距离使用平均距离
        from utils import calc_mean_distance
        original_msd = calc_mean_distance(original_complete_df, original_complete_df.loc[sample_indices], original_not_feature_col)
    else:
        original_msd = calc_mean_squared_distance(original_complete_df, original_complete_df.loc[sample_indices], original_not_feature_col)
    
    # 在降维空间中计算平均距离
    if distance_metric == 'manhattan':
        from utils import calc_mean_distance
        reduced_msd = calc_mean_distance(reduced_complete_df, sample, reduced_not_feature_col)
    else:
        reduced_msd = calc_mean_squared_distance(reduced_complete_df, sample, reduced_not_feature_col)
    
    return original_msd, reduced_msd


def _parallel_computation(computation_func, args_list, n_workers=None):
    """
    使用多进程执行并行计算。
    
    参数:
    -----------
    computation_func : function
        要并行执行的函数
    args_list : list
        每次计算的参数列表
    n_workers : int, optional
        工作进程数。如果为None，则使用所有可用的CPU
    
    返回:
    --------
    list
        计算结果列表
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    with Pool(n_workers) as pool:
        results = pool.map(computation_func, args_list)
    
    return results


def evaluate_sampling_method_pca(complete_space_df, sample_size, not_feature_col, 
                               sampling_method_name, deviation=0.05, pca_components=50, 
                               n_initial_samples=10, distance_metric='euclidean', n_workers=None):
    """
    使用PCA对特定采样方法评估空间熵和平均距离。
    强制降维到50维，基于降维空间确定采样策略和次数，然后在原始空间和降维空间中分别计算指标。
    
    参数:
    -----------
    complete_space_df : pandas.DataFrame
        完整空间DataFrame
    sample_size : int
        采样点数量
    not_feature_col : list
        要排除的非特征列列表
    sampling_method_name : str
        采样方法名称，支持: 'cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'random', 'weighted_itr_cvt', 'ward', 'kennard_stone'
    deviation : float, default=0.05
        预期偏差，占平均值的百分比（例如0.05表示5%）
    pca_components : int, default=50
        保留的PCA组件数量（强制为50）
    n_initial_samples : int, default=10
        初始样本数量
    distance_metric : str, default='euclidean'
        采样方法的距离度量，支持 'euclidean', 'manhattan', 'cosine', 'tanimoto'
    n_workers : int, optional
        并行计算的工作进程数。如果为None，则使用所有可用的CPU
    
    返回:
    --------
    dict
        包含两个空间中空间熵和平均距离评估结果的字典
    """
    
    # 获取采样函数
    sampling_functions = {
        'cvt': cvt_sampling_df_norepeat,
        'cvt_gold': cvt_sampling_gold_df_norepeat,
        'fps': fps_sampling_df_norepeat,
        'lhs': lhs_sampling_df_norepeat,
        'sobol': sobol_sampling_df_norepeat,
        'random': rand_sampling_df_no_repeat,
        'weighted_itr_cvt': weighted_itr_cvt_sampling_df_norepeat,
        'ward': ward_clustering_df_norepeat,
        'kennard_stone': kennard_stone_sampling_df_norepeat
    }
    
    if sampling_method_name not in sampling_functions:
        raise ValueError(f"不支持的采样方法: {sampling_method_name}. 支持的方法: {list(sampling_functions.keys())}")
    
    sampling_function = sampling_functions[sampling_method_name]
    
    # 数据预处理：PCA降维和标准化
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty:
        raise ValueError("DataFrame必须包含数值特征列才能计算空间熵")
    
    print(f"原始特征维度: {complete_features.shape[1]}")
    
    # 存储原始完整数据框
    original_complete_df = complete_space_df.copy()
    original_not_feature_col = not_feature_col.copy()
    
    # 步骤1：标准化
    scaler = StandardScaler()
    normalized_complete = scaler.fit_transform(complete_features)
    
    # 步骤2：PCA降维 - 强制降维到pca_components维
    pca = PCA(n_components=min(pca_components, complete_features.shape[1], complete_features.shape[0]))
    pca_complete = pca.fit_transform(normalized_complete)
    
    print(f"PCA强制降维到: {pca_complete.shape[1]} 维 (保留方差: {np.sum(pca.explained_variance_ratio_):.4f})")
    
    # 步骤3：构建处理后的降维数据集
    reduced_complete_df = complete_space_df[not_feature_col].copy()
    
    # 添加PCA降维后的特征列
    pca_feature_names = [f'pca_component_{i}' for i in range(pca_complete.shape[1])]
    pca_df = pd.DataFrame(pca_complete, columns=pca_feature_names, index=complete_space_df.index)
    reduced_complete_df = pd.concat([reduced_complete_df, pca_df], axis=1)
    
    # 更新降维空间的非特征列列表（与原始空间相同）
    reduced_not_feature_col = not_feature_col.copy()
    
    # 使用95%置信水平z值
    z_value = 1.96
    
    print(f"开始评估 {sampling_method_name} 采样方法...")
    
    # 确定距离度量名称用于输出
    distance_measure_name = "平均距离" if distance_metric == 'manhattan' else "平均平方距离"
    
    # ============ 空间熵评估 - 基于降维空间确定采样次数 ============
    print("在原始空间和降维空间中计算空间熵...")
    
    # 并行初始采样用于熵计算
    entropy_args = [(original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
                     reduced_not_feature_col, sampling_function, distance_metric, None) for _ in range(n_initial_samples)]
    
    print(f"并行运行 {n_initial_samples} 个初始熵计算...")
    initial_entropy_results = _parallel_computation(_compute_single_entropy_dual_space, entropy_args, n_workers)
    
    # 分离原始空间和降维空间的结果
    initial_original_entropies = [result[0] for result in initial_entropy_results]
    initial_reduced_entropies = [result[1] for result in initial_entropy_results]
    
    # 基于降维空间的熵计算所需样本数
    mean_reduced_entropy = np.mean(initial_reduced_entropies)
    std_reduced_entropy = np.std(initial_reduced_entropies, ddof=1)
    expected_std_reduced_entropy = abs(mean_reduced_entropy) * deviation
    required_samples_entropy = max(n_initial_samples, int(np.ceil((z_value**2 * std_reduced_entropy**2) / (expected_std_reduced_entropy**2))))
    
    # 也计算原始空间的统计信息用于显示
    mean_original_entropy = np.mean(initial_original_entropies)
    std_original_entropy = np.std(initial_original_entropies, ddof=1)
    expected_std_original_entropy = abs(mean_original_entropy) * deviation
    
    print(f'{sampling_method_name} 采样熵 (原始空间) - 初始标准差: {std_original_entropy:.6f}, 期望标准差: {expected_std_original_entropy:.6f}')
    print(f'{sampling_method_name} 采样熵 (降维空间) - 初始标准差: {std_reduced_entropy:.6f}, 期望标准差: {expected_std_reduced_entropy:.6f}')
    print(f'{sampling_method_name} 采样熵所需样本数 (基于降维空间): {required_samples_entropy}')
    
    # 额外采样
    additional_samples_entropy = max(0, required_samples_entropy - n_initial_samples)
    all_original_entropies = initial_original_entropies.copy()
    all_reduced_entropies = initial_reduced_entropies.copy()
    
    if additional_samples_entropy > 0:
        print(f"并行运行 {additional_samples_entropy} 个额外熵计算...")
        additional_entropy_args = [(original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
                                    reduced_not_feature_col, sampling_function, distance_metric, None) for _ in range(additional_samples_entropy)]
        additional_entropy_results = _parallel_computation(_compute_single_entropy_dual_space, additional_entropy_args, n_workers)
        
        additional_original_entropies = [result[0] for result in additional_entropy_results]
        additional_reduced_entropies = [result[1] for result in additional_entropy_results]
        all_original_entropies.extend(additional_original_entropies)
        all_reduced_entropies.extend(additional_reduced_entropies)
    
    final_mean_original_entropy = np.mean(all_original_entropies)
    final_mean_reduced_entropy = np.mean(all_reduced_entropies)
    print(f'{sampling_method_name} 采样最终平均熵 (原始空间): {final_mean_original_entropy:.6f}')
    print(f'{sampling_method_name} 采样最终平均熵 (降维空间): {final_mean_reduced_entropy:.6f}')
    
    # ============ 平均距离评估 - 基于降维空间确定采样次数 ============
    print(f"在原始空间和降维空间中计算{distance_measure_name.lower()}...")
    
    # 并行初始采样用于平均距离计算  
    msd_args = [(original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
                 reduced_not_feature_col, sampling_function, distance_metric, None) for _ in range(n_initial_samples)]
    
    print(f"并行运行 {n_initial_samples} 个初始{distance_measure_name.lower()}计算...")
    initial_msd_results = _parallel_computation(_compute_single_msd_dual_space, msd_args, n_workers)
    
    # 分离原始空间和降维空间的结果
    initial_original_msds = [result[0] for result in initial_msd_results]
    initial_reduced_msds = [result[1] for result in initial_msd_results]
    
    # 基于降维空间的平均距离计算所需样本数
    mean_reduced_msd = np.mean(initial_reduced_msds)
    std_reduced_msd = np.std(initial_reduced_msds, ddof=1)
    expected_std_reduced_msd = abs(mean_reduced_msd) * deviation
    required_samples_msd = max(n_initial_samples, int(np.ceil((z_value**2 * std_reduced_msd**2) / (expected_std_reduced_msd**2))))
    
    # 也计算原始空间的统计信息用于显示
    mean_original_msd = np.mean(initial_original_msds)
    std_original_msd = np.std(initial_original_msds, ddof=1)
    expected_std_original_msd = abs(mean_original_msd) * deviation
    
    print(f'{sampling_method_name} 采样{distance_measure_name} (原始空间) - 初始标准差: {std_original_msd:.6f}, 期望标准差: {expected_std_original_msd:.6f}')
    print(f'{sampling_method_name} 采样{distance_measure_name} (降维空间) - 初始标准差: {std_reduced_msd:.6f}, 期望标准差: {expected_std_reduced_msd:.6f}')
    print(f'{sampling_method_name} 采样{distance_measure_name}所需样本数 (基于降维空间): {required_samples_msd}')
    
    # 额外采样
    additional_samples_msd = max(0, required_samples_msd - n_initial_samples)
    all_original_msds = initial_original_msds.copy()
    all_reduced_msds = initial_reduced_msds.copy()
    
    if additional_samples_msd > 0:
        print(f"并行运行 {additional_samples_msd} 个额外{distance_measure_name.lower()}计算...")
        additional_msd_args = [(original_complete_df, reduced_complete_df, sample_size, original_not_feature_col, 
                                reduced_not_feature_col, sampling_function, distance_metric, None) for _ in range(additional_samples_msd)]
        additional_msd_results = _parallel_computation(_compute_single_msd_dual_space, additional_msd_args, n_workers)
        
        additional_original_msds = [result[0] for result in additional_msd_results]
        additional_reduced_msds = [result[1] for result in additional_msd_results]
        all_original_msds.extend(additional_original_msds)
        all_reduced_msds.extend(additional_reduced_msds)
    
    final_mean_original_msd = np.mean(all_original_msds)
    final_mean_reduced_msd = np.mean(all_reduced_msds)
    print(f'{sampling_method_name} 采样最终{distance_measure_name.lower()} (原始空间): {final_mean_original_msd:.6f}')
    print(f'{sampling_method_name} 采样最终{distance_measure_name.lower()} (降维空间): {final_mean_reduced_msd:.6f}')
    
    # 返回结果
    return {
        'method': sampling_method_name,
        'sample_size': sample_size,
        'pca_components': pca_complete.shape[1],
        'samples_used_entropy': len(all_original_entropies),
        'samples_used_msd': len(all_original_msds),
        'original_space': {
            'entropy': {
                'mean': final_mean_original_entropy,
                'std': np.std(all_original_entropies, ddof=1),
                'samples_used': len(all_original_entropies)
            },
            'mean_distance': {
                'mean': final_mean_original_msd,
                'std': np.std(all_original_msds, ddof=1),
                'samples_used': len(all_original_msds)
            }
        },
        'reduced_space': {
            'entropy': {
                'mean': final_mean_reduced_entropy,
                'std': np.std(all_reduced_entropies, ddof=1),
                'samples_used': len(all_reduced_entropies)
            },
            'mean_distance': {
                'mean': final_mean_reduced_msd,
                'std': np.std(all_reduced_msds, ddof=1),
                'samples_used': len(all_reduced_msds)
            }
        }
    }


def compare_sampling_methods_pca(complete_space_df, sample_size, not_feature_col, 
                               methods=['cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'random', 'weighted_itr_cvt', 'ward', 'kennard_stone'],
                               deviation=0.05, pca_components=50, n_initial_samples=10, distance_metrics=['euclidean'], n_workers=None):
    """
    使用PCA比较多种采样方法的空间熵和平均距离。
    基于降维空间确定采样策略和次数。
    
    参数:
    -----------
    complete_space_df : pandas.DataFrame
        完整空间DataFrame
    sample_size : int
        采样点数量
    not_feature_col : list
        要排除的非特征列列表
    methods : list, default=['cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'random', 'weighted_itr_cvt', 'ward', 'kennard_stone']
        要比较的采样方法列表
    deviation : float, default=0.05
        期望偏差
    pca_components : int, default=50
        保留的PCA组件数量（强制为50）
    n_initial_samples : int, default=10
        初始样本数量
    distance_metrics : list, default=['euclidean']
        采样方法的距离度量列表，支持 'euclidean', 'manhattan', 'cosine', 'tanimoto'
    n_workers : int, optional
        并行计算的工作进程数。如果为None，则使用所有可用的CPU
    
    返回:
    --------
    list
        包含所有采样方法评估结果的列表
    """
    results = []
    
    for distance_metric in distance_metrics:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"评估采样方法: {method.upper()} 使用 {distance_metric.upper()} 距离")
            print(f"{'='*60}")
            
            try:
                result = evaluate_sampling_method_pca(
                    complete_space_df=complete_space_df,
                    sample_size=sample_size,
                    not_feature_col=not_feature_col,
                    sampling_method_name=method,
                    deviation=deviation,
                    pca_components=pca_components,
                    n_initial_samples=n_initial_samples,
                    distance_metric=distance_metric,
                    n_workers=n_workers
                )
                # 将距离度量添加到结果中以供识别
                result['distance_metric'] = distance_metric
                results.append(result)
            except Exception as e:
                print(f"评估 {method} 方法与 {distance_metric} 距离时出错: {str(e)}")
                continue
    
    return results


def print_comparison_results_pca(results):
    """
    打印PCA评估的采样方法比较结果。
    
    参数:
    -----------
    results : list
        由compare_sampling_methods_pca函数返回的结果列表
    """
    if not results:
        print("没有可用的比较结果")
        return
    
    print(f"\n{'='*120}")
    print("PCA采样方法比较结果（基于降维空间确定采样次数）")
    print(f"{'='*120}")
    
    # 为两个空间创建比较表
    spaces = ['original_space', 'reduced_space']
    space_names = ['原始空间', '降维空间']
    
    for space, space_name in zip(spaces, space_names):
        print(f"\n{space_name}结果:")
        print(f"{'方法':<12} {'距离':<12} {'熵':<15} {'熵标准差':<15} {'平均距离':<15} {'距离标准差':<15} {'样本数':<8}")
        print(f"{'-'*110}")
        
        for result in results:
            method = result['method']
            distance = result.get('distance_metric', 'euclidean')
            entropy_mean = result[space]['entropy']['mean']
            entropy_std = result[space]['entropy']['std']
            msd_mean = result[space]['mean_distance']['mean']
            msd_std = result[space]['mean_distance']['std']
            samples_used = result[space]['entropy']['samples_used']
            
            print(f"{method:<12} {distance:<12} {entropy_mean:<15.6f} {entropy_std:<15.6f} {msd_mean:<15.6f} {msd_std:<15.6f} {samples_used:<8}")
    
    # 找出每个空间的最佳方法
    print(f"\n{'='*120}")
    print("按距离度量和空间划分的最佳方法:")
    
    # 按距离度量分组结果
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for result in results:
        distance_metric = result.get('distance_metric', 'euclidean')
        grouped_results[distance_metric].append(result)
    
    for distance_metric, metric_results in grouped_results.items():
        print(f"\n{distance_metric.upper()} 距离:")
        
        for space, space_name in zip(spaces, space_names):
            print(f"  {space_name}:")
            # 熵最高的方法
            best_entropy = max(metric_results, key=lambda x: x[space]['entropy']['mean'])
            print(f"    最高熵: {best_entropy['method']} (熵: {best_entropy[space]['entropy']['mean']:.6f})")
            
            # 平均距离最低的方法
            best_dist = min(metric_results, key=lambda x: x[space]['mean_distance']['mean'])
            print(f"    最低平均距离: {best_dist['method']} (距离: {best_dist[space]['mean_distance']['mean']:.6f})")


def plot_pareto_frontier_pca(results, save_path=None):
    """
    使用seaborn绘制熵和负平均距离的帕累托前沿散点图，
    为每个距离度量和空间（原始vs降维）分别绘制图表。
    
    参数:
    -----------
    results : list
        由compare_sampling_methods_pca函数返回的结果列表
    save_path : str, optional
        保存图表的路径前缀。如果为None，将显示但不保存图表。
    """
    if not results:
        print("没有可用于绘图的结果")
        return
    
    # 按距离度量分组结果
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for result in results:
        distance_metric = result.get('distance_metric', 'euclidean')
        grouped_results[distance_metric].append(result)
    
    # 定义要绘制的空间
    spaces = ['original_space', 'reduced_space']
    space_names = ['原始空间', '降维空间']
    
    # 为每个距离度量和空间组合创建子图
    n_metrics = len(grouped_results)
    n_spaces = len(spaces)
    fig, axes = plt.subplots(n_spaces, n_metrics, figsize=(6*n_metrics, 6*n_spaces))
    
    # 处理单一度量的情况
    if n_metrics == 1 and n_spaces == 1:
        axes = [[axes]]
    elif n_metrics == 1:
        axes = [[ax] for ax in axes]
    elif n_spaces == 1:
        axes = [axes]
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    
    for row_idx, (space, space_name) in enumerate(zip(spaces, space_names)):
        for col_idx, (distance_metric, metric_results) in enumerate(grouped_results.items()):
            ax = axes[row_idx][col_idx]
            
            # 提取绘图数据
            data = []
            for result in metric_results:
                data.append({
                    'method': result['method'].upper(),
                    'entropy': result[space]['entropy']['mean'],
                    'neg_mean_dist': -result[space]['mean_distance']['mean'],  # 负值用于最大化
                    'mean_dist': result[space]['mean_distance']['mean']
                })
            
            df = pd.DataFrame(data)
            
            # 使用seaborn创建散点图
            sns.scatterplot(data=df, x='entropy', y='neg_mean_dist', hue='method', 
                           s=150, alpha=0.8, edgecolor='black', linewidth=1, ax=ax)
            
            # 识别帕累托前沿点
            entropies = df['entropy'].values
            neg_mean_dists = df['neg_mean_dist'].values
            methods = df['method'].values
            
            pareto_indices = []
            for i in range(len(entropies)):
                is_pareto = True
                for j in range(len(entropies)):
                    if i != j:
                        # 如果点j在两个目标上都更好，则点j支配点i
                        if entropies[j] >= entropies[i] and neg_mean_dists[j] >= neg_mean_dists[i]:
                            if entropies[j] > entropies[i] or neg_mean_dists[j] > neg_mean_dists[i]:
                                is_pareto = False
                                break
                if is_pareto:
                    pareto_indices.append(i)
            
            # 按熵排序帕累托点以绘制连接线
            pareto_points = [(entropies[i], neg_mean_dists[i]) for i in pareto_indices]
            pareto_points.sort(key=lambda x: x[0])
            
            # 绘制帕累托前沿线
            if len(pareto_points) > 1:
                pareto_x, pareto_y = zip(*pareto_points)
                ax.plot(pareto_x, pareto_y, 'r--', linewidth=3, alpha=0.8, label='Pareto Frontier')
            
            # 突出显示帕累托最优点
            pareto_df = df.iloc[pareto_indices]
            if not pareto_df.empty:
                ax.scatter(pareto_df['entropy'], pareto_df['neg_mean_dist'], 
                           c='red', s=200, alpha=0.9, marker='*', 
                           edgecolors='black', linewidth=2, 
                           label='Pareto Optimal', zorder=5)
            
            # 自定义图表
            ax.set_xlabel('Entropy', fontsize=12, fontweight='bold')
            ax.set_ylabel('Negative Mean Distance', fontsize=12, fontweight='bold')
            
            # 简化标题
            space_label = 'Original' if space == 'original_space' else 'Reduced'
            ax.set_title(f'{space_label} Space - {distance_metric.upper()}', 
                         fontsize=14, fontweight='bold', pad=20)
            
            # 为每个点添加文本注释
            for i, row in df.iterrows():
                ax.annotate(row['method'], (row['entropy'], row['neg_mean_dist']), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=9, alpha=0.9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # 改进图例
            handles, labels = ax.get_legend_handles_labels()
            if len(pareto_points) > 1:
                # 如果存在帕累托前沿，添加到图例中
                ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), 
                          loc='upper left', fontsize=10, frameon=True, 
                          fancybox=True, shadow=True)
            else:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          fontsize=10, frameon=True, fancybox=True, shadow=True)
            
            ax.grid(True, alpha=0.3)
            
            # 打印此距离度量和空间的帕累托最优方法
            print(f"\n{'='*80}")
            print(f"{distance_metric.upper()} Distance - {space_label} Space - Pareto Optimal Methods:")
            print(f"{'='*80}")
            print(f"{'Method':<15} | {'Entropy':<12} | {'Mean Distance':<12}")
            print(f"{'-'*80}")
            for i in pareto_indices:
                method = methods[i]
                entropy = entropies[i]
                original_dist = -neg_mean_dists[i]
                print(f"{method:<15} | {entropy:<12.6f} | {original_dist:<12.6f}")
    
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(f"{save_path}_pca_pareto_frontiers.png", dpi=300, bbox_inches='tight')
        print(f"\nPCA Pareto frontier plots saved to: {save_path}_pca_pareto_frontiers.png")
    else:
        plt.show()


if __name__ == "__main__":
    # 设置多进程启动方法以确保兼容性
    mp.set_start_method('spawn', force=True)
    
    # 示例用法
    import time
    start_time = time.time()
    print("开始PCA采样方法比较测试...")
    print(f"可用CPU核心数: {mp.cpu_count()}")
    
    # 读取测试数据
    try:
        # 读取SMILES数据和特征数据
        smiles_df = pd.read_csv('zinc_aryl_aldehyde.csv')
        features_df = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')
        
        # 合并SMILES和特征数据
        complete_space_df = pd.concat([smiles_df, features_df], axis=1)
        print(f"完整空间数据形状: {complete_space_df.shape}")
        
        # 设置参数
        sample_size = 50
        not_feature_col = ['smiles']
        methods_to_test = ['random', 'cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'weighted_itr_cvt', 'ward', 'kennard_stone']
        distance_metrics_to_test = ['euclidean', 'manhattan', 'cosine']
        
        # 使用PCA和所有距离度量比较采样方法
        results = compare_sampling_methods_pca(
            complete_space_df=complete_space_df,
            sample_size=sample_size,
            not_feature_col=not_feature_col,
            methods=methods_to_test,
            distance_metrics=distance_metrics_to_test,
            deviation=0.05,
            pca_components=50,  # 强制降维到50维
            n_initial_samples=10,
            n_workers=None  # 使用所有可用的CPU
        )
        
        # 打印比较结果
        print_comparison_results_pca(results)
        
        # 绘制帕累托前沿
        print(f"\n{'='*120}")
        print("生成PCA帕累托前沿图...")
        print(f"{'='*120}")
        plot_pareto_frontier_pca(results, save_path='pca_pareto_frontier_plot')
        
        print(f"\n总运行时间: {time.time() - start_time:.2f} 秒")
        
    except FileNotFoundError as e:
        print(f"测试文件未找到: {e}")
        print("请确保zinc_aryl_aldehyde.csv和fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv文件存在")
    except Exception as e:
        print(f"测试过程中出错: {e}")