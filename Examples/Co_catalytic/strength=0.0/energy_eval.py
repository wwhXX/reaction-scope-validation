import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import cvt

def compute_repulsion_energy_for_dataset(data_points, repulsion_points, repulsion_strength, metric='euclidean'):
    """
    计算数据集中每个点受到所有排斥点的总排斥能
    
    参数:
        data_points: numpy.ndarray，数据集中的所有点
        repulsion_points: numpy.ndarray，排斥点（已drop的点）
        repulsion_strength: float，排斥强度系数
        metric: str，距离度量方法
    
    返回:
        numpy.ndarray，每个数据点的总排斥能
    """
    if len(repulsion_points) == 0:
        return np.zeros(len(data_points))
    
    total_energies = np.zeros(len(data_points))
    
    for i, data_point in enumerate(data_points):
        energy = 0
        for repulsion_point in repulsion_points:
            diff = data_point - repulsion_point
            
            if metric == 'euclidean':
                distance_sq = np.sum(diff**2) + 1e-10  # 避免除零
                # 排斥能：repulsion_strength / distance²
                energy += repulsion_strength / distance_sq
            elif metric == 'manhattan':
                manhattan_dist = np.sum(np.abs(diff)) + 1e-10
                energy += repulsion_strength / (manhattan_dist**2)
            elif metric == 'cosine':
                distance_sq = np.sum(diff**2) + 1e-10
                energy += repulsion_strength / distance_sq
        
        total_energies[i] = energy
    
    return total_energies

def calculate_repulsion_coefficient(data_features, repulsion_strength_exp=0.0, distance_metric='euclidean'):
    """
    计算排斥系数，参考cvt.py中的方法
    
    参数:
        data_features: numpy.ndarray，数据特征
        repulsion_strength_exp: float，排斥强度指数（默认为0）
        distance_metric: str，距离度量方法
    
    返回:
        float，最终排斥强度
    """
    print("计算数据中两点间的最大距离...")
    
    # 计算完整距离矩阵并找出最大值
    distance_matrix = pairwise_distances(data_features, data_features, metric=distance_metric)
    max_distance = np.max(distance_matrix)
    
    print(f"数据中最大距离: {max_distance:.6f}")
    
    # 计算均匀采样下采样点间的期望平方距离
    d = data_features.shape[1]  # 特征维数
    expected_sq_distance = (max_distance**2) * d / 12
    print(f"均匀采样下采样点间期望平方距离: {expected_sq_distance:.6f}")
    
    # 计算排斥函数系数，使排斥力和CVT能量在同一数量级
    n_points = len(data_features)
    base_repulsion_coefficient = expected_sq_distance**2 * n_points
    
    # 与10^repulsion_strength_exp相乘得到最终系数
    final_repulsion_strength = base_repulsion_coefficient * (10 ** repulsion_strength_exp)
    
    print(f"基础排斥系数: {base_repulsion_coefficient:.6e}")
    print(f"最终排斥强度 (base * 10^{repulsion_strength_exp}): {final_repulsion_strength:.6e}")
    
    return final_repulsion_strength

def calculate_energy_for_iteration(iteration_id, final_drop_df, not_feature_cols=['smiles'], distance_metric='euclidean', repulsion_strength_exp=0.0):
    """
    计算指定迭代到当前为止的排斥能
    
    参数:
        iteration_id: int，迭代编号（1-5）
        final_drop_df: DataFrame，包含所有最终drop的点
        not_feature_cols: list，非特征列
        distance_metric: str，距离度量方法
        repulsion_strength_exp: float，排斥强度指数
    
    返回:
        DataFrame，包含每个数据点及其排斥能
    """
    print(f'Processing iteration {iteration_id}...')
    
    # 读取数据集
    labeled_data = pd.read_csv(f'labeled_points_itr0.csv')
    
    # 提取特征数据（排除非特征列、labels列和ScreenLabel列）
    exclude_cols = not_feature_cols + ['labels', 'ScreenLabel']
    feature_cols = [col for col in labeled_data.columns if col not in exclude_cols]
    X_all = labeled_data[feature_cols].values
    
    # 计算排斥系数（只在第一次迭代时计算）
    if iteration_id == 1:
        repulsion_strength = calculate_repulsion_coefficient(X_all, repulsion_strength_exp, distance_metric)
    else:
        # 重复计算以保持一致性
        repulsion_strength = calculate_repulsion_coefficient(X_all, repulsion_strength_exp, distance_metric)
    
    # 获取到当前迭代为止所有drop的点
    dropped_smiles = set()
    
    for itr in range(1, iteration_id + 1):
        try:
            sampled_data = pd.read_csv(f'sampled_points_itr{itr}.csv')
            
            # 从final_drop.csv中找到属于此迭代的drop点
            for _, row in final_drop_df.iterrows():
                smiles = row['smiles']
                # 检查这个smiles是否在当前迭代的采样点中
                if smiles in sampled_data['smiles'].values:
                    dropped_smiles.add(smiles)
                    
        except FileNotFoundError:
            print(f'Warning: sampled_points_itr{itr}.csv not found, skipping this iteration')
            continue
    
    print(f'Found {len(dropped_smiles)} dropped points up to iteration {iteration_id}')
    
    if len(dropped_smiles) == 0:
        # 如果没有drop点，所有点的排斥能都是0
        energy_results = labeled_data.copy()
        energy_results['repulsion_energy'] = 0.0
        return energy_results
    
    # 获取drop点的特征向量
    drop_indices = []
    for idx, row in labeled_data.iterrows():
        if row['smiles'] in dropped_smiles:
            drop_indices.append(idx)
    
    if len(drop_indices) == 0:
        print(f'Warning: No matching dropped points found in labeled data for iteration {iteration_id}')
        energy_results = labeled_data.copy()
        energy_results['repulsion_energy'] = 0.0
        return energy_results
    
    X_dropped = X_all[drop_indices]
    
    print(f'Computing repulsion energies using {distance_metric} distance with strength {repulsion_strength:.6e}...')
    
    # 计算排斥能
    repulsion_energies = compute_repulsion_energy_for_dataset(X_all, X_dropped, repulsion_strength, distance_metric)
    
    # 构建结果DataFrame
    energy_results = labeled_data.copy()
    energy_results['repulsion_energy'] = repulsion_energies
    
    print(f'Energy computation completed. Min energy: {repulsion_energies.min():.6f}, Max energy: {repulsion_energies.max():.6f}')
    
    return energy_results

def main():
    """
    主函数：为5个迭代分别计算排斥能并保存
    """
    print('Starting energy evaluation for all iterations...')
    
    # 读取final_drop.csv
    try:
        final_drop_df = pd.read_csv('final_drop.csv')
        print(f'Loaded {len(final_drop_df)} total dropped points from final_drop.csv')
    except FileNotFoundError:
        print('Error: final_drop.csv not found')
        return
    
    # 设置排斥强度指数为0
    repulsion_strength_exp = 0.0
    print(f'Using repulsion strength exponent: {repulsion_strength_exp}')
    
    # 为每个迭代计算排斥能
    for iteration_id in range(1, 7):  # 1到6
        try:
            energy_results = calculate_energy_for_iteration(
                iteration_id=iteration_id,
                final_drop_df=final_drop_df,
                not_feature_cols=['smiles'],
                distance_metric='euclidean',
                repulsion_strength_exp=repulsion_strength_exp
            )
            
            # 保存结果
            output_filename = f'energy_eval_itr{iteration_id}.csv'
            energy_results.to_csv(output_filename, index=False)
            print(f'Saved energy evaluation results for iteration {iteration_id} to {output_filename}')
            print(f'Dataset size: {len(energy_results)}, Average repulsion energy: {energy_results["repulsion_energy"].mean():.6f}\n')
            
        except Exception as e:
            print(f'Error processing iteration {iteration_id}: {e}\n')
            continue
    
    print('Energy evaluation completed for all iterations.')

if __name__ == '__main__':
    main()