import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import optuna
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from math import sqrt
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt

def compute_cluster_centroid(cluster_points, metric='euclidean'):
    """
    根据距离度量计算聚类中心点
    
    参数:
        cluster_points: numpy.ndarray，聚类中的数据点
        metric: str，距离度量方法
    
    返回:
        numpy.ndarray，聚类中心点
    """
    if len(cluster_points) == 0:
        raise ValueError("聚类点集不能为空")
    
    if metric == 'manhattan':
        # Manhattan距离的最优中心点是中位数
        return np.median(cluster_points, axis=0)
    else:
        # Euclidean和Cosine距离的最优中心点是算术平均值
        return np.mean(cluster_points, axis=0)

def compute_pairwise_distances(X, Y=None, metric='euclidean'):
    """
    计算数据点之间的成对距离，支持多种距离度量
    
    参数:
        X: numpy.ndarray，数据点矩阵
        Y: numpy.ndarray，可选，第二组数据点矩阵。如果为None，则计算X内部的距离
        metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
        numpy.ndarray，距离矩阵
    """
    if metric in ['euclidean', 'manhattan']:
        return pairwise_distances(X, Y, metric=metric)
    elif metric == 'cosine':
        # 使用sklearn的cosine距离计算
        return pairwise_distances(X, Y, metric='cosine')
    else:
        raise ValueError(f"不支持的距离度量: {metric}. 支持的度量: 'euclidean', 'manhattan', 'cosine'")


def cvt_sampling_df(data, k, not_feature_columns, max_iters=500, tol=1e-4):
    """
    CVT采样算法（支持DataFrame输入，保留非数值列）
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        max_iters: 最大迭代次数
        tol: 中心点变化的收敛阈值
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 1. 提取数值特征用于CVT计算
    X = data.drop(not_feature_columns, axis=1).values
    
    # 2. 初始化中心点（从数值特征中随机选择k个）
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for _ in range(max_iters):
        #处理NAN
        if np.isnan(centers_X).any():
            centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
        # 3. Voronoi划分：计算所有点到中心点的距离
        distances = pairwise_distances(X, centers_X)

        labels = np.argmin(distances, axis=1)
        
        # 4. 更新中心点为Voronoi单元的质心
        new_centers_X = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # 5. 检查收敛
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT算法收敛, 迭代次数: {_}")
            break
        centers_X = new_centers_X
    
    # 6. 找到距离中心点最近的原始数据点作为最终采样点
    if np.isnan(centers_X).any():
        centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
    final_distances = pairwise_distances(X, centers_X)
    selected_indices = np.argmin(final_distances, axis=0)
    
    # 7. 构建返回的DataFrame
    centers = data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), selected_indices)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def classify_by_centers(data, centers, not_feature_columns):
    """
    获取一组数据对应的分类
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        centers: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        not_feature_columns: list，非描述列的列名
    
    返回:
        output: DataFrame，添加了"labels"分类列（包含原始所有列）
        注意类别与centers的索引对应
    """
    raw_data = data.copy()
    for not_feature_column in not_feature_columns:
        if not_feature_column in data.columns:
            data = data.drop(not_feature_column, axis=1)
        if not_feature_column in centers.columns:
            centers = centers.drop(not_feature_column, axis=1)
    data = data.values
    centers = centers.values
    # 计算每个样本到各个中心点的距离
    distances = np.sqrt(((data[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
    
    # 找到每个样本距离最近的中心点索引
    labels = np.argmin(distances, axis=1).astype(int)
    if 'labels' in raw_data.columns:
        raw_data = raw_data.drop('labels', axis=1)
    output_data = pd.concat([raw_data, pd.DataFrame(labels, columns=['labels'])], axis=1)

    
    return output_data


def get_sampling(task_itr_id, drop_classes, not_feature_cols, k, max_iters=500, tol=1e-4):
    '''
    进行一次采样与按采样分类。
    
    参数:
        task_itr_id: int，迭代次数(从1开始)
        drop_classes: list，需要排除的类别
        not_feature_cols: list，非特征列的列名
        k: int，采样中心点数量
        max_iters: int，最大迭代次数
        tol: float，中心点变化的收敛阈值
    
    返回:
        sampled_points: DataFrame，采样到的中心点（包含原始所有列，即其中的label为上一迭代的分类）
        labeled_points: DataFrame，按采样分类后的数据和类别（包含原始所有列，类别为sampled_points索引）
    '''
    data = pd.read_csv(f'./itr/labeled_points_itr{task_itr_id-1}.csv')
    
    if drop_classes != []:
        for drop_class in drop_classes:
            data = data[data['labels'] != drop_class].reset_index(drop=True)
    print(data)
    # 执行CVT采样
    sampled_points, unsampled_points = cvt_sampling_df(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        max_iters=max_iters,
        tol=tol
    )
    sampled_points.to_csv(f'./itr/sampled_points_itr{task_itr_id}.csv', index=False)
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols)
    labeled_points.to_csv(f'./itr/labeled_points_itr{task_itr_id}.csv', index=False)
    return sampled_points, labeled_points

def get_sampling_weighted(data, task_itr_id, not_feature_cols, k, max_iters=500, tol=1e-4):
    '''
    进行一次采样与按采样分类。
    
    参数:
        task_itr_id: int，迭代次数(从1开始)
        not_feature_cols: list，非特征列的列名
        k: int，采样中心点数量
        max_iters: int，最大迭代次数
        tol: float，中心点变化的收敛阈值
    
    返回:
        sampled_points: DataFrame，采样到的中心点（包含原始所有列，即其中的label为上一迭代的分类）
        labeled_points: DataFrame，按采样分类后的数据和类别（包含原始所有列，类别为sampled_points索引）
    '''
    sampled_data = data[data['ScreenLabel']!='BASE']
    print(data)
    # 执行CVT采样
    sampled_points, unsampled_points = weighted_itr_cvt_sampling_df_norepeat(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        max_iters=max_iters,
        tol=tol,
        sampled_data=sampled_data,
    )
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols)
    return sampled_points, labeled_points
def weighted_itr_cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=2000, tol=1e-5, 
                                        sampled_data=None, repulsion_strength=0.0, cvt_weight=1.0,
                                        learning_rate=0.01, adaptive_lr=True, distance_metric='euclidean',
                                        cvt_init_iters=100):
    """
    加权迭代CVT采样算法：结合CVT能量最小化和对已采样点的平方反比排斥力
    改进版本：先使用解析CVT方法获得良好初猜，再进行梯度下降优化
    
    参数:
        data: DataFrame，包含特征列和非数值列
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名
        max_iters: 最大迭代次数
        tol: 收敛阈值
        sampled_data: DataFrame，已采样的数据点（不可移动，产生排斥力）。仅对ScreenLabel为'Excluded_Sampled'的点计算排斥力
        repulsion_strength: float，排斥力强度系数
        cvt_weight: float，CVT能量权重
        learning_rate: float，梯度下降学习率
        adaptive_lr: bool，是否使用自适应学习率
        distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
        cvt_init_iters: int，初始CVT解析优化的迭代次数
    
    返回:
        centers: DataFrame，采样到的中心点
        unselected_points: DataFrame，未被采样的点
    """
    
    def compute_distances_metric(points, centers, metric):
        """计算指定度量的距离"""
        return compute_pairwise_distances(points, centers, metric)
    
    def compute_cvt_energy_gradients(movable_centers, fixed_centers, data_points, metric='euclidean'):
        """
        计算CVT能量和梯度，包含可移动中心点和固定中心点
        
        参数:
            movable_centers: 可移动的中心点（本次采样点），参与梯度优化
            fixed_centers: 固定的中心点（已采样点），不参与梯度优化
            data_points: 所有数据点
            metric: 距离度量方法
        
        返回:
            energy: CVT能量
            gradients: 仅针对可移动中心点的梯度
        """
        # 合并所有中心点进行距离计算
        if fixed_centers is not None and len(fixed_centers) > 0:
            all_centers = np.vstack([movable_centers, fixed_centers])
        else:
            all_centers = movable_centers
        
        # 计算所有数据点到所有中心点的距离
        distances = compute_distances_metric(data_points, all_centers, metric)
        labels = np.argmin(distances, axis=1)
        
        energy = 0
        gradients = np.zeros_like(movable_centers)
        
        # 计算所有中心点的CVT能量
        for i in range(len(all_centers)):
            cluster_points = data_points[labels == i]
            if len(cluster_points) > 0:
                diff = cluster_points - all_centers[i]
                if metric == 'euclidean':
                    energy += np.sum(diff**2)
                elif metric == 'manhattan':
                    energy += np.sum(np.abs(diff))
                elif metric == 'cosine':
                    energy += np.sum(diff**2)
        
        # 只计算可移动中心点的梯度
        for i in range(len(movable_centers)):
            cluster_points = data_points[labels == i]  # 属于第i个可移动中心点的数据点
            if len(cluster_points) > 0:
                diff = cluster_points - movable_centers[i]
                if metric == 'euclidean':
                    gradients[i] = -2 * np.sum(diff, axis=0)
                elif metric == 'manhattan':
                    gradients[i] = -np.mean(np.sign(diff), axis=0)
                elif metric == 'cosine':
                    gradients[i] = -2 * np.sum(diff, axis=0)
        
        return energy, gradients
    
    def compute_repulsion_energy_gradients(centers, repulsion_points, strength, metric='euclidean'):
        """计算排斥能量和梯度"""
        energy = 0
        gradients = np.zeros_like(centers)
        
        for i, center in enumerate(centers):
            for repulsion_point in repulsion_points:
                diff = center - repulsion_point
                
                if metric == 'euclidean':
                    distance_sq = np.sum(diff**2) + 1e-10  # 避免除零
                    # 排斥能量：strength / distance²
                    energy += strength / distance_sq
                    # 排斥梯度：2 * strength * diff / distance⁴
                    gradients[i] += 2 * strength * diff / (distance_sq**2)
                elif metric == 'manhattan':
                    # 曼哈顿距离的排斥力近似
                    manhattan_dist = np.sum(np.abs(diff)) + 1e-10
                    energy += strength / (manhattan_dist**2)
                    # 近似梯度
                    gradients[i] += 2 * strength * np.sign(diff) / (manhattan_dist**3)
                elif metric == 'cosine':
                    # 余弦距离的排斥力：使用欧几里得距离近似
                    distance_sq = np.sum(diff**2) + 1e-10
                    energy += strength / distance_sq
                    gradients[i] += 2 * strength * diff / (distance_sq**2)
        
        return energy, gradients
    
    def compute_total_energy_and_gradients(movable_centers, fixed_centers, data_points, repulsion_points, 
                                         repulsion_strength, cvt_weight, metric):
        """计算总能量和梯度"""
        # CVT能量和梯度
        cvt_energy, cvt_gradients = compute_cvt_energy_gradients(movable_centers, fixed_centers, data_points, metric)
        
        # 排斥能量和梯度
        if repulsion_points is not None and len(repulsion_points) > 0:
            repulsion_energy, repulsion_gradients = compute_repulsion_energy_gradients(
                movable_centers, repulsion_points, repulsion_strength, metric)
        else:
            repulsion_energy = 0
            repulsion_gradients = np.zeros_like(movable_centers)
        
        # 组合能量和梯度
        total_energy = cvt_weight * cvt_energy + repulsion_energy
        total_gradients = cvt_weight * cvt_gradients + repulsion_gradients
        
        return total_energy, total_gradients
    
    # 1. 计算data中两点间的最大距离，用于确定排斥函数系数
    all_data_features = data.drop(not_feature_columns, axis=1).values
    print("计算数据中两点间的最大距离...")
    
    # 计算完整距离矩阵并找出最大值
    distance_matrix = compute_pairwise_distances(all_data_features, all_data_features, distance_metric)
    max_distance = np.max(distance_matrix)
    
    print(f"数据中最大距离: {max_distance:.6f}")
    
    # 2. 计算均匀采样下采样点间的期望平方距离
    # 假设数据在一个d维超立方体中均匀分布，采样k个点
    # 期望距离的平方大约为 (max_distance^2 * d / 12) 其中d是维数
    d = all_data_features.shape[1]  # 特征维数
    expected_sq_distance = (max_distance**2) * d / 12
    print(f"均匀采样下采样点间期望平方距离: {expected_sq_distance:.6f}")
    
    # 3. 计算排斥函数系数，使排斥力和CVT能量在同一数量级
    # CVT能量的量级大约是expected_sq_distance * 数据点数量
    # 排斥能量的量级大约是repulsion_coefficient / expected_sq_distance
    # 令两者相等：repulsion_coefficient / expected_sq_distance ≈ expected_sq_distance * n_points
    n_points = len(all_data_features)
    base_repulsion_coefficient = expected_sq_distance**2 * n_points
    
    # 4. 与10^repulsion_strength相乘得到最终系数
    final_repulsion_strength = base_repulsion_coefficient * (10 ** repulsion_strength)
    
    print(f"基础排斥系数: {base_repulsion_coefficient:.6e}")
    print(f"最终排斥强度 (base * 10^{repulsion_strength}): {final_repulsion_strength:.6e}")
    
    # 5. 数据预处理：准备CVT和排斥计算所需的不同数据集
    if sampled_data is not None:
        # 1.1 未采样数据（用于本次采样）：去除所有ScreenLabel不为'BASE'的点
        available_data = data[data['ScreenLabel'] == 'BASE'].copy()
        
        # 1.2 所有已采样点特征（用于CVT计算中的固定中心点）
        all_sampled_features = sampled_data.drop(not_feature_columns, axis=1).values
        print(f"找到 {len(all_sampled_features)} 个已采样点作为CVT固定中心点")
        
        # 1.3 仅ScreenLabel为'Excluded_Sampled'的点（用于排斥力计算）
        if 'ScreenLabel' in sampled_data.columns:
            excluded_sampled_data = sampled_data[sampled_data['ScreenLabel'] == 'Excluded_Sampled']
            if len(excluded_sampled_data) > 0:
                repulsion_features = excluded_sampled_data.drop(not_feature_columns, axis=1).values
                print(f"使用 {len(excluded_sampled_data)} 个ScreenLabel='Excluded_Sampled'的点计算排斥力")
            else:
                repulsion_features = None
                print("没有找到ScreenLabel='Excluded_Sampled'的点，不计算排斥力")
        else:
            # 如果没有ScreenLabel列，不计算排斥力
            repulsion_features = None
            print("未找到ScreenLabel列，不计算排斥力")
        
        # 1.4 所有数据点（用于CVT距离计算）
        all_data_features = data.drop(not_feature_columns, axis=1).values
        print(f"CVT计算将使用所有 {len(all_data_features)} 个数据点")
    else:
        available_data = data.copy()
        all_sampled_features = None
        repulsion_features = None
        all_data_features = data.drop(not_feature_columns, axis=1).values
        print("无已采样数据，将进行标准CVT采样")
    
    # 2. 提取特征数据
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. 使用解析CVT方法获得初始猜测
    print(f"阶段1：使用解析CVT方法初始化中心点（迭代{cvt_init_iters}次）")
    
    # 3.1 随机初始化中心点
    indices = np.random.choice(len(X), k, replace=False)
    centers = X[indices].copy().astype(float)
    
    # 3.2 解析CVT优化（只考虑可移动的数据点，不包括排斥力）
    cvt_data_points = X  # 用于CVT计算的数据点
    
    for cvt_iter in range(cvt_init_iters):
        # 处理NAN
        if np.isnan(centers).any():
            centers[np.isnan(centers)] = np.mean(centers[~np.isnan(centers)])
        
        # 如果有固定中心点，需要与可移动中心点合并进行Voronoi划分
        if all_sampled_features is not None and len(all_sampled_features) > 0:
            all_centers_for_voronoi = np.vstack([centers, all_sampled_features])
        else:
            all_centers_for_voronoi = centers
        
        # Voronoi划分：计算数据点到所有中心点的距离
        distances = compute_distances_metric(cvt_data_points, all_centers_for_voronoi, distance_metric)
        labels = np.argmin(distances, axis=1)
        
        # 更新可移动中心点为各自Voronoi单元的质心（解析解）
        new_centers = np.zeros_like(centers)
        for i in range(k):
            # 获取属于第i个可移动中心点的数据点
            cluster_points = cvt_data_points[labels == i]
            if len(cluster_points) > 0:
                new_centers[i] = compute_cluster_centroid(cluster_points, distance_metric)
            else:
                # 如果某个中心点没有分配到任何数据点，保持不变
                new_centers[i] = centers[i]
        
        # 检查收敛
        if np.linalg.norm(new_centers - centers) < tol:
            print(f"解析CVT收敛，迭代次数: {cvt_iter}")
            break
        centers = new_centers
    
    print(f"阶段2：基于解析CVT初猜进行加权能量优化（最多{max_iters}次迭代）")
    
    # 4. 能量最小化迭代
    prev_energy = float('inf')
    lr = learning_rate
    
    print(f"开始加权能量优化，排斥强度: {final_repulsion_strength:.6e}, CVT权重: {cvt_weight}")
    
    for iteration in range(max_iters):
        # 计算总能量和梯度
        total_energy, gradients = compute_total_energy_and_gradients(
            centers, all_sampled_features, X, repulsion_features, final_repulsion_strength, cvt_weight, distance_metric)
        
        # 梯度下降更新
        centers_new = centers - lr * gradients
        
        # 自适应学习率
        if adaptive_lr:
            new_energy, _ = compute_total_energy_and_gradients(
                centers_new, all_sampled_features, X, repulsion_features, final_repulsion_strength, cvt_weight, distance_metric)
            
            if new_energy > total_energy:  # 能量增加，减小学习率
                lr *= 0.8
                if lr < learning_rate * 1e-4:  # 学习率过小，停止
                    print(f"学习率过小，提前停止。迭代次数: {iteration}")
                    break
                continue
            elif abs(new_energy - prev_energy) < tol * abs(prev_energy) if prev_energy != 0 else tol:
                print(f"加权能量优化收敛, 迭代次数: {iteration}, 最终能量: {new_energy:.6f}")
                break
            else:
                lr = min(lr * 1.05, learning_rate)  # 逐渐恢复学习率
        
        # 更新中心点
        centers = centers_new
        prev_energy = total_energy
        
        # 定期输出进度
        if iteration % 100 == 0:
            print(f"迭代 {iteration}: 总能量 = {total_energy:.6f}, 学习率 = {lr:.6f}")
    
    # 5. 找到距离最终中心点最近的实际数据点
    final_distances = compute_distances_metric(X, centers, distance_metric)
    
    # 使用贪心算法确保每个数据点最多被选择一次
    selected_indices = []
    used_indices = set()
    
    for center_idx in range(k):
        distances_to_center = final_distances[:, center_idx]
        candidate_indices = np.argsort(distances_to_center)
        
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            raise ValueError(f"无法找到足够的不重复采样点。需要{k}个点，但只能找到{len(selected_indices)}个不重复的点")
    
    selected_indices = np.array(selected_indices)
    
    # 6. 构建返回的DataFrame
    result_centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), result_centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"加权CVT采样完成，采样了 {len(result_centers)} 个点")
    return result_centers, unselected_points