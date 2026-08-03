import warnings
from deprecated.sphinx import deprecated
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from scipy.stats.qmc import LatinHypercube, Sobol
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cosine

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

def cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=500, tol=1e-5, sampled_data=None, distance_metric='euclidean'):
    """
    CVT采样算法（支持DataFrame输入，保留非数值列，排除已采样数据）
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        max_iters: 最大迭代次数
        tol: 中心点变化的收敛阈值
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
        distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于CVT计算
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. 初始化中心点（从可用数据的数值特征中随机选择k个）
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for iteration in range(max_iters):
        # 处理NAN
        if np.isnan(centers_X).any():
            centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
        # 4. Voronoi划分：计算可用点到中心点的距离
        distances = compute_pairwise_distances(X, centers_X, distance_metric)

        labels = np.argmin(distances, axis=1)
        
        # 5. 更新中心点为Voronoi单元的质心
        new_centers_X = np.zeros_like(centers_X)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers_X[i] = compute_cluster_centroid(cluster_points, distance_metric)
            else:
                # 如果某个中心点没有分配到任何数据点，保持不变
                new_centers_X[i] = centers_X[i]
        
        # 6. 检查收敛
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT算法收敛, 迭代次数: {iteration}")
            break
        centers_X = new_centers_X
    
    # 7. 找到距离中心点最近的可用数据点作为最终采样点（确保不重复）
    if np.isnan(centers_X).any():
        centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
    final_distances = compute_pairwise_distances(X, centers_X, distance_metric)
    
    # 使用贪心算法确保每个数据点最多被选择一次
    selected_indices = []
    used_indices = set()
    
    # 为每个中心点找到最近且未被使用的数据点
    for center_idx in range(k):
        # 获取当前中心点到所有数据点的距离
        distances_to_center = final_distances[:, center_idx]
        # 按距离排序获得候选索引
        candidate_indices = np.argsort(distances_to_center)
        
        # 找到第一个未被使用的数据点
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            # 如果所有候选点都被使用了，说明可用数据点不足
            raise ValueError(f"无法找到足够的不重复采样点。需要{k}个点，但只能找到{len(selected_indices)}个不重复的点")
    
    selected_indices = np.array(selected_indices)
    
    # 8. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def cvt_sampling_gold_df_norepeat(data, k, not_feature_columns, max_iters=500, tol=1e-5, sampled_data=None, distance_metric='euclidean'):
    """
    CVT采样算法（支持DataFrame输入，保留非数值列，排除已采样数据，直接使用虚拟中心点作为采样点）
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        max_iters: 最大迭代次数
        tol: 中心点变化的收敛阈值
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
        distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
        centers: DataFrame，虚拟采样中心点（包含所有列，非特征列使用占位符）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于CVT计算
    feature_columns = [col for col in data.columns if col not in not_feature_columns]
    X = available_data[feature_columns].values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. 初始化中心点（从可用数据的数值特征中随机选择k个）
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for iteration in range(max_iters):
        # 处理NAN
        if np.isnan(centers_X).any():
            centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
        # 4. Voronoi划分：计算可用点到中心点的距离
        distances = compute_pairwise_distances(X, centers_X, distance_metric)

        labels = np.argmin(distances, axis=1)
        
        # 5. 更新中心点为Voronoi单元的质心
        new_centers_X = np.zeros_like(centers_X)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers_X[i] = compute_cluster_centroid(cluster_points, distance_metric)
            else:
                # 如果某个中心点没有分配到任何数据点，保持不变
                new_centers_X[i] = centers_X[i]
        
        # 6. 检查收敛
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT算法收敛, 迭代次数: {iteration}")
            break
        centers_X = new_centers_X
    
    # 7. 直接使用收敛的中心点作为虚拟采样点，为非特征列生成占位符
    if np.isnan(centers_X).any():
        centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
    
    # 8. 创建虚拟采样点DataFrame
    virtual_centers_dict = {}
    
    # 添加特征列（使用收敛的中心点值）
    for i, col in enumerate(feature_columns):
        virtual_centers_dict[col] = centers_X[:, i]
    
    # 为非特征列生成占位符
    for col in not_feature_columns:
        if col in data.columns:
            if data[col].dtype == 'object' or pd.api.types.is_string_dtype(data[col]):
                # 字符串类型：使用 'virtual_center_X' 格式
                virtual_centers_dict[col] = [f'virtual_center_{i}' for i in range(k)]
            elif pd.api.types.is_numeric_dtype(data[col]):
                # 数值类型：使用特征的平均值或其他合理默认值
                virtual_centers_dict[col] = [data[col].mean()] * k
            else:
                # 其他类型：使用字符串占位符
                virtual_centers_dict[col] = [f'virtual_center_{i}' for i in range(k)]
    
    # 创建虚拟中心点DataFrame
    centers = pd.DataFrame(virtual_centers_dict)
    
    # 9. unselected_points设为所有原始数据点（因为我们使用的是虚拟中心）
    unselected_points = data.copy()
    
    return centers, unselected_points

def pam_sampling_df_norepeat(data, k, not_feature_columns, max_iters=500, sampled_data=None, random_state=None):
    """
    PAM（K-Medoids）采样算法（支持DataFrame输入，保留非数值列，排除已采样数据）
    使用原生实现替代sklearn-extra
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        max_iters: 最大迭代次数
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
        random_state: int，随机种子
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于PAM计算
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. 初始化medoids（从可用数据中随机选择k个）
    medoid_indices = np.random.choice(len(X), k, replace=False)
    
    # 4. 计算所有点之间的距离矩阵
    distance_matrix = pairwise_distances(X, X)
    
    for i in range(max_iters):
        # 5. 分配每个点到最近的medoid
        medoid_distances = distance_matrix[:, medoid_indices]
        labels = np.argmin(medoid_distances, axis=1)
        
        # 6. 尝试更新每个medoid
        new_medoid_indices = medoid_indices.copy()
        
        for cluster_idx in range(k):
            cluster_points = np.where(labels == cluster_idx)[0]
            if len(cluster_points) == 0:
                continue
                
            current_medoid = medoid_indices[cluster_idx]
            current_cost = np.sum(distance_matrix[cluster_points][:, current_medoid])
            
            # 寻找该簇中总距离最小的点作为新medoid
            best_medoid = current_medoid
            best_cost = current_cost
            
            for candidate in cluster_points:
                if candidate in medoid_indices:  # 跳过已经是medoid的点
                    continue
                candidate_cost = np.sum(distance_matrix[cluster_points][:, candidate])
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_medoid = candidate
            
            new_medoid_indices[cluster_idx] = best_medoid
        
        # 7. 检查收敛
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        
        medoid_indices = new_medoid_indices
    
    # 8. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[medoid_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    return centers, unselected_points

def fps_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, random_state=None, distance_metric='euclidean'):
    """
    FPS（最远点采样/Farthest Point Sampling）算法（支持DataFrame输入，保留非数值列，排除已采样数据）
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
        random_state: int，随机种子
        distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于FPS计算
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. FPS算法实现
    selected_indices = []
    
    # 随机选择第一个点
    first_idx = np.random.randint(0, len(X))
    selected_indices.append(first_idx)
    
    # 计算所有点之间的距离矩阵（只计算一次）
    distance_matrix = compute_pairwise_distances(X, X, distance_metric)
    
    # 迭代选择剩余的k-1个点
    for i in range(k - 1):
        # 计算每个未选择点到所有已选择点的最小距离
        min_distances = np.full(len(X), np.inf)
        
        for selected_idx in selected_indices:
            distances_to_selected = distance_matrix[:, selected_idx]
            min_distances = np.minimum(min_distances, distances_to_selected)
        
        # 将已选择的点的距离设为0，避免重复选择
        min_distances[selected_indices] = 0
        
        # 选择距离最远的点
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
    
    selected_indices = np.array(selected_indices)
    
    # 4. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    return centers, unselected_points

def lhs_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    使用拉丁超立方采样(LHS)从DataFrame中选取样本点
    
    参数:
    data: pd.DataFrame - 包含所有数据的DataFrame
    k: int - 要采样的中心点数量
    not_feature_columns: list - 不作为特征的列名列表
    sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
    distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
    centers: pd.DataFrame - 采样到的中心点(包含原始所有列)
    unselected_points: pd.DataFrame - 未被采样的点(包含原始所有列)
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于LHS计算
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 使用scipy库的LHS采样
    sampler = LatinHypercube(d=X.shape[1])
    samples = sampler.random(n=k)
    
    # 将LHS样本映射到特征空间
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    lhs_points = samples * (max_vals - min_vals) + min_vals
    
    # 找到距离LHS点最近的可用数据点作为最终采样点（确保不重复）
    if np.isnan(lhs_points).any():
        lhs_points[np.isnan(lhs_points)] = np.mean(lhs_points[~np.isnan(lhs_points)])
    final_distances = compute_pairwise_distances(X, lhs_points, distance_metric)
    
    # 使用贪心算法确保每个数据点最多被选择一次
    selected_indices = []
    used_indices = set()
    
    # 为每个LHS点找到最近且未被使用的数据点
    for lhs_idx in range(k):
        # 获取当前LHS点到所有数据点的距离
        distances_to_lhs = final_distances[:, lhs_idx]
        # 按距离排序获得候选索引
        candidate_indices = np.argsort(distances_to_lhs)
        
        # 找到第一个未被使用的数据点
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            # 如果所有候选点都被使用了，说明可用数据点不足
            raise ValueError(f"无法找到足够的不重复采样点。需要{k}个点，但只能找到{len(selected_indices)}个不重复的点")
    
    selected_indices = np.array(selected_indices)
    
    # 8. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def rand_sampling_df_no_repeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    使用随机采样从DataFrame中选取样本点
    
    参数:
    data: pd.DataFrame - 包含所有数据的DataFrame
    k: int - 要采样的中心点数量
    not_feature_columns: list - 不作为特征的列名列表
    sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
    distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
    centers: pd.DataFrame - 采样到的中心点(包含原始所有列)
    unselected_points: pd.DataFrame - 未被采样的点(包含原始所有列)
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()

    feature_columns = [col for col in available_data.columns if col not in not_feature_columns]
    
    # 确保所有特征都是数值型
    if not all(pd.api.types.is_numeric_dtype(available_data[col]) for col in feature_columns):
        raise ValueError("所有特征列必须是数值类型")
    
    # 随机选择k个索引
    if k > len(available_data):
        raise ValueError(f"采样数量 {k} 不能大于数据集大小 {len(data)}")
    
    center_indices = np.random.choice(len(available_data), k, replace=False)
    
    # 构建返回结果
    centers = available_data.iloc[center_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), center_indices)
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

def sobol_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    使用Sobol序列采样从DataFrame中选取样本点
    
    参数:
    data: pd.DataFrame - 包含所有数据的DataFrame
    k: int - 要采样的中心点数量
    not_feature_columns: list - 不作为特征的列名列表
    sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
    distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
    centers: pd.DataFrame - 采样到的中心点(包含原始所有列)
    unselected_points: pd.DataFrame - 未被采样的点(包含原始所有列)
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于Sobol采样
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 使用scipy库的Sobol采样
    sampler = Sobol(d=X.shape[1])
    samples = sampler.random(n=k)
    
    # 将Sobol样本映射到特征空间
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    sobol_points = samples * (max_vals - min_vals) + min_vals
    
    # 找到距离Sobol点最近的可用数据点作为最终采样点（确保不重复）
    if np.isnan(sobol_points).any():
        sobol_points[np.isnan(sobol_points)] = np.mean(sobol_points[~np.isnan(sobol_points)])
    final_distances = compute_pairwise_distances(X, sobol_points, distance_metric)
    
    # 使用贪心算法确保每个数据点最多被选择一次
    selected_indices = []
    used_indices = set()
    
    # 为每个Sobol点找到最近且未被使用的数据点
    for sobol_idx in range(k):
        # 获取当前Sobol点到所有数据点的距离
        distances_to_sobol = final_distances[:, sobol_idx]
        # 按距离排序获得候选索引
        candidate_indices = np.argsort(distances_to_sobol)
        
        # 找到第一个未被使用的数据点
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            # 如果所有候选点都被使用了，说明可用数据点不足
            raise ValueError(f"无法找到足够的不重复采样点。需要{k}个点，但只能找到{len(selected_indices)}个不重复的点")
    
    selected_indices = np.array(selected_indices)
    
    # 8. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def itr_cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=500, tol=1e-5, sampled_data=None, distance_metric='euclidean'):
    """
    迭代CVT采样算法（支持DataFrame输入，保留非数值列，排除已采样数据，已采样点参与距离计算但不可移动）
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        max_iters: 最大迭代次数
        tol: 中心点变化的收敛阈值
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index），这些点参与距离计算但不可移动
        distance_metric: str，距离度量方法，支持'euclidean'和'manhattan'
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
        
        # 提取已采样数据的特征用于距离计算
        sampled_features = sampled_data.drop(not_feature_columns, axis=1).values
    else:
        available_data = data.copy()
        sampled_features = None
    
    # 2. 提取可用数据的数值特征用于CVT计算
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. 初始化中心点（从可用数据的数值特征中随机选择k个）
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    # 4. 如果有已采样数据，将其与新中心点合并用于距离计算
    if sampled_features is not None:
        all_centers_X = np.vstack([sampled_features, centers_X])
        n_sampled = len(sampled_features)
    else:
        all_centers_X = centers_X
        n_sampled = 0
    
    # 5. 定义距离计算函数
    def compute_distances(points, centers, metric):
        if metric == 'euclidean':
            return pairwise_distances(points, centers, metric='euclidean')
        elif metric == 'manhattan':
            return pairwise_distances(points, centers, metric='manhattan')
        else:
            raise ValueError(f"不支持的距离度量: {metric}. 支持的度量: 'euclidean', 'manhattan'")
    
    for iteration in range(max_iters):
        # 处理NAN
        if np.isnan(all_centers_X).any():
            all_centers_X[np.isnan(all_centers_X)] = np.mean(all_centers_X[~np.isnan(all_centers_X)])
        
        # 6. Voronoi划分：计算可用点到所有中心点（包括已采样点）的距离
        distances = compute_distances(X, all_centers_X, distance_metric)
        labels = np.argmin(distances, axis=1)
        
        # 7. 更新中心点为Voronoi单元的质心（只更新新采样的中心点，不更新已采样点）
        new_centers_X = centers_X.copy()
        
        for i in range(k):
            center_idx = n_sampled + i  # 在all_centers_X中的索引
            cluster_points = X[labels == center_idx]
            if len(cluster_points) > 0:
                new_centers_X[i] = compute_cluster_centroid(cluster_points, distance_metric)
        
        # 8. 检查收敛（只检查新采样中心点的变化）
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"迭代CVT算法收敛, 迭代次数: {iteration}")
            break
        
        # 更新中心点
        centers_X = new_centers_X
        if sampled_features is not None:
            all_centers_X = np.vstack([sampled_features, centers_X])
        else:
            all_centers_X = centers_X
    
    # 9. 找到距离最终中心点最近的可用数据点作为最终采样点（确保不重复）
    if np.isnan(all_centers_X).any():
        all_centers_X[np.isnan(all_centers_X)] = np.mean(all_centers_X[~np.isnan(all_centers_X)])
    
    # 只考虑新采样的中心点进行最终点选择
    final_distances = compute_distances(X, centers_X, distance_metric)
    
    # 使用贪心算法确保每个数据点最多被选择一次
    selected_indices = []
    used_indices = set()
    
    # 为每个中心点找到最近且未被使用的数据点
    for center_idx in range(k):
        # 获取当前中心点到所有数据点的距离
        distances_to_center = final_distances[:, center_idx]
        # 按距离排序获得候选索引
        candidate_indices = np.argsort(distances_to_center)
        
        # 找到第一个未被使用的数据点
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            # 如果所有候选点都被使用了，说明可用数据点不足
            raise ValueError(f"无法找到足够的不重复采样点。需要{k}个点，但只能找到{len(selected_indices)}个不重复的点")
    
    selected_indices = np.array(selected_indices)
    
    # 10. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def weighted_itr_cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=2000, tol=1e-5, 
                                        sampled_data=None, repulsion_strength=1.0, cvt_weight=1.0,
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
    
    # 1. 数据预处理：准备CVT和排斥计算所需的不同数据集
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 1.1 未采样数据（用于本次采样）
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
        
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
    
    print(f"开始加权能量优化，排斥强度: {repulsion_strength}, CVT权重: {cvt_weight}")
    
    for iteration in range(max_iters):
        # 计算总能量和梯度
        total_energy, gradients = compute_total_energy_and_gradients(
            centers, all_sampled_features, X, repulsion_features, repulsion_strength, cvt_weight, distance_metric)
        
        # 梯度下降更新
        centers_new = centers - lr * gradients
        
        # 自适应学习率
        if adaptive_lr:
            new_energy, _ = compute_total_energy_and_gradients(
                centers_new, all_sampled_features, X, repulsion_features, repulsion_strength, cvt_weight, distance_metric)
            
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


def get_cvt_sampling_norepeat(task_itr_id, drop_classes, not_feature_cols, k, max_iters=500, tol=1e-4, sampled_data=None):
    '''
    进行一次采样与按采样分类。
    
    参数:
        task_itr_id: int，迭代次数(从1开始)
        drop_classes: list，需要排除的类别
        not_feature_cols: list，非特征列的列名
        k: int，采样中心点数量
        max_iters: int，最大迭代次数
        tol: float，中心点变化的收敛阈值
        sampled_data: DataFrame，已采样的数据（包含原始所有列，类别为sampled_points索引）
    
    返回:
        sampled_points: DataFrame，采样到的中心点（包含原始所有列，即其中的label为上一迭代的分类）
        labeled_points: DataFrame，按采样分类后的数据和类别（包含原始所有列，类别为sampled_points索引）
    '''
    data = pd.read_csv(f'./itr/labeled_points_itr{task_itr_id-1}.csv')
    
    if drop_classes != []:
        for drop_class in drop_classes:
            data = data[data['labels'] != drop_class].reset_index(drop=True)
    # 执行CVT采样
    sampled_points, unsampled_points = cvt_sampling_df_norepeat(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        max_iters=max_iters,
        tol=tol,
        sampled_data=sampled_data
    )
    # sampled_points.to_csv(f'./itr/sampled_points_itr{task_itr_id}.csv', index=False)
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols)
    labeled_points.to_csv(f'./itr/labeled_points_itr{task_itr_id}.csv', index=False)
    return sampled_points, labeled_points


def get_sampling_lhs_norepeat(task_itr_id, drop_classes, not_feature_cols, k, sampled_data=None):
    '''
    进行一次采样与按采样分类。
    
    参数:
        task_itr_id: int，迭代次数(从1开始)
        drop_classes: list，需要排除的类别
        not_feature_cols: list，非特征列的列名
        k: int，采样中心点数量
        sampled_data: DataFrame，已采样的数据（包含原始所有列，类别为sampled_points索引）
    
    返回:
        sampled_points: DataFrame，采样到的中心点（包含原始所有列，即其中的label为上一迭代的分类）
        labeled_points: DataFrame，按采样分类后的数据和类别（包含原始所有列，类别为sampled_points索引）
    '''
    data = pd.read_csv(f'./itr/labeled_points_itr{task_itr_id-1}.csv')
    
    if drop_classes != []:
        for drop_class in drop_classes:
            data = data[data['labels'] != drop_class].reset_index(drop=True)
    # 执行LHS采样
    sampled_points, unsampled_points = lhs_sampling_df_norepeat(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        sampled_data=sampled_data
    )
    sampled_points.to_csv(f'./itr/sampled_points_itr{task_itr_id}.csv', index=False)
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols)
    labeled_points.to_csv(f'./itr/labeled_points_itr{task_itr_id}.csv', index=False)
    return sampled_points, labeled_points

def kennard_stone_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean', random_state=None):
    """
    Kennard-Stone采样算法（支持DataFrame输入，保留非数值列，排除已采样数据）
    
    Kennard-Stone算法是一种基于距离的采样方法，通过选择样本空间中彼此距离最远的点来确保样本的代表性。
    算法步骤：
    1. 选择距离最远的两个点作为初始点
    2. 对于剩余的每个点，选择距离已选择点集合最远的点
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
        distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
        random_state: int，随机种子，用于距离相等时的随机选择
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于Kennard-Stone计算
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    if k < 2:
        raise ValueError("Kennard-Stone采样需要至少2个采样点")
    
    # 3. 计算所有点之间的距离矩阵
    distance_matrix = compute_pairwise_distances(X, X, distance_metric)
    
    # 4. Kennard-Stone算法实现
    selected_indices = []
    remaining_indices = list(range(len(X)))
    
    # 步骤1：选择距离最远的两个点作为初始点
    max_distance = 0
    initial_pair = (0, 1)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if distance_matrix[i, j] > max_distance:
                max_distance = distance_matrix[i, j]
                initial_pair = (i, j)
    
    # 添加初始的两个点
    selected_indices.extend([initial_pair[0], initial_pair[1]])
    remaining_indices.remove(initial_pair[0])
    remaining_indices.remove(initial_pair[1])
    
    # 步骤2：迭代选择剩余的k-2个点
    for _ in range(k - 2):
        max_min_distance = -1
        best_candidate = None
        
        # 对每个剩余点，计算其到已选择点的最小距离
        for candidate_idx in remaining_indices:
            min_distance_to_selected = float('inf')
            
            # 找到当前候选点到已选择点集合的最小距离
            for selected_idx in selected_indices:
                distance = distance_matrix[candidate_idx, selected_idx]
                min_distance_to_selected = min(min_distance_to_selected, distance)
            
            # 选择具有最大的"最小距离"的点
            if min_distance_to_selected > max_min_distance:
                max_min_distance = min_distance_to_selected
                best_candidate = candidate_idx
            elif min_distance_to_selected == max_min_distance and best_candidate is not None:
                # 距离相等时随机选择
                if np.random.random() < 0.5:
                    best_candidate = candidate_idx
        
        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
        else:
            # 如果没找到合适的候选点，随机选择一个
            if remaining_indices:
                random_choice = np.random.choice(remaining_indices)
                selected_indices.append(random_choice)
                remaining_indices.remove(random_choice)
    
    selected_indices = np.array(selected_indices)
    
    # 5. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"Kennard-Stone采样完成，采样了 {len(centers)} 个点")
    return centers, unselected_points

def ward_clustering_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    Ward层次聚类采样算法（支持DataFrame输入，保留非数值列，排除已采样数据）
    使用Ward连通性标准进行聚集分层聚类，选择每个聚类的中心点作为代表
    
    参数:
        data: DataFrame，包含特征列和非数值列（如'reactant_aldehyde', 'conv'）
        k: 需要采样的中心点数量
        not_feature_columns: list，不参与距离计算的特征列名（数值型）
        sampled_data: DataFrame，已采样的数据点（格式与data一致，包含未reset的index）
        distance_metric: str，距离度量方法，支持'euclidean'、'manhattan'、'cosine'
    
    返回:
        centers: DataFrame，采样到的中心点（包含原始所有列）
        unselected_points: DataFrame，未被采样的点（原始所有列）
    """
    # 1. 如果有已采样数据，从候选池中排除这些点（仅匹配not_feature_columns）
    if sampled_data is not None:
        # 创建一个标记列来识别重复项
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        # 在data中标记哪些行与sampled_data重复
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        # 遍历sampled_keys，标记重复的行
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        # 保留未被采样的数据
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    # 2. 提取可用数据的数值特征用于聚类
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"可用数据点数量({len(X)})少于要求的采样数量({k})")
    
    # 3. 设置聚类算法的距离度量
    # Ward算法要求使用欧几里得距离，对于其他距离度量使用complete linkage
    if distance_metric == 'euclidean':
        linkage = 'ward'
        metric = 'euclidean'
    elif distance_metric == 'manhattan':
        linkage = 'complete'
        metric = 'manhattan'
    elif distance_metric == 'cosine':
        linkage = 'complete'
        metric = 'cosine'
    else:
        raise ValueError(f"不支持的距离度量: {distance_metric}. 支持的度量: 'euclidean', 'manhattan', 'cosine'")
    
    print(f"使用 {linkage} linkage 和 {metric} 距离度量进行层次聚类")
    
    # 4. 执行层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric
    )
    
    # 处理NaN值
    if np.isnan(X).any():
        # 简单处理：用列均值填充NaN
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    cluster_labels = clustering.fit_predict(X)
    
    # 5. 为每个聚类选择最具代表性的点（距离聚类中心最近的点）
    selected_indices = []
    used_indices = set()
    
    for cluster_id in range(k):
        # 获取当前聚类的所有点
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_points = X[cluster_indices]
        
        # 计算聚类中心
        cluster_center = np.mean(cluster_points, axis=0)
        
        # 找到距离聚类中心最近且未被使用的点
        distances_to_center = compute_pairwise_distances(
            cluster_points, cluster_center.reshape(1, -1), distance_metric
        ).flatten()
        
        # 按距离排序获得候选索引
        sorted_cluster_indices = cluster_indices[np.argsort(distances_to_center)]
        
        # 找到第一个未被使用的数据点
        for candidate_idx in sorted_cluster_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            # 如果所有候选点都被使用了，选择距离中心最近的点（允许重复）
            best_idx = sorted_cluster_indices[0]
            selected_indices.append(best_idx)
    
    # 确保我们有足够的采样点
    if len(selected_indices) < k:
        # 如果聚类数量不足，从剩余点中随机选择
        remaining_indices = np.setdiff1d(np.arange(len(X)), selected_indices)
        additional_needed = k - len(selected_indices)
        if len(remaining_indices) >= additional_needed:
            additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
            selected_indices.extend(additional_indices)
        else:
            # 如果还是不够，允许重复选择
            additional_indices = np.random.choice(len(X), additional_needed, replace=True)
            selected_indices.extend(additional_indices)
    
    selected_indices = np.array(selected_indices[:k])  # 确保正好k个点
    
    # 6. 构建返回的DataFrame（从可用数据中选择）
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"Ward聚类采样完成，采样了 {len(centers)} 个点")
    return centers, unselected_points

if __name__ == '__main__':
    np.random.seed(42)
    data = pd.read_csv('test_quchong.csv')
    previous_data = pd.read_csv('test_sampled.csv')
    sampled_data, unsampled_data = cvt_sampling_df_norepeat(data=data, k=20, not_feature_columns=['smiles', 'conv'], sampled_data=previous_data)
    print(sampled_data)
    # sampled_data, unsampled_data = lhs_sampling_df_norepeat(data=data, k=20, not_feature_columns=['smiles', 'conv'], sampled_data=previous_data)
    # print(sampled_data)
    # sampled_data, unsampled_data = rand_sampling_df_no_repeat(data=data, k=20, not_feature_columns=['smiles', 'conv'], sampled_data=previous_data)
    # print(sampled_data)