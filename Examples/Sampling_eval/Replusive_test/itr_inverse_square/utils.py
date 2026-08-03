import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from sampling import cvt_sampling_df_norepeat, rand_sampling_df_no_repeat, cvt_sampling_gold_df_norepeat, fps_sampling_df_norepeat, sobol_sampling_df_norepeat
import tqdm


def entropy(data, k=5, distance_metric='euclidean'):
    """
    使用K近邻方法估计numpy数组的空间熵。
    
    参数:
    -----------
    data : numpy.ndarray
        包含空间数据点的输入numpy数组
    k : int, 默认=5
        考虑的最近邻数量
    distance_metric : str, 默认='euclidean'
        KNN计算的距离度量，支持'euclidean', 'manhattan', 'cosine', 'tanimoto'
        
    返回:
    --------
    float
        估计的空间熵值
    """
    if data.size == 0 or len(data) < k + 1:
        return 0.0
    
    if data.shape[1] == 0:
        raise ValueError("数据必须包含数值列才能计算空间熵")
    
    n_points = len(data)
    
    # Tanimoto相似度需要预计算距离矩阵
    if distance_metric == 'tanimoto':
        # 计算Tanimoto距离矩阵（使用Jaccard距离作为近似）
        distance_matrix = squareform(pdist(data, metric='jaccard'))
        
        # 为每个点找到k个最近邻（排除点本身）
        distances = []
        for i in range(n_points):
            # 获取第i个点到所有其他点的距离
            dists = distance_matrix[i, :]
            # 排除到自身的距离
            dists = np.concatenate([dists[:i], dists[i+1:]])
            # 按距离排序并选择最近的k个
            k_nearest = np.sort(dists)[:k]
            distances.append(k_nearest)
        distances = np.array(distances)
    else:
        # 使用NearestNeighbors处理其他距离度量
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=distance_metric)
        nbrs.fit(data)
        
        # 为每个点找到k个最近邻（排除点本身）
        distances, indices = nbrs.kneighbors(data)
        
        # 移除第一列（到自身的距离 = 0）
        distances = distances[:, 1:]
    
    # 计算每个点到第k近邻的距离
    kth_distances = distances[:, -1]
    
    # 通过添加小的epsilon避免log(0)
    epsilon = 1e-10
    kth_distances = np.maximum(kth_distances, epsilon)
    
    # 使用Kozachenko-Leonenko估计器计算熵
    # H = log(n-1) - digamma(k) + log(2) * d + (d/n) * sum(log(r_k))
    # 简化版本: H = log(n) + log(volume_factor) + (1/n) * sum(log(distances))
    
    dimension = data.shape[1]
    
    # d维单位球体积: V_d = π^(d/2) / Γ(d/2 + 1)
    # 为简化，我们使用直接估计方法
    log_distances = np.log(kth_distances)
    
    # Kozachenko-Leonenko熵估计器
    entropy_estimate = (
        np.log(n_points - 1) - 
        np.log(k) + 
        dimension * np.log(2) + 
        dimension * np.mean(log_distances)
    )
    
    return entropy_estimate


def mean_squared_distance_to_nearest(complete_data, sampling_data, distance_metric='euclidean'):
    """
    计算完整数据集中每个点到最近采样点的平均平方距离。
    
    参数:
    -----------
    complete_data : numpy.ndarray
        完整数据集的numpy数组
    sampling_data : numpy.ndarray
        采样点的numpy数组
    distance_metric : str, 默认='euclidean'
        距离计算的度量方法
        
    返回:
    --------
    float
        完整数据集点到最近采样点的平均平方距离
    """
    if complete_data.size == 0 or sampling_data.size == 0:
        return 0.0
    
    if complete_data.shape[1] == 0 or sampling_data.shape[1] == 0:
        raise ValueError("数据必须包含数值列才能计算距离")
    
    # 计算完整数据集每个点到所有采样点的距离
    distance_matrix = cdist(complete_data, sampling_data, metric=distance_metric)
    
    # 对每个完整数据集的点，找到到采样点的最小距离
    min_distances = np.min(distance_matrix, axis=1)
    
    # 计算平方距离的平均值
    mean_squared_distance = np.mean(min_distances ** 2)
    
    return mean_squared_distance


def mean_distance_to_nearest(complete_data, sampling_data, distance_metric='euclidean'):
    """
    计算完整数据集中每个点到最近采样点的平均距离。
    
    参数:
    -----------
    complete_data : numpy.ndarray
        完整数据集的numpy数组
    sampling_data : numpy.ndarray
        采样点的numpy数组
    distance_metric : str, 默认='euclidean'
        距离计算的度量方法
        
    返回:
    --------
    float
        完整数据集点到最近采样点的平均距离
    """
    if complete_data.size == 0 or sampling_data.size == 0:
        return 0.0
    
    if complete_data.shape[1] == 0 or sampling_data.shape[1] == 0:
        raise ValueError("数据必须包含数值列才能计算距离")
    
    # 计算完整数据集每个点到所有采样点的距离
    distance_matrix = cdist(complete_data, sampling_data, metric=distance_metric)
    
    # 对每个完整数据集的点，找到到采样点的最小距离
    min_distances = np.min(distance_matrix, axis=1)
    
    # 计算距离的平均值（不平方）
    mean_distance = np.mean(min_distances)
    
    return mean_distance


def calc_mean_distance(complete_space_df, sampling_points_df, not_feature_col, distance_metric='euclidean', scaler=None):
    """
    计算完整数据集到采样点的平均距离，使用完整空间的标准化参数。
    
    参数:
    -----------
    complete_space_df : pandas.DataFrame
        完整空间的DataFrame，用于确定标准化参数（如果scaler为None）
    sampling_points_df : pandas.DataFrame
        采样点的DataFrame
    not_feature_col : list
        需要排除的非特征列列表
    distance_metric : str, 默认='euclidean'
        距离计算的度量方法
    scaler : sklearn.preprocessing.StandardScaler, 可选
        预训练的标准化器。如果提供，将直接使用；否则会基于complete_space_df训练新的标准化器
        
    返回:
    --------
    float
        完整数据集到采样点的平均距离值
    """
    # 排除非特征列，保留数值特征列
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    
    # 提取特征数据
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty or sampling_features.empty:
        raise ValueError("DataFrame必须包含数值特征列才能计算距离")
    
    # 如果没有提供预训练的标准化器，则训练新的
    if scaler is None:
        # 确保列顺序一致
        common_cols = complete_features.columns.intersection(sampling_features.columns)
        complete_features = complete_features[common_cols]
        sampling_features = sampling_features[common_cols]
        
        # 使用完整空间的数据拟合标准化器
        scaler = StandardScaler()
        scaler.fit(complete_features)
    else:
        # 使用预训练的标准化器，确保特征列顺序与训练时一致
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else sampling_features.columns
        complete_features = complete_features[expected_features]
        sampling_features = sampling_features[expected_features]
    
    # 对数据进行标准化
    normalized_complete = scaler.transform(complete_features)
    normalized_sampling = scaler.transform(sampling_features)
    
    # 调用mean_distance_to_nearest函数计算平均距离
    return mean_distance_to_nearest(normalized_complete, normalized_sampling, distance_metric=distance_metric)


def calc_mean_squared_distance(complete_space_df, sampling_points_df, not_feature_col, distance_metric='euclidean', scaler=None):
    """
    计算完整数据集到采样点的平均平方距离，使用完整空间的标准化参数。
    
    参数:
    -----------
    complete_space_df : pandas.DataFrame
        完整空间的DataFrame，用于确定标准化参数（如果scaler为None）
    sampling_points_df : pandas.DataFrame
        采样点的DataFrame
    not_feature_col : list
        需要排除的非特征列列表
    distance_metric : str, 默认='euclidean'
        距离计算的度量方法
    scaler : sklearn.preprocessing.StandardScaler, 可选
        预训练的标准化器。如果提供，将直接使用；否则会基于complete_space_df训练新的标准化器
        
    返回:
    --------
    float
        完整数据集到采样点的平均平方距离值
    """
    # 排除非特征列，保留数值特征列
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    
    # 提取特征数据
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty or sampling_features.empty:
        raise ValueError("DataFrame必须包含数值特征列才能计算距离")
    
    # 如果没有提供预训练的标准化器，则训练新的
    if scaler is None:
        # 确保列顺序一致
        common_cols = complete_features.columns.intersection(sampling_features.columns)
        complete_features = complete_features[common_cols]
        sampling_features = sampling_features[common_cols]
        
        # 使用完整空间的数据拟合标准化器
        scaler = StandardScaler()
        scaler.fit(complete_features)
    else:
        # 使用预训练的标准化器，确保特征列顺序与训练时一致
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else sampling_features.columns
        complete_features = complete_features[expected_features]
        sampling_features = sampling_features[expected_features]
    
    # 对数据进行标准化
    normalized_complete = scaler.transform(complete_features)
    normalized_sampling = scaler.transform(sampling_features)
    
    # 调用mean_squared_distance_to_nearest函数计算平均平方距离
    return mean_squared_distance_to_nearest(normalized_complete, normalized_sampling, distance_metric=distance_metric)


def calc_entropy(complete_space_df, sampling_points_df, not_feature_col, k=5, distance_metric='euclidean', scaler=None):
    """
    计算采样点的空间熵，使用完整空间的标准化参数。
    
    参数:
    -----------
    complete_space_df : pandas.DataFrame
        完整空间的DataFrame，用于确定标准化参数（如果scaler为None）
    sampling_points_df : pandas.DataFrame
        采样点的DataFrame
    not_feature_col : list
        需要排除的非特征列列表
    k : int, 默认=5
        考虑的最近邻数量
    distance_metric : str, 默认='euclidean'
        KNN计算的距离度量
    scaler : sklearn.preprocessing.StandardScaler, 可选
        预训练的标准化器。如果提供，将直接使用；否则会基于complete_space_df训练新的标准化器
        
    返回:
    --------
    float
        采样点的估计空间熵值
    """
    # 排除非特征列，保留数值特征列
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    
    # 提取特征数据
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    
    if sampling_features.empty:
        raise ValueError("DataFrame必须包含数值特征列才能计算空间熵")
    
    # 如果没有提供预训练的标准化器，则训练新的
    if scaler is None:
        complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
        if complete_features.empty:
            raise ValueError("DataFrame必须包含数值特征列才能计算空间熵")
        
        # 确保列顺序一致
        common_cols = complete_features.columns.intersection(sampling_features.columns)
        complete_features = complete_features[common_cols]
        sampling_features = sampling_features[common_cols]
        
        # 使用完整空间的数据拟合标准化器
        scaler = StandardScaler()
        scaler.fit(complete_features)
    else:
        # 使用预训练的标准化器，确保特征列顺序与训练时一致
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else sampling_features.columns
        sampling_features = sampling_features[expected_features]
    
    # 对采样点进行标准化
    normalized_sampling = scaler.transform(sampling_features)
    # normalized_sampling = sampling_features.values
    
    # 调用entropy函数计算熵（传入numpy数组）
    return entropy(normalized_sampling, k=k, distance_metric=distance_metric)


def sample_eval(complete_space_df, sampling_points_df, not_feature_col, deviation=0.05, variance_threshold=0.95, distance_metric='euclidean'):
    """
    评估采样质量，通过多次随机采样计算空间熵并确定所需的采样数量。
    预先将数据进行PCA降维和标准化处理。
    
    参数:
    -----------
    complete_space_df : pandas.DataFrame
        完整空间的DataFrame
    sampling_points_df : pandas.DataFrame
        采样点的DataFrame
    not_feature_col : list
        需要排除的非特征列列表
    deviation : float, 默认=0.002
        期望偏差，表示均值的百分比（如0.05表示5%）
    variance_threshold : float, 默认=0.99
        PCA保留的累积方差比例阈值
    distance_metric : str, 默认='euclidean'
        距离度量方法，支持'euclidean', 'manhattan', 'cosine', 'tanimoto'
    """
    sample_size = len(sampling_points_df)
    
    # 预处理数据：PCA降维和标准化
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty:
        raise ValueError("DataFrame必须包含数值特征列才能计算空间熵")
    
    print(f"原始特征维度: {complete_features.shape[1]}")
    
    # 第一步：标准化
    scaler = StandardScaler()
    normalized_complete = scaler.fit_transform(complete_features)
    
    # 第二步：PCA降维
    pca = PCA()
    pca.fit(normalized_complete)
    
    # 确定保留的主成分数量
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    
    # 重新训练PCA以保留指定数量的主成分
    pca = PCA(n_components=n_components)
    pca_complete = pca.fit_transform(normalized_complete)
    
    print(f"PCA降维后维度: {pca_complete.shape[1]} (保留方差: {cumsum_variance[n_components-1]:.4f})")
    
    # 第三步：构建处理后的完整数据集
    # 保留非特征列，替换特征列为PCA降维后的结果
    processed_complete_df = complete_space_df[not_feature_col].copy()
    
    # 添加PCA降维后的特征列
    pca_feature_names = [f'pca_component_{i}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_complete, columns=pca_feature_names, index=complete_space_df.index)
    processed_complete_df = pd.concat([processed_complete_df, pca_df], axis=1)
    
    # 第四步：处理采样数据
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    normalized_sampling = scaler.transform(sampling_features)
    pca_sampling = pca.transform(normalized_sampling)
    
    processed_sampling_df = sampling_points_df[not_feature_col].copy()
    pca_sampling_df = pd.DataFrame(pca_sampling, columns=pca_feature_names, index=sampling_points_df.index)
    processed_sampling_df = pd.concat([processed_sampling_df, pca_sampling_df], axis=1)
    
    # 更新不包含特征列列表（现在只有原来的非特征列）
    processed_not_feature_col = not_feature_col.copy()
    
    # 第一步：进行10次随机采样，计算空间熵
    rand_initial_entropies = []
    for i in range(10):
        # 更新numpy随机数种子
        np.random.seed()
        random_sample, _ = rand_sampling_df_no_repeat(processed_complete_df, sample_size, processed_not_feature_col)
        entropy_value = calc_entropy(processed_complete_df, random_sample, processed_not_feature_col, distance_metric=distance_metric)
        rand_initial_entropies.append(entropy_value)
    
    # 计算均值和标准差
    rand_mean_entropy = np.mean(rand_initial_entropies)
    rand_std_entropy = np.std(rand_initial_entropies, ddof=1)  # 使用样本标准差
    
    # 使用95%置信水平的z值
    z_value = 1.96
    
    # 计算期望标准差（基于偏差参数）
    expected_std = abs(rand_mean_entropy) * deviation

    print(rand_std_entropy)
    print(expected_std)
    
    # 使用公式：n = z² × s² / σ²
    # 其中 s 是样本标准差，σ 是期望标准差
    required_samples = max(10, int(np.ceil((z_value**2 * rand_std_entropy**2) / (expected_std**2))))

    print(f'Random sampling required samples: {required_samples}')
    
    # 计算需要补充的采样数
    additional_samples = max(0, required_samples - 10)
    
    # 补充随机采样
    rand_all_entropies = rand_initial_entropies.copy()
    for i in tqdm.tqdm(range(additional_samples), desc="Additional random sampling"):
        # 更新numpy随机数种子
        np.random.seed()
        random_sample, _ = rand_sampling_df_no_repeat(processed_complete_df, sample_size, processed_not_feature_col)
        entropy_value = calc_entropy(processed_complete_df, random_sample, processed_not_feature_col, distance_metric=distance_metric)
        rand_all_entropies.append(entropy_value)
    
    # 计算最终的均值
    rand_final_mean_entropy = np.mean(rand_all_entropies)
    print(f'Random sampling final mean entropy: {rand_final_mean_entropy}')
    
    # 随机采样的mean_squared_distance计算
    rand_initial_mean_squared_distances = []
    for i in range(10):
        # 更新numpy随机数种子
        np.random.seed()
        random_sample, _ = rand_sampling_df_no_repeat(processed_complete_df, sample_size, processed_not_feature_col)
        mean_squared_distance_value = calc_mean_squared_distance(processed_complete_df, random_sample, processed_not_feature_col, distance_metric=distance_metric)
        rand_initial_mean_squared_distances.append(mean_squared_distance_value)
    
    # 计算均值和标准差
    rand_mean_mean_squared_distance = np.mean(rand_initial_mean_squared_distances)
    rand_std_mean_squared_distance = np.std(rand_initial_mean_squared_distances, ddof=1)  # 使用样本标准差
    
    # 计算期望标准差（基于偏差参数）
    rand_expected_std_mean_squared_distance = abs(rand_mean_mean_squared_distance) * deviation
    
    print(rand_std_mean_squared_distance)
    print(rand_expected_std_mean_squared_distance)
    
    # 使用公式：n = z² × s² / σ²
    rand_required_samples_mean_squared_distance = max(10, int(np.ceil((z_value**2 * rand_std_mean_squared_distance**2) / (rand_expected_std_mean_squared_distance**2))))

    print(f'Random sampling mean_squared_distance required samples: {rand_required_samples_mean_squared_distance}')
    
    # 计算需要补充的采样数
    rand_additional_samples_mean_squared_distance = max(0, rand_required_samples_mean_squared_distance - 10)
    
    # 补充随机采样的mean_squared_distance计算
    rand_all_mean_squared_distances = rand_initial_mean_squared_distances.copy()
    for i in tqdm.tqdm(range(rand_additional_samples_mean_squared_distance), desc="Additional random mean_squared_distance sampling"):
        # 更新numpy随机数种子
        np.random.seed()
        random_sample, _ = rand_sampling_df_no_repeat(processed_complete_df, sample_size, processed_not_feature_col)
        mean_squared_distance_value = calc_mean_squared_distance(processed_complete_df, random_sample, processed_not_feature_col, distance_metric=distance_metric)
        rand_all_mean_squared_distances.append(mean_squared_distance_value)
    
    # 计算最终的均值
    rand_final_mean_mean_squared_distance = np.mean(rand_all_mean_squared_distances)
    print(f'Random sampling final mean mean_squared_distance: {rand_final_mean_mean_squared_distance}')
    
    # 第二步：使用CVT采样执行相同的过程
    # cvt_initial_entropies = []
    # for i in range(10):
    #     # 更新numpy随机数种子
    #     np.random.seed()
    #     cvt_sample, _ = cvt_sampling_gold_df_norepeat(processed_complete_df, sample_size, processed_not_feature_col)
    #     entropy_value = calc_entropy(processed_complete_df, cvt_sample, processed_not_feature_col)
    #     cvt_initial_entropies.append(entropy_value)
    
    # # 计算均值和标准差
    # cvt_mean_entropy = np.mean(cvt_initial_entropies)
    # cvt_std_entropy = np.std(cvt_initial_entropies, ddof=1)  # 使用样本标准差
    
    # # 计算期望标准差（基于偏差参数）
    # cvt_expected_std = abs(cvt_mean_entropy) * deviation
    
    # # 使用公式：n = z² × s² / σ²
    # cvt_required_samples = max(10, int(np.ceil((z_value**2 * cvt_std_entropy**2) / (cvt_expected_std**2))))

    # print(f'CVT sampling required samples: {cvt_required_samples}')
    
    # # 计算需要补充的采样数
    # cvt_additional_samples = max(0, cvt_required_samples - 10)
    
    # # 补充CVT采样
    # cvt_all_entropies = cvt_initial_entropies.copy()
    # for i in tqdm.tqdm(range(cvt_additional_samples), desc="Additional CVT sampling"):
    #     # 更新numpy随机数种子
    #     np.random.seed()
    #     cvt_sample, _ = cvt_sampling_gold_df_norepeat(processed_complete_df, sample_size, processed_not_feature_col)
    #     entropy_value = calc_entropy(processed_complete_df, cvt_sample, processed_not_feature_col)
    #     cvt_all_entropies.append(entropy_value)
    
    # # 计算最终的均值
    # cvt_final_mean_entropy = np.mean(cvt_all_entropies)
    # print(f'CVT sampling final mean entropy: {cvt_final_mean_entropy}')
    
    # CVT_gold采样的mean_squared_distance计算
    cvt_gold_initial_mean_squared_distances = []
    for i in range(10):
        # 更新numpy随机数种子
        np.random.seed()
        cvt_gold_sample, _ = cvt_sampling_gold_df_norepeat(processed_complete_df, sample_size, processed_not_feature_col)
        mean_squared_distance_value = calc_mean_squared_distance(processed_complete_df, cvt_gold_sample, processed_not_feature_col, distance_metric=distance_metric)
        cvt_gold_initial_mean_squared_distances.append(mean_squared_distance_value)
    
    # 计算均值和标准差
    cvt_gold_mean_mean_squared_distance = np.mean(cvt_gold_initial_mean_squared_distances)
    cvt_gold_std_mean_squared_distance = np.std(cvt_gold_initial_mean_squared_distances, ddof=1)  # 使用样本标准差
    
    # 计算期望标准差（基于偏差参数）
    cvt_gold_expected_std_mean_squared_distance = abs(cvt_gold_mean_mean_squared_distance) * deviation
    
    # 使用公式：n = z² × s² / σ²
    cvt_gold_required_samples_mean_squared_distance = max(10, int(np.ceil((z_value**2 * cvt_gold_std_mean_squared_distance**2) / (cvt_gold_expected_std_mean_squared_distance**2))))

    print(f'CVT_gold sampling mean_squared_distance required samples: {cvt_gold_required_samples_mean_squared_distance}')
    
    # 计算需要补充的采样数
    cvt_gold_additional_samples_mean_squared_distance = max(0, cvt_gold_required_samples_mean_squared_distance - 10)
    
    # 补充CVT_gold采样的mean_squared_distance计算
    cvt_gold_all_mean_squared_distances = cvt_gold_initial_mean_squared_distances.copy()
    for i in tqdm.tqdm(range(cvt_gold_additional_samples_mean_squared_distance), desc="Additional CVT_gold mean_squared_distance sampling"):
        # 更新numpy随机数种子
        np.random.seed()
        cvt_gold_sample, _ = cvt_sampling_gold_df_norepeat(processed_complete_df, sample_size, processed_not_feature_col)
        mean_squared_distance_value = calc_mean_squared_distance(processed_complete_df, cvt_gold_sample, processed_not_feature_col, distance_metric=distance_metric)
        cvt_gold_all_mean_squared_distances.append(mean_squared_distance_value)
    
    # 计算最终的均值
    cvt_gold_final_mean_mean_squared_distance = np.mean(cvt_gold_all_mean_squared_distances)
    print(f'CVT_gold sampling final mean mean_squared_distance: {cvt_gold_final_mean_mean_squared_distance}')
    
    # 第三步：使用FPS采样执行相同的过程
    fps_initial_entropies = []
    for i in range(10):
        # 更新numpy随机数种子
        np.random.seed()
        fps_sample, _ = fps_sampling_df_norepeat(processed_complete_df, sample_size, processed_not_feature_col)
        entropy_value = calc_entropy(processed_complete_df, fps_sample, processed_not_feature_col, distance_metric=distance_metric)
        fps_initial_entropies.append(entropy_value)
    
    # 计算均值和标准差
    fps_mean_entropy = np.mean(fps_initial_entropies)
    fps_std_entropy = np.std(fps_initial_entropies, ddof=1)  # 使用样本标准差
    
    # 计算期望标准差（基于偏差参数）
    fps_expected_std = abs(fps_mean_entropy) * deviation
    
    # 使用公式：n = z² × s² / σ²
    fps_required_samples = max(10, int(np.ceil((z_value**2 * fps_std_entropy**2) / (fps_expected_std**2))))

    print(f'FPS sampling required samples: {fps_required_samples}')
    
    # 计算需要补充的采样数
    fps_additional_samples = max(0, fps_required_samples - 10)
    
    # 补充FPS采样
    fps_all_entropies = fps_initial_entropies.copy()
    for i in tqdm.tqdm(range(fps_additional_samples), desc="Additional FPS sampling"):
        # 更新numpy随机数种子
        np.random.seed()
        fps_sample, _ = fps_sampling_df_norepeat(processed_complete_df, sample_size, processed_not_feature_col)
        entropy_value = calc_entropy(processed_complete_df, fps_sample, processed_not_feature_col, distance_metric=distance_metric)
        fps_all_entropies.append(entropy_value)
    
    # 计算最终的均值
    fps_final_mean_entropy = np.mean(fps_all_entropies)
    print(f'FPS sampling final mean entropy: {fps_final_mean_entropy}')
    
    # 第四步：对输入采样点进行评估
    # 计算输入采样点的空间熵
    input_sample_entropy = calc_entropy(processed_complete_df, processed_sampling_df, processed_not_feature_col, distance_metric=distance_metric)
    print(f'Input sampling entropy: {input_sample_entropy}')
    
    # 计算输入采样点的mean_squared_distance
    input_sample_mean_squared_distance = calc_mean_squared_distance(processed_complete_df, processed_sampling_df, processed_not_feature_col, distance_metric=distance_metric)
    print(f'Input sampling mean_squared_distance: {input_sample_mean_squared_distance}')
    
    # 新的打分函数：sigmoid(0.2*((采样熵-随机熵)/(FPS熵-随机熵))+0.8*(采样MSD-随机MSD)/(CVT MSD-随机MSD))
    # 计算熵比例
    if fps_final_mean_entropy != rand_final_mean_entropy:
        entropy_ratio = (input_sample_entropy - rand_final_mean_entropy) / (fps_final_mean_entropy - rand_final_mean_entropy)
    else:
        entropy_ratio = 0.0
    
    # 计算MSD比例
    if cvt_gold_final_mean_mean_squared_distance != rand_final_mean_mean_squared_distance:
        msd_ratio = (input_sample_mean_squared_distance - rand_final_mean_mean_squared_distance) / (cvt_gold_final_mean_mean_squared_distance - rand_final_mean_mean_squared_distance)
    else:
        msd_ratio = 0.0
    
    # 计算组合分数
    combined_score = 0.5 * entropy_ratio + 0.5 * msd_ratio
    
    # 应用sigmoid函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-(x-0.0)))
    
    score = sigmoid(combined_score)
    
    return score



if __name__ == "__main__":
    # 功能测试
    import time
    start_time = time.time()
    print("开始功能测试...")
    
    # 读取smiles数据和特征数据
    smiles_df = pd.read_csv('zinc_aryl_aldehyde.csv')
    features_df = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')
    
    # 合并smiles和特征数据
    complete_space_df = pd.concat([smiles_df, features_df], axis=1)
    print(f"完整空间数据形状: {complete_space_df.shape}")
    
    #读取采样数据并合并
    print('CVT采样评估：')
    final_drop_df = pd.read_csv('final_drop.csv')
    final_sampling_df = pd.read_csv('final_sampling.csv')
    combined_sampling = pd.concat([final_drop_df, final_sampling_df], ignore_index=True)
    print(f"合并后采样数据形状: {combined_sampling.shape}")
    
    # 在完整空间中找到对应的smiles相同的行
    sampling_points_df = complete_space_df[complete_space_df['smiles'].isin(combined_sampling['smiles'])]
    print(f"匹配的采样点数量: {len(sampling_points_df)}")

    # 设置不包含的特征列
    not_feature_col = ['smiles']
    
    # 计算采样分数
    print("计算采样分数...")
    score = sample_eval(complete_space_df, sampling_points_df, not_feature_col)
    
    print(f"采样评估分数: {score:.4f}")
    
    # 使用人工数据作为采样点
    print('人工采样评估：')
    sampling_smiles_df = pd.read_csv('人工.csv')
    sampling_features_df = pd.read_csv('fp_spoc_morgan41024_Maccs_人工_alcohol.csv')
    sampling_points_df = pd.concat([sampling_smiles_df, sampling_features_df], axis=1)
    print(f"人工采样数据形状: {sampling_points_df.shape}")
    print(f"采样点数量: {len(sampling_points_df)}")
    
    # 设置不包含的特征列
    not_feature_col = ['smiles']
    
    # 计算采样分数
    print("计算采样分数...")
    score = sample_eval(complete_space_df, sampling_points_df, not_feature_col)
    
    print(f"采样评估分数: {score:.4f}")
    print("功能测试完成！")
    print(f"总运行时间: {time.time() - start_time:.2f}秒")