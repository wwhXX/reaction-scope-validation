import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
from functools import partial

def tanimoto_similarity(p, t):
    """
    Calculate Tanimoto similarity using the continuous formula: p*t/(p^2+t^2-p*t)
    
    Parameters:
        p: numpy.ndarray, first vector or matrix
        t: numpy.ndarray, second vector or matrix
    
    Returns:
        numpy.ndarray, Tanimoto similarity values
    """
    p_dot_t = np.sum(p * t, axis=-1)
    p_squared = np.sum(p * p, axis=-1)
    t_squared = np.sum(t * t, axis=-1)
    
    denominator = p_squared + t_squared - p_dot_t
    # Add small epsilon to avoid division by zero
    denominator = np.maximum(denominator, 1e-10)
    
    return p_dot_t / denominator

def tanimoto_distance(p, t):
    """
    Calculate Tanimoto distance (1 - Tanimoto similarity)
    
    Parameters:
        p: numpy.ndarray, first vector or matrix
        t: numpy.ndarray, second vector or matrix
    
    Returns:
        numpy.ndarray, Tanimoto distance values
    """
    return 1.0 - tanimoto_similarity(p, t)

def _compute_tanimoto_distances_chunk(args):
    """
    Helper function for parallel computation of Tanimoto distances
    """
    i_start, i_end, X_chunk, Y = args
    n_Y = Y.shape[0]
    chunk_size = i_end - i_start
    distances_chunk = np.zeros((chunk_size, n_Y))
    
    for i in range(chunk_size):
        for j in range(n_Y):
            distances_chunk[i, j] = tanimoto_distance(X_chunk[i:i+1], Y[j:j+1])
    
    return i_start, distances_chunk

def _compute_tanimoto_distances_vectorized(X, Y):
    """
    Vectorized computation of Tanimoto distances (faster for small to medium datasets)
    """
    if Y is None:
        Y = X
    
    n_X, n_Y = X.shape[0], Y.shape[0]
    distances = np.zeros((n_X, n_Y))
    
    # Expand dimensions for broadcasting
    X_expanded = X[:, np.newaxis, :]  # (n_X, 1, n_features)
    Y_expanded = Y[np.newaxis, :, :]  # (1, n_Y, n_features)
    
    # Compute dot products, squared norms efficiently
    X_dot_Y = np.sum(X_expanded * Y_expanded, axis=2)  # (n_X, n_Y)
    X_squared = np.sum(X_expanded * X_expanded, axis=2)  # (n_X, n_Y)
    Y_squared = np.sum(Y_expanded * Y_expanded, axis=2)  # (n_X, n_Y)
    
    # Compute Tanimoto similarity
    denominator = X_squared + Y_squared - X_dot_Y
    denominator = np.maximum(denominator, 1e-10)
    
    tanimoto_sim = X_dot_Y / denominator
    distances = 1.0 - tanimoto_sim
    
    return distances

def compute_tanimoto_distances(X, Y=None, n_jobs=None, chunk_threshold=1000):
    """
    Compute pairwise Tanimoto distances between data points with optional parallel processing
    
    Parameters:
        X: numpy.ndarray, data point matrix (n_samples, n_features)
        Y: numpy.ndarray, optional, second set of data points. If None, compute distances within X
        n_jobs: int, optional, number of parallel jobs. If None, uses all available CPUs.
                If 1, uses sequential processing. If -1, uses all CPUs.
        chunk_threshold: int, threshold for using parallel processing (default: 1000)
    
    Returns:
        numpy.ndarray, distance matrix
    """
    if Y is None:
        Y = X
    
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    
    # For small datasets, use vectorized computation
    if n_X * n_Y < chunk_threshold:
        return _compute_tanimoto_distances_vectorized(X, Y)
    
    # For large datasets, use parallel processing
    if n_jobs is None:
        n_jobs = cpu_count()
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs == 1:
        # Sequential processing
        distances = np.zeros((n_X, n_Y))
        for i in range(n_X):
            for j in range(n_Y):
                distances[i, j] = tanimoto_distance(X[i:i+1], Y[j:j+1])
        return distances
    
    # Determine chunk size
    chunk_size = max(1, n_X // n_jobs)
    chunks = []
    
    for i in range(0, n_X, chunk_size):
        i_end = min(i + chunk_size, n_X)
        X_chunk = X[i:i_end]
        chunks.append((i, i_end, X_chunk, Y))
    
    # Parallel processing
    distances = np.zeros((n_X, n_Y))
    
    if len(chunks) > 1:  # Only use multiprocessing if we have multiple chunks
        with Pool(processes=min(n_jobs, len(chunks))) as pool:
            results = pool.map(_compute_tanimoto_distances_chunk, chunks)
        
        # Combine results
        for i_start, distances_chunk in results:
            i_end = i_start + distances_chunk.shape[0]
            distances[i_start:i_end, :] = distances_chunk
    else:
        # Single chunk, process directly
        i_start, distances_chunk = _compute_tanimoto_distances_chunk(chunks[0])
        distances[i_start:i_start + distances_chunk.shape[0], :] = distances_chunk
    
    return distances

def compute_cluster_centroid(cluster_points, metric='euclidean'):
    """
    Calculate cluster centroid based on distance metric
    
    Parameters:
        cluster_points: numpy.ndarray, data points in the cluster
        metric: str, distance metric method
    
    Returns:
        numpy.ndarray, cluster centroid
    """
    if len(cluster_points) == 0:
        raise ValueError("Cluster point set cannot be empty")
    
    if metric == 'manhattan':
        return np.median(cluster_points, axis=0)
    elif metric == 'tanimoto':
        return np.mean(cluster_points, axis=0)
    else:
        return np.mean(cluster_points, axis=0)

def compute_pairwise_distances(X, Y=None, metric='euclidean', n_jobs=None):
    """
    Compute pairwise distances between data points, supporting multiple distance metrics
    
    Parameters:
        X: numpy.ndarray, data point matrix
        Y: numpy.ndarray, optional, second set of data points. If None, compute distances within X
        metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine', 'tanimoto'
        n_jobs: int, optional, number of parallel jobs for tanimoto computation
    
    Returns:
        numpy.ndarray, distance matrix
    """
    if metric in ['euclidean', 'manhattan']:
        return pairwise_distances(X, Y, metric=metric)
    elif metric == 'cosine':
        return pairwise_distances(X, Y, metric='cosine')
    elif metric == 'tanimoto':
        return compute_tanimoto_distances(X, Y, n_jobs=n_jobs)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}. Supported metrics: 'euclidean', 'manhattan', 'cosine', 'tanimoto'")


def cvt_sampling_df(data, k, not_feature_columns, max_iters=500, tol=1e-4):
    """
    CVT sampling algorithm (supports DataFrame input, preserves non-numeric columns)
    
    Parameters:
        data: DataFrame, contains feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        k: number of center points to sample
        not_feature_columns: list, column names not participating in distance calculation (numeric type)
        max_iters: maximum number of iterations
        tol: convergence threshold for center point changes
    
    Returns:
        centers: DataFrame, sampled center points (containing all original columns)
        unselected_points: DataFrame, points not sampled (all original columns)
    """
    X = data.drop(not_feature_columns, axis=1).values
    
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for _ in range(max_iters):
        if np.isnan(centers_X).any():
            centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
        distances = pairwise_distances(X, centers_X)

        labels = np.argmin(distances, axis=1)
        
        new_centers_X = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT algorithm converged, iterations: {_}")
            break
        centers_X = new_centers_X
    
    if np.isnan(centers_X).any():
        centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
    final_distances = pairwise_distances(X, centers_X)
    selected_indices = np.argmin(final_distances, axis=0)
    
    centers = data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), selected_indices)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def classify_by_centers(data, centers, not_feature_columns, distance_metric='euclidean'):
    """
    Get the classification for a set of data
    
    Parameters:
        data: DataFrame, contains feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        centers: DataFrame, contains feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        not_feature_columns: list, column names of non-descriptive columns
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine', 'tanimoto'
    
    Returns:
        output: DataFrame, with added "labels" classification column (containing all original columns)
        Note: categories correspond to centers indices
    """
    raw_data = data.copy()
    for not_feature_column in not_feature_columns:
        if not_feature_column in data.columns:
            data = data.drop(not_feature_column, axis=1)
        if not_feature_column in centers.columns:
            centers = centers.drop(not_feature_column, axis=1)
    data = data.values
    centers = centers.values
    distances = compute_pairwise_distances(data, centers, distance_metric)
    
    labels = np.argmin(distances, axis=1).astype(int)
    if 'labels' in raw_data.columns:
        raw_data = raw_data.drop('labels', axis=1)
    output_data = pd.concat([raw_data, pd.DataFrame(labels, columns=['labels'])], axis=1)

    
    return output_data


def get_sampling(task_itr_id, drop_classes, not_feature_cols, k, max_iters=500, tol=1e-4):
    '''
    Perform one sampling and classification based on sampling.
    
    Parameters:
        task_itr_id: int, iteration number (starting from 1)
        drop_classes: list, classes to exclude
        not_feature_cols: list, column names of non-feature columns
        k: int, number of sampling center points
        max_iters: int, maximum number of iterations
        tol: float, convergence threshold for center point changes
    
    Returns:
        sampled_points: DataFrame, sampled center points (containing all original columns, i.e., label is from previous iteration)
        labeled_points: DataFrame, data and categories after sampling classification (containing all original columns, categories are sampled_points indices)
    '''
    data = pd.read_csv(f'./itr/labeled_points_itr{task_itr_id-1}.csv')
    
    if drop_classes != []:
        for drop_class in drop_classes:
            data = data[data['labels'] != drop_class].reset_index(drop=True)
    print(data)
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

def get_sampling_weighted(data, task_itr_id, not_feature_cols, k, max_iters=500, tol=1e-4, distance_metric='euclidean', n_jobs=None):
    '''
    Perform one sampling and classification based on sampling.
    
    Parameters:
        task_itr_id: int, iteration number (starting from 1)
        not_feature_cols: list, column names of non-feature columns
        k: int, number of sampling center points
        max_iters: int, maximum number of iterations
        tol: float, convergence threshold for center point changes
        distance_metric: str, distance metric method
    
    Returns:
        sampled_points: DataFrame, sampled center points (containing all original columns, i.e., label is from previous iteration)
        labeled_points: DataFrame, data and categories after sampling classification (containing all original columns, categories are sampled_points indices)
    '''
    sampled_data = data[data['ScreenLabel']!='BASE']
    print(data)
    sampled_points, unsampled_points = weighted_itr_cvt_sampling_df_norepeat(
        data=data,
        k=k,
        not_feature_columns=not_feature_cols,
        max_iters=max_iters,
        tol=tol,
        sampled_data=sampled_data,
        distance_metric=distance_metric,
        n_jobs=n_jobs,
    )
    labeled_points = classify_by_centers(data, sampled_points, not_feature_cols, distance_metric)
    return sampled_points, labeled_points
def weighted_itr_cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=2000, tol=1e-5,
                                        sampled_data=None, repulsion_strength=0.0, cvt_weight=1.0,
                                        learning_rate=0.01, adaptive_lr=True, distance_metric='euclidean',
                                        cvt_init_iters=100, n_jobs=None,
                                        approximate_distance=False, approx_k=None,
                                        fixed_lambda=False):
    """
    Weighted iterative CVT sampling algorithm: combines CVT energy minimization with inverse square repulsion force from already sampled points
    Improved version: first use analytical CVT method for good initial guess, then perform gradient descent optimization
    
    Parameters:
        data: DataFrame, contains feature columns and non-numeric columns
        k: number of center points to sample
        not_feature_columns: list, column names not participating in distance calculation
        max_iters: maximum number of iterations
        tol: convergence threshold
        sampled_data: DataFrame, already sampled data points (immovable, generate repulsion force). Only calculate repulsion force for points with ScreenLabel='Excluded_Sampled'
        repulsion_strength: float, repulsion force strength coefficient
        cvt_weight: float, CVT energy weight
        learning_rate: float, gradient descent learning rate
        adaptive_lr: bool, whether to use adaptive learning rate
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine', 'tanimoto'
        cvt_init_iters: int, number of iterations for initial CVT analytical optimization
        n_jobs: int, optional, number of parallel jobs for tanimoto distance computation
        approximate_distance: bool, if True, estimate max_distance via CVT sampling instead of computing the full pairwise distance matrix
        approx_k: int, number of CVT sample points used for distance approximation when approximate_distance=True; defaults to k
        fixed_lambda: bool, if True, set base_repulsion_coefficient=1 directly and skip all distance-based coefficient computation

    Returns:
        centers: DataFrame, sampled center points
        unselected_points: DataFrame, unsampled points
    """
    
    def compute_distances_metric(points, centers, metric):
        return compute_pairwise_distances(points, centers, metric, n_jobs=n_jobs)
    
    def compute_cvt_energy_gradients(movable_centers, fixed_centers, data_points, metric='euclidean'):
        if fixed_centers is not None and len(fixed_centers) > 0:
            all_centers = np.vstack([movable_centers, fixed_centers])
        else:
            all_centers = movable_centers
        
        distances = compute_distances_metric(data_points, all_centers, metric)
        labels = np.argmin(distances, axis=1)
        
        energy = 0
        gradients = np.zeros_like(movable_centers)
        
        if metric == 'tanimoto':
            for i in range(len(all_centers)):
                cluster_points = data_points[labels == i]
                if len(cluster_points) > 0:
                    center = all_centers[i].reshape(1, -1)
                    p_dot_c = np.sum(cluster_points * center, axis=1)
                    p_squared = np.sum(cluster_points * cluster_points, axis=1)
                    c_squared = np.sum(center * center)
                    
                    denominator = p_squared + c_squared - p_dot_c
                    denominator = np.maximum(denominator, 1e-10)
                    
                    tanimoto_sim = p_dot_c / denominator
                    tanimoto_dist = 1.0 - tanimoto_sim
                    energy += np.sum(tanimoto_dist)
        else:
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
        
        if metric == 'tanimoto':
            for i in range(len(movable_centers)):
                cluster_points = data_points[labels == i]
                if len(cluster_points) > 0:
                    center = movable_centers[i]
                    
                    p_dot_c = np.sum(cluster_points * center, axis=1)  # (n_points,)
                    p_squared = np.sum(cluster_points * cluster_points, axis=1)  # (n_points,)
                    c_squared = np.sum(center * center)  # scalar
                    
                    denominator = p_squared + c_squared - p_dot_c  # (n_points,)
                    denominator = np.maximum(denominator, 1e-10)
                    
                    gradient_sum = np.zeros_like(center)
                    for j, p in enumerate(cluster_points):
                        dS_dc = (p * denominator[j] - p_dot_c[j] * (2*center - p)) / (denominator[j]**2)
                        gradient_sum += -dS_dc
                    
                    gradients[i] = gradient_sum
        else:
            for i in range(len(movable_centers)):
                cluster_points = data_points[labels == i]
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
        energy = 0
        gradients = np.zeros_like(centers)
        
        for i, center in enumerate(centers):
            for repulsion_point in repulsion_points:
                diff = center - repulsion_point
                
                if metric == 'tanimoto':
                    tanimoto_sim = tanimoto_similarity(center.reshape(1, -1), repulsion_point.reshape(1, -1))
                    tanimoto_dist = 1.0 - tanimoto_sim
                    tanimoto_dist = np.maximum(tanimoto_dist, 1e-10)
                    
                    energy += strength / tanimoto_dist
                    
                    p = repulsion_point
                    c = center
                    
                    p_dot_c = np.sum(p * c)
                    p_squared = np.sum(p * p)
                    c_squared = np.sum(c * c)
                    
                    denominator = p_squared + c_squared - p_dot_c
                    denominator = np.maximum(denominator, 1e-10)
                    
                    dS_dc = (p * denominator - p_dot_c * (2*c - p)) / (denominator**2)
                    
                    repulsion_gradient = strength * dS_dc / (tanimoto_dist**2)
                    gradients[i] += repulsion_gradient
                    
                elif metric == 'euclidean':
                    distance_sq = np.sum(diff**2) + 1e-10
                    energy += strength / distance_sq
                    gradients[i] += -2 * strength * diff / (distance_sq**2)
                elif metric == 'manhattan':
                    manhattan_dist = np.sum(np.abs(diff)) + 1e-10
                    energy += strength / (manhattan_dist**2)
                    gradients[i] += -2 * strength * np.sign(diff) / (manhattan_dist**3)
                elif metric == 'cosine':
                    distance_sq = np.sum(diff**2) + 1e-10
                    energy += strength / distance_sq
                    gradients[i] += -2 * strength * diff / (distance_sq**2)
        
        return energy, gradients
    
    def compute_total_energy_and_gradients(movable_centers, fixed_centers, data_points, repulsion_points, 
                                         repulsion_strength, cvt_weight, metric):
        cvt_energy, cvt_gradients = compute_cvt_energy_gradients(movable_centers, fixed_centers, data_points, metric)
        
        if repulsion_points is not None and len(repulsion_points) > 0:
            repulsion_energy, repulsion_gradients = compute_repulsion_energy_gradients(
                movable_centers, repulsion_points, repulsion_strength, metric)
        else:
            repulsion_energy = 0
            repulsion_gradients = np.zeros_like(movable_centers)
        
        total_energy = cvt_weight * cvt_energy + repulsion_energy
        total_energy = total_energy / k
        total_gradients = cvt_weight * cvt_gradients + repulsion_gradients
        total_gradients = total_gradients / k
        
        return total_energy, total_gradients
    
    all_data_features = data.drop(not_feature_columns, axis=1).values

    if fixed_lambda:
        base_repulsion_coefficient = 1
        print("fixed_lambda=True: base_repulsion_coefficient set to 1, skipping distance computation")
    else:
        if approximate_distance:
            _approx_k = approx_k if approx_k is not None else k
            print(f"approximate_distance=True: estimating max_distance via CVT sampling with {_approx_k} points...")
            cvt_centers, _ = cvt_sampling_df(data, _approx_k, not_feature_columns)
            labeled_data = classify_by_centers(data, cvt_centers, not_feature_columns, distance_metric)
            center_features = cvt_centers.drop(not_feature_columns, axis=1).values
            labels = labeled_data['labels'].values
            distances_to_centers = np.zeros(len(all_data_features))
            for label_idx in range(len(center_features)):
                mask = labels == label_idx
                if np.any(mask):
                    dists = compute_pairwise_distances(
                        all_data_features[mask], center_features[label_idx:label_idx+1], distance_metric
                    ).flatten()
                    distances_to_centers[mask] = dists
            max_distance = float(np.mean(distances_to_centers))
            print(f"Approximate max_distance (mean distance to assigned CVT center): {max_distance:.6f}")
        else:
            print("Calculating maximum distance between two points in data...")
            distance_matrix = compute_pairwise_distances(all_data_features, all_data_features, distance_metric)
            max_distance = np.max(distance_matrix)
            print(f"Maximum distance in data: {max_distance:.6f}")

        d = all_data_features.shape[1]
        expected_sq_distance = (max_distance**2) * d / 12
        print(f"Expected square distance between sampling points under uniform sampling: {expected_sq_distance:.6f}")

        n_points = len(all_data_features)
        base_repulsion_coefficient = expected_sq_distance**2 * n_points

    final_repulsion_strength = base_repulsion_coefficient * (10 ** repulsion_strength)
    
    print(f"Base repulsion coefficient: {base_repulsion_coefficient:.6e}")
    print(f"Final repulsion strength (base * 10^{repulsion_strength}): {final_repulsion_strength:.6e}")
    
    if sampled_data is not None:
        available_data = data[data['ScreenLabel'] == 'BASE'].copy()
        
        all_sampled_features = sampled_data.drop(not_feature_columns, axis=1).values
        print(f"Found {len(all_sampled_features)} already sampled points as CVT fixed center points")
        
        if 'ScreenLabel' in sampled_data.columns:
            excluded_sampled_data = sampled_data[sampled_data['ScreenLabel'] == 'Excluded_Sampled']
            if len(excluded_sampled_data) > 0:
                repulsion_features = excluded_sampled_data.drop(not_feature_columns, axis=1).values
                print(f"Using {len(excluded_sampled_data)} points with ScreenLabel='Excluded_Sampled' for repulsion force calculation")
            else:
                repulsion_features = None
                print("No points with ScreenLabel='Excluded_Sampled' found, no repulsion force calculated")
        else:
            repulsion_features = None
            print("ScreenLabel column not found, no repulsion force calculated")
    
        all_data_features = data.drop(not_feature_columns, axis=1).values
        print(f"CVT calculation will use all {len(all_data_features)} data points")
    else:
        available_data = data.copy()
        all_sampled_features = None
        repulsion_features = None
        all_data_features = data.drop(not_feature_columns, axis=1).values
        print("No sampled data available, performing standard CVT sampling")
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data points ({len(X)}) is less than required sampling count ({k})")
    
    print(f"Stage 1: Initializing center points using analytical CVT method ({cvt_init_iters} iterations)")
    
    indices = np.random.choice(len(X), k, replace=False)
    centers = X[indices].copy().astype(float)
    
    cvt_data_points = X
    
    for cvt_iter in range(cvt_init_iters):
        if np.isnan(centers).any():
            centers[np.isnan(centers)] = np.mean(centers[~np.isnan(centers)])
        
        if all_sampled_features is not None and len(all_sampled_features) > 0:
            all_centers_for_voronoi = np.vstack([centers, all_sampled_features])
        else:
            all_centers_for_voronoi = centers
        
        distances = compute_distances_metric(cvt_data_points, all_centers_for_voronoi, distance_metric)
        labels = np.argmin(distances, axis=1)
        
        new_centers = np.zeros_like(centers)
        for i in range(k):
            cluster_points = cvt_data_points[labels == i]
            if len(cluster_points) > 0:
                new_centers[i] = compute_cluster_centroid(cluster_points, distance_metric)
            else:
                new_centers[i] = centers[i]
        
        if np.linalg.norm(new_centers - centers) < tol:
            print(f"Analytical CVT converged, iterations: {cvt_iter}")
            break
        centers = new_centers
    
    print(f"Stage 2: Weighted energy optimization based on analytical CVT initial guess (max {max_iters} iterations)")
    
    prev_energy = float('inf')
    lr = learning_rate
    
    print(f"Starting weighted energy optimization, repulsion strength: {final_repulsion_strength:.6e}, CVT weight: {cvt_weight}")
    
    for iteration in range(max_iters):
        total_energy, gradients = compute_total_energy_and_gradients(
            centers, all_sampled_features, X, repulsion_features, final_repulsion_strength, cvt_weight, distance_metric)
        
        centers_new = centers - lr * gradients
        
        if adaptive_lr:
            new_energy, _ = compute_total_energy_and_gradients(
                centers_new, all_sampled_features, X, repulsion_features, final_repulsion_strength, cvt_weight, distance_metric)

            new_energy_val = float(new_energy) if np.isscalar(new_energy) else float(np.sum(new_energy))
            total_energy_val = float(total_energy) if np.isscalar(total_energy) else float(np.sum(total_energy))

            if new_energy_val > total_energy_val:
                lr *= 0.8
                if lr < learning_rate * 1e-4:
                    print(f"Learning rate too small, early stopping. Iterations: {iteration}")
                    break
                continue
            else:
                prev_energy_val = float(prev_energy) if np.isscalar(prev_energy) else float(np.sum(prev_energy))
                energy_diff = abs(new_energy_val - prev_energy_val)
                tol_check = tol * abs(prev_energy_val) if prev_energy_val != 0 else tol
                if energy_diff < tol_check:
                    print(f"Weighted energy optimization converged, iterations: {iteration}, final energy: {new_energy_val:.6f}")
                    break
                else:
                    lr = min(lr * 1.05, learning_rate)
        
        centers = centers_new
        prev_energy = total_energy
        
        if iteration % 10 == 0:
            energy_value = float(total_energy) if np.isscalar(total_energy) else float(np.sum(total_energy))
            print(f"Iteration {iteration}: Total energy = {energy_value:.6f}, Learning rate = {lr:.6f}")
    
    final_distances = compute_distances_metric(X, centers, distance_metric)
    
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
            raise ValueError(f"Unable to find enough non-duplicate sampling points. Need {k} points, but can only find {len(selected_indices)} non-duplicate points")
    
    selected_indices = np.array(selected_indices)
    
    result_centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), result_centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"Weighted CVT sampling completed, sampled {len(result_centers)} points")
    return result_centers, unselected_points