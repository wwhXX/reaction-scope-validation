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

# Import calculation functions from utils.py
from utils import calc_entropy, calc_mean_squared_distance

# Import sampling methods from sampling.py
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


def _compute_single_entropy(args):
    """
    Helper function for parallel entropy computation.
    
    Parameters:
    -----------
    args : tuple
        (processed_complete_df, sample_size, processed_not_feature_col, sampling_function, distance_metric, seed)
    
    Returns:
    --------
    float
        Computed entropy value
    """
    processed_complete_df, sample_size, processed_not_feature_col, sampling_function, distance_metric, seed = args
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    
    # Call sampling function
    try:
        sample, _ = sampling_function(processed_complete_df, sample_size, processed_not_feature_col, distance_metric=distance_metric)
    except TypeError:
        sample, _ = sampling_function(processed_complete_df, sample_size, processed_not_feature_col)
    
    # Compute entropy
    return calc_entropy(processed_complete_df, sample, processed_not_feature_col, distance_metric=distance_metric)


def _compute_single_msd(args):
    """
    Helper function for parallel mean squared distance computation.
    
    Parameters:
    -----------
    args : tuple
        (processed_complete_df, sample_size, processed_not_feature_col, sampling_function, distance_metric, seed)
    
    Returns:
    --------
    float
        Computed mean squared distance value
    """
    processed_complete_df, sample_size, processed_not_feature_col, sampling_function, distance_metric, seed = args
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    
    # Call sampling function
    try:
        sample, _ = sampling_function(processed_complete_df, sample_size, processed_not_feature_col, distance_metric=distance_metric)
    except TypeError:
        sample, _ = sampling_function(processed_complete_df, sample_size, processed_not_feature_col)
    
    # Compute mean distance based on distance metric
    if distance_metric == 'manhattan':
        from utils import calc_mean_distance
        return calc_mean_distance(processed_complete_df, sample, processed_not_feature_col)
    else:
        return calc_mean_squared_distance(processed_complete_df, sample, processed_not_feature_col)


def _parallel_computation(computation_func, args_list, n_workers=None):
    """
    Execute parallel computation using multiprocessing.
    
    Parameters:
    -----------
    computation_func : function
        Function to execute in parallel
    args_list : list
        List of arguments for each computation
    n_workers : int, optional
        Number of worker processes. If None, uses all available CPUs
    
    Returns:
    --------
    list
        List of computation results
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    with Pool(n_workers) as pool:
        results = pool.map(computation_func, args_list)
    
    return results


def evaluate_sampling_method(complete_space_df, sample_size, not_feature_col, 
                           sampling_method_name, deviation=0.05, 
                           n_initial_samples=10, distance_metric='euclidean', n_workers=None):
    """
    Evaluate spatial entropy and mean squared distance for a specific sampling method.
    
    Parameters:
    -----------
    complete_space_df : pandas.DataFrame
        Complete space DataFrame
    sample_size : int
        Number of sampling points
    not_feature_col : list
        List of non-feature columns to exclude
    sampling_method_name : str
        Sampling method name, supports: 'cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'random', 'weighted_itr_cvt', 'ward', 'kennard_stone'
    deviation : float, default=0.05
        Expected deviation as percentage of mean (e.g., 0.05 means 5%)
    n_initial_samples : int, default=10
        Number of initial samples
    distance_metric : str, default='euclidean'
        Distance metric for sampling methods, supports 'euclidean', 'manhattan', 'cosine', 'tanimoto'
    n_workers : int, optional
        Number of worker processes for parallel computation. If None, uses all available CPUs
    
    Returns:
    --------
    dict
        Dictionary containing spatial entropy and mean squared distance evaluation results
    """
    
    # Get sampling function
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
        raise ValueError(f"Unsupported sampling method: {sampling_method_name}. Supported methods: {list(sampling_functions.keys())}")
    
    sampling_function = sampling_functions[sampling_method_name]
    
    # Preprocess data: Only standardization (no PCA)
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty:
        raise ValueError("DataFrame must contain numeric feature columns to compute spatial entropy")
    
    print(f"Original feature dimensions: {complete_features.shape[1]}")
    
    # Step 1: Standardization only
    scaler = StandardScaler()
    normalized_complete = scaler.fit_transform(complete_features)
    
    print(f"Using standardized space with {normalized_complete.shape[1]} dimensions")
    
    # Step 2: Build processed complete dataset with standardized features
    processed_complete_df = complete_space_df[not_feature_col].copy()
    
    # Add standardized feature columns
    standardized_feature_names = [f'std_{col}' for col in complete_features.columns]
    standardized_df = pd.DataFrame(normalized_complete, columns=standardized_feature_names, index=complete_space_df.index)
    processed_complete_df = pd.concat([processed_complete_df, standardized_df], axis=1)
    
    # Update non-feature column list (now only original non-feature columns)
    processed_not_feature_col = not_feature_col.copy()
    
    # Use 95% confidence level z-value
    z_value = 1.96
    
    print(f"Starting evaluation for {sampling_method_name} sampling method...")
    
    # ============ Spatial Entropy Evaluation ============
    print("Computing spatial entropy...")
    
    # Parallel initial sampling for entropy calculation
    entropy_args = [(processed_complete_df, sample_size, processed_not_feature_col, 
                     sampling_function, distance_metric, None) for _ in range(n_initial_samples)]
    
    print(f"Running {n_initial_samples} initial entropy computations in parallel...")
    initial_entropies = _parallel_computation(_compute_single_entropy, entropy_args, n_workers)
    
    # Calculate mean and standard deviation
    mean_entropy = np.mean(initial_entropies)
    std_entropy = np.std(initial_entropies, ddof=1)
    
    # Calculate expected standard deviation
    expected_std_entropy = abs(mean_entropy) * deviation
    
    # Calculate required number of samples
    required_samples_entropy = max(n_initial_samples, int(np.ceil((z_value**2 * std_entropy**2) / (expected_std_entropy**2))))
    
    print(f'{sampling_method_name} sampling entropy - Initial std: {std_entropy:.6f}, Expected std: {expected_std_entropy:.6f}')
    print(f'{sampling_method_name} sampling entropy required samples: {required_samples_entropy}')
    
    # Additional sampling
    additional_samples_entropy = max(0, required_samples_entropy - n_initial_samples)
    all_entropies = initial_entropies.copy()
    
    if additional_samples_entropy > 0:
        print(f"Running {additional_samples_entropy} additional entropy computations in parallel...")
        additional_entropy_args = [(processed_complete_df, sample_size, processed_not_feature_col, 
                                    sampling_function, distance_metric, None) for _ in range(additional_samples_entropy)]
        additional_entropies = _parallel_computation(_compute_single_entropy, additional_entropy_args, n_workers)
        all_entropies.extend(additional_entropies)
    
    final_mean_entropy = np.mean(all_entropies)
    print(f'{sampling_method_name} sampling final mean entropy: {final_mean_entropy:.6f}')
    
    # ============ Mean Distance Evaluation ============
    # Determine distance measure name for output
    distance_measure_name = "Mean Distance" if distance_metric == 'manhattan' else "Mean Squared Distance"
    print(f"Computing {distance_measure_name.lower()}...")
    
    # Parallel initial sampling for mean distance calculation  
    msd_args = [(processed_complete_df, sample_size, processed_not_feature_col, 
                 sampling_function, distance_metric, None) for _ in range(n_initial_samples)]
    
    print(f"Running {n_initial_samples} initial {distance_measure_name} computations in parallel...")
    initial_mean_distances = _parallel_computation(_compute_single_msd, msd_args, n_workers)
    
    # Calculate mean and standard deviation
    mean_msd = np.mean(initial_mean_distances)
    std_msd = np.std(initial_mean_distances, ddof=1)
    
    # Calculate expected standard deviation
    expected_std_msd = abs(mean_msd) * deviation
    
    # Calculate required number of samples
    required_samples_msd = max(n_initial_samples, int(np.ceil((z_value**2 * std_msd**2) / (expected_std_msd**2))))
    
    print(f'{sampling_method_name} sampling {distance_measure_name} - Initial std: {std_msd:.6f}, Expected std: {expected_std_msd:.6f}')
    print(f'{sampling_method_name} sampling {distance_measure_name} required samples: {required_samples_msd}')
    
    # Additional sampling
    additional_samples_msd = max(0, required_samples_msd - n_initial_samples)
    all_msds = initial_mean_distances.copy()
    
    if additional_samples_msd > 0:
        print(f"Running {additional_samples_msd} additional {distance_measure_name} computations in parallel...")
        additional_msd_args = [(processed_complete_df, sample_size, processed_not_feature_col, 
                                sampling_function, distance_metric, None) for _ in range(additional_samples_msd)]
        additional_msds = _parallel_computation(_compute_single_msd, additional_msd_args, n_workers)
        all_msds.extend(additional_msds)
    
    final_mean_msd = np.mean(all_msds)
    print(f'{sampling_method_name} sampling final {distance_measure_name.lower()}: {final_mean_msd:.6f}')
    
    # Return results
    return {
        'method': sampling_method_name,
        'sample_size': sample_size,
        'entropy': {
            'mean': final_mean_entropy,
            'std': np.std(all_entropies, ddof=1),
            'samples_used': len(all_entropies)
        },
        'mean_distance': {
            'mean': final_mean_msd,
            'std': np.std(all_msds, ddof=1),
            'samples_used': len(all_msds)
        }
    }


def compare_sampling_methods(complete_space_df, sample_size, not_feature_col, 
                           methods=['cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'random', 'weighted_itr_cvt', 'ward', 'kennard_stone'],
                           deviation=0.05, n_initial_samples=10, distance_metrics=['euclidean'], n_workers=None):
    """
    Compare spatial entropy and mean squared distance for multiple sampling methods.
    
    Parameters:
    -----------
    complete_space_df : pandas.DataFrame
        Complete space DataFrame
    sample_size : int
        Number of sampling points
    not_feature_col : list
        List of non-feature columns to exclude
    methods : list, default=['cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'random', 'weighted_itr_cvt', 'ward', 'kennard_stone']
        List of sampling methods to compare
    deviation : float, default=0.05
        Expected deviation
    n_initial_samples : int, default=10
        Number of initial samples
    distance_metrics : list, default=['euclidean']
        List of distance metrics for sampling methods, supports 'euclidean', 'manhattan', 'cosine', 'tanimoto'
    n_workers : int, optional
        Number of worker processes for parallel computation. If None, uses all available CPUs
    
    Returns:
    --------
    list
        List containing evaluation results for all sampling methods
    """
    results = []
    
    for distance_metric in distance_metrics:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating sampling method: {method.upper()} with {distance_metric.upper()} distance")
            print(f"{'='*60}")
            
            try:
                result = evaluate_sampling_method(
                    complete_space_df=complete_space_df,
                    sample_size=sample_size,
                    not_feature_col=not_feature_col,
                    sampling_method_name=method,
                    deviation=deviation,
                    n_initial_samples=n_initial_samples,
                    distance_metric=distance_metric,
                    n_workers=n_workers
                )
                # Add distance metric to result for identification
                result['distance_metric'] = distance_metric
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {method} method with {distance_metric} distance: {str(e)}")
                continue
    
    return results


def print_comparison_results(results):
    """
    Print sampling method comparison results.
    
    Parameters:
    -----------
    results : list
        Results list returned by compare_sampling_methods function
    """
    if not results:
        print("No comparison results available")
        return
    
    print(f"\n{'='*80}")
    print("Sampling Method Comparison Results")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'Method':<12} {'Distance':<12} {'Entropy':<15} {'Entropy Std':<15} {'Mean Distance':<15} {'Distance Std':<15}")
    print(f"{'-'*90}")
    
    for result in results:
        method = result['method']
        distance = result.get('distance_metric', 'euclidean')
        entropy_mean = result['entropy']['mean']
        entropy_std = result['entropy']['std']
        msd_mean = result['mean_distance']['mean']
        msd_std = result['mean_distance']['std']
        
        print(f"{method:<12} {distance:<12} {entropy_mean:<15.6f} {entropy_std:<15.6f} {msd_mean:<15.6f} {msd_std:<15.6f}")
    
    # Find best methods
    print(f"\n{'='*80}")
    print("Best Methods by Distance Metric:")
    
    # Group results by distance metric
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for result in results:
        distance_metric = result.get('distance_metric', 'euclidean')
        grouped_results[distance_metric].append(result)
    
    for distance_metric, metric_results in grouped_results.items():
        print(f"\n{distance_metric.upper()} Distance:")
        # Method with highest entropy
        best_entropy = max(metric_results, key=lambda x: x['entropy']['mean'])
        print(f"  Highest entropy: {best_entropy['method']} (entropy: {best_entropy['entropy']['mean']:.6f})")
        
        # Method with lowest mean distance
        best_msd = min(metric_results, key=lambda x: x['mean_distance']['mean'])
        print(f"  Lowest mean distance: {best_msd['method']} (distance: {best_msd['mean_distance']['mean']:.6f})")


def plot_pareto_frontier(results, save_path=None):
    """
    Plot Pareto frontier scatter plot for entropy and negative mean squared distance using seaborn, 
    with separate plots for each distance metric.
    
    Parameters:
    -----------
    results : list
        Results list returned by compare_sampling_methods function
    save_path : str, optional
        Path prefix to save the plots. If None, plots will be displayed but not saved.
    """
    if not results:
        print("No results available for plotting")
        return
    
    # Group results by distance metric
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for result in results:
        distance_metric = result.get('distance_metric', 'euclidean')
        grouped_results[distance_metric].append(result)
    
    # Create subplots for each distance metric
    n_metrics = len(grouped_results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    for idx, (distance_metric, metric_results) in enumerate(grouped_results.items()):
        ax = axes[idx]
        
        # Extract data for plotting
        data = []
        for result in metric_results:
            data.append({
                'method': result['method'].upper(),
                'entropy': result['entropy']['mean'],
                'neg_msd': -result['mean_distance']['mean'],  # Negative for maximization
                'msd': result['mean_distance']['mean']
            })
        
        df = pd.DataFrame(data)
        
        # Create scatter plot using seaborn
        sns.scatterplot(data=df, x='entropy', y='neg_msd', hue='method', 
                       s=150, alpha=0.8, edgecolor='black', linewidth=1, ax=ax)
        
        # Identify Pareto frontier points
        entropies = df['entropy'].values
        neg_msds = df['neg_msd'].values
        methods = df['method'].values
        
        pareto_indices = []
        for i in range(len(entropies)):
            is_pareto = True
            for j in range(len(entropies)):
                if i != j:
                    # Point j dominates point i if it's better in both objectives
                    if entropies[j] >= entropies[i] and neg_msds[j] >= neg_msds[i]:
                        if entropies[j] > entropies[i] or neg_msds[j] > neg_msds[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        # Sort Pareto points by entropy for connecting line
        pareto_points = [(entropies[i], neg_msds[i]) for i in pareto_indices]
        pareto_points.sort(key=lambda x: x[0])
        
        # Draw Pareto frontier line
        if len(pareto_points) > 1:
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(pareto_x, pareto_y, 'r--', linewidth=3, alpha=0.8, label='Pareto Frontier')
        
        # Highlight Pareto optimal points
        pareto_df = df.iloc[pareto_indices]
        if not pareto_df.empty:
            ax.scatter(pareto_df['entropy'], pareto_df['neg_msd'], 
                       c='red', s=200, alpha=0.9, marker='*', 
                       edgecolors='black', linewidth=2, 
                       label='Pareto Optimal', zorder=5)
        
        # Customize plot
        ax.set_xlabel('Entropy (Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Negative Mean Distance\\n(Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_title(f'Pareto Frontier: {distance_metric.upper()} Distance', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations for each point
        for i, row in df.iterrows():
            ax.annotate(row['method'], (row['entropy'], row['neg_msd']), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Improve legend
        handles, labels = ax.get_legend_handles_labels()
        if len(pareto_points) > 1:
            # Add Pareto frontier to legend if it exists
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), 
                      loc='upper left', fontsize=10, frameon=True, 
                      fancybox=True, shadow=True)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                      fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        ax.grid(True, alpha=0.3)
        
        # Print Pareto optimal methods for this distance metric
        print(f"\\n{'='*70}")
        print(f"Pareto Optimal Methods for {distance_metric.upper()} Distance:")
        print(f"{'='*70}")
        print(f"{'Method':<15} | {'Entropy':<12} | {'Mean Distance':<12}")
        print(f"{'-'*70}")
        for i in pareto_indices:
            method = methods[i]
            entropy = entropies[i]
            original_msd = -neg_msds[i]
            print(f"{method:<15} | {entropy:<12.6f} | {original_msd:<12.6f}")
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_pareto_frontiers.png", dpi=300, bbox_inches='tight')
        print(f"\nPareto frontier plots saved to: {save_path}_pareto_frontiers.png")
    else:
        plt.show()


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    # Example usage
    import time
    start_time = time.time()
    print("Starting sampling method comparison test...")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Read test data
    try:
        # Read SMILES data and feature data
        smiles_df = pd.read_csv('zinc_aryl_aldehyde.csv')
        features_df = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')
        
        # Merge SMILES and feature data
        complete_space_df = pd.concat([smiles_df, features_df], axis=1)
        print(f"Complete space data shape: {complete_space_df.shape}")
        
        # Set parameters
        sample_size = 50
        not_feature_col = ['smiles']
        methods_to_test = ['random', 'cvt', 'cvt_gold', 'fps', 'lhs', 'sobol', 'weighted_itr_cvt', 'ward', 'kennard_stone']
        distance_metrics_to_test = ['euclidean', 'manhattan', 'cosine']
        
        # Compare sampling methods with all distance metrics
        results = compare_sampling_methods(
            complete_space_df=complete_space_df,
            sample_size=sample_size,
            not_feature_col=not_feature_col,
            methods=methods_to_test,
            distance_metrics=distance_metrics_to_test,
            deviation=0.05,
            n_initial_samples=10,
            n_workers=None  # Use all available CPUs
        )
        
        # Print comparison results
        print_comparison_results(results)
        
        # Plot Pareto frontier
        print(f"\n{'='*80}")
        print("Generating Pareto Frontier Plots...")
        print(f"{'='*80}")
        plot_pareto_frontier(results, save_path='pareto_frontier_plot')
        
        print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
        
    except FileNotFoundError as e:
        print(f"Test file not found: {e}")
        print("Please ensure zinc_aryl_aldehyde.csv and fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv files exist")
    except Exception as e:
        print(f"Error during testing: {e}")