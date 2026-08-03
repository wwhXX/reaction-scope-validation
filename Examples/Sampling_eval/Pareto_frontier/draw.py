# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from adjustText import adjust_text


def parse_log_data(log_file_path, is_pca=False):
    """
    Parse log file to extract comparison results data
    
    Parameters:
    -----------
    log_file_path : str
        Path to log file
    is_pca : bool
        Whether this is PCA results (has both original and reduced space)
    
    Returns:
    --------
    dict
        Dictionary containing parsed data by distance metric
    """
    results = defaultdict(list)
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if is_pca:
        # For PCA log, extract original space results
        # Look for the "原始空间结果:" section
        pattern = r'原始空间结果:\s*\n方法.*?\n-+\n(.*?)(?=\n降维空间结果:|\n========|$)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            table_content = match.group(1).strip()
            lines = table_content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-'):
                    # Parse each data line: method distance entropy entropy_std mean_distance distance_std samples_used
                    parts = line.split()
                    if len(parts) >= 7:
                        method = parts[0]
                        distance = parts[1]
                        try:
                            entropy = float(parts[2])
                            entropy_std = float(parts[3])
                            mean_distance = float(parts[4])
                            distance_std = float(parts[5])
                            
                            results[distance].append({
                                'method': method,
                                'entropy': entropy,
                                'entropy_std': entropy_std,
                                'mean_distance': mean_distance,
                                'distance_std': distance_std
                            })
                        except ValueError:
                            continue
    else:
        # For regular log, extract from main comparison table
        pattern = r'Method\s+Distance\s+Entropy\s+Entropy Std\s+Mean Distance\s+Distance Std\s+(.*?)(?=Best Methods|$)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            table_content = match.group(1).strip()
            lines = table_content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-'):
                    # Parse each data line
                    parts = line.split()
                    if len(parts) >= 6:  # method, distance, entropy, entropy_std, mean_distance, distance_std
                        method = parts[0]
                        distance = parts[1]
                        try:
                            entropy = float(parts[2])
                            entropy_std = float(parts[3])
                            mean_distance = float(parts[4])
                            distance_std = float(parts[5])
                            
                            results[distance].append({
                                'method': method,
                                'entropy': entropy,
                                'entropy_std': entropy_std,
                                'mean_distance': mean_distance,
                                'distance_std': distance_std
                            })
                        except ValueError:
                            continue
    
    return results


def plot_comparison(original_data, pca_data, distance_metric='euclidean', save_path=None):
    """
    Plot comparison between original space and PCA space results in a single plot
    
    Parameters:
    -----------
    original_data : dict
        Results from original space
    pca_data : dict
        Results from PCA space
    distance_metric : str
        Distance metric to plot ('euclidean', 'manhattan', 'cosine')
    save_path : str, optional
        Path to save the plot
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get data for the specified distance metric
    orig_results = original_data.get(distance_metric, [])
    pca_results = pca_data.get(distance_metric, [])
    
    if not orig_results or not pca_results:
        print(f"No data found for {distance_metric} distance metric")
        return
    
    # Prepare data for plotting
    orig_df = pd.DataFrame(orig_results)
    pca_df = pd.DataFrame(pca_results)
    
    # Get unique methods from both datasets
    all_methods = list(set(orig_df['method'].tolist() + pca_df['method'].tolist()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_methods)))
    method_colors = {method: colors[i] for i, method in enumerate(all_methods)}
    
    # Plot original space results (solid circles)
    for _, row in orig_df.iterrows():
        method = row['method']
        ax.scatter(row['entropy'], -row['mean_distance'], 
                  c=[method_colors[method]], s=200, alpha=0.8, 
                  marker='o', edgecolor='black', linewidth=2,
                  label=f'{method.upper()} (Original)')
    
    # Plot PCA space results (triangles)
    for _, row in pca_df.iterrows():
        method = row['method']
        ax.scatter(row['entropy'], -row['mean_distance'], 
                  c=[method_colors[method]], s=200, alpha=0.8,
                  marker='^', edgecolor='black', linewidth=2,
                  label=f'{method.upper()} (PCA)')
    
    # Add Pareto frontiers
    def add_pareto_frontier(df, linestyle, color, label_suffix):
        entropies = df['entropy'].values
        neg_distances = (-df['mean_distance']).values
        
        pareto_indices = []
        for i in range(len(entropies)):
            is_pareto = True
            for j in range(len(entropies)):
                if i != j:
                    if entropies[j] >= entropies[i] and neg_distances[j] >= neg_distances[i]:
                        if entropies[j] > entropies[i] or neg_distances[j] > neg_distances[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        if len(pareto_indices) > 1:
            pareto_points = [(entropies[i], neg_distances[i]) for i in pareto_indices]
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            
            # Draw main line
            line = ax.plot(pareto_x, pareto_y, linestyle=linestyle, linewidth=3, 
                   alpha=0.8, color=color, label=f'Pareto Frontier {label_suffix}')
            return line[0]
        return None
            
    
    orig_lines = add_pareto_frontier(orig_df, '-', 'red', '(Original)')
    pca_lines = add_pareto_frontier(pca_df, '-', 'blue', '(PCA)')
    
    # Add annotations for each point using adjustText to avoid overlaps
    texts_orig = []
    for _, row in orig_df.iterrows():
        text = ax.text(row['entropy'], -row['mean_distance'], 
                      row['method'].upper(),
                      fontsize=14, zorder=10,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        texts_orig.append(text)
    
    # Adjust text positions to avoid overlaps with points and lines
    # Increase force_text to push labels away from each other and points
    adjust_text(texts_orig, ax=ax,
                expand_points=(2.0, 2.0),  # Expand the space around points
                expand_text=(1.5, 1.5),    # Expand the space around text
                force_text=(0.8, 0.8),     # Force between text labels
                force_points=(0.6, 0.6),   # Force between text and points
                lim=500,                   # Maximum iterations
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))
    
    # Customize plot
    ax.set_xlabel('Entropy', fontsize=24, fontweight='bold')
    ax.set_ylabel('Negative MSD', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_xlim(4000, 6500)
    ax.set_ylim(-4000, -500)
    ax.grid(True, alpha=0.3)
    
    # Set solid borders for the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # Create custom legend
    legend_elements = []
    # Add space type markers
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                                    markersize=12, label='Original Space', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='gray', linestyle='None',
                                    markersize=12, label='PCA Space', markeredgecolor='black'))
    # Add Pareto frontier lines
    legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3, 
                                    linestyle='-', label='Pareto Frontier (Original)'))
    legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=3, 
                                    linestyle='-', label='Pareto Frontier (PCA)'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=22, frameon=True, 
             fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_{distance_metric}_unified_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Unified comparison plot saved to: {save_path}_{distance_metric}_unified_comparison.png")
        plt.savefig(f"{save_path}_{distance_metric}_unified_comparison.svg", format='svg', bbox_inches='tight')
        print(f"Unified comparison plot saved to: {save_path}_{distance_metric}_unified_comparison.svg")
    else:
        plt.show()


def plot_comparison_large_font(original_data, pca_data, distance_metric='euclidean', save_path=None):
    """
    Plot comparison between original space and PCA space results with large font sizes
    
    Parameters:
    -----------
    original_data : dict
        Results from original space
    pca_data : dict
        Results from PCA space
    distance_metric : str
        Distance metric to plot ('euclidean', 'manhattan', 'cosine')
    save_path : str, optional
        Path to save the plot
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get data for the specified distance metric
    orig_results = original_data.get(distance_metric, [])
    pca_results = pca_data.get(distance_metric, [])
    
    if not orig_results or not pca_results:
        print(f"No data found for {distance_metric} distance metric")
        return
    
    # Prepare data for plotting
    orig_df = pd.DataFrame(orig_results)
    pca_df = pd.DataFrame(pca_results)
    
    # Get unique methods from both datasets
    all_methods = list(set(orig_df['method'].tolist() + pca_df['method'].tolist()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_methods)))
    method_colors = {method: colors[i] for i, method in enumerate(all_methods)}
    
    # Plot original space results (solid circles)
    for _, row in orig_df.iterrows():
        method = row['method']
        ax.scatter(row['entropy'], -row['mean_distance'], 
                  c=[method_colors[method]], s=400, alpha=0.8, 
                  marker='o', edgecolor='black', linewidth=4,
                  label=f'{method.upper()} (Original)')
    
    # Plot PCA space results (triangles)
    for _, row in pca_df.iterrows():
        method = row['method']
        ax.scatter(row['entropy'], -row['mean_distance'], 
                  c=[method_colors[method]], s=400, alpha=0.8,
                  marker='^', edgecolor='black', linewidth=4,
                  label=f'{method.upper()} (PCA)')
    
    # Add Pareto frontiers
    def add_pareto_frontier(df, linestyle, color, label_suffix):
        entropies = df['entropy'].values
        neg_distances = (-df['mean_distance']).values
        
        pareto_indices = []
        for i in range(len(entropies)):
            is_pareto = True
            for j in range(len(entropies)):
                if i != j:
                    if entropies[j] >= entropies[i] and neg_distances[j] >= neg_distances[i]:
                        if entropies[j] > entropies[i] or neg_distances[j] > neg_distances[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        if len(pareto_indices) > 1:
            pareto_points = [(entropies[i], neg_distances[i]) for i in pareto_indices]
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            
            # Draw main line
            line = ax.plot(pareto_x, pareto_y, linestyle=linestyle, linewidth=6, 
                   alpha=0.8, color=color, label=f'Pareto Frontier {label_suffix}')
            return line[0]
        return None
            
    
    orig_lines = add_pareto_frontier(orig_df, '-', 'red', '(Original)')
    pca_lines = add_pareto_frontier(pca_df, '-', 'blue', '(PCA)')
    
    # Add annotations for each point using adjustText to avoid overlaps
    texts_orig = []
    for _, row in orig_df.iterrows():
        text = ax.text(row['entropy'], -row['mean_distance'], 
                      row['method'].upper(),
                      fontsize=24, zorder=10,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        texts_orig.append(text)
    
    # Adjust text positions to avoid overlaps with points and lines
    # Increase force_text to push labels away from each other and points
    adjust_text(texts_orig, ax=ax,
                expand_points=(20.0, 20.0),  # Expand the space around points
                expand_text=(20.0, 20.0),    # Expand the space around text
                force_text=(1.2, 1.2),     # Force between text labels
                force_points=(1.0, 1.0),   # Force between text and points
                lim=800,                   # Maximum iterations
                arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.7))
    
    # Customize plot
    ax.set_xlabel('Entropy', fontsize=18, fontweight='bold')
    ax.set_ylabel('Negative MSD', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(4000, 6500)
    ax.set_ylim(-4000, -500)
    ax.grid(True, alpha=0.3)
    
    # Set solid borders for the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    
    # Create custom legend positioned at lower left
    legend_elements = []
    # Add space type markers
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                                    markersize=12, label='Original Space', markeredgecolor='black'))
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='gray', linestyle='None',
                                    markersize=12, label='PCA Space', markeredgecolor='black'))
    # Add Pareto frontier lines
    legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3, 
                                    linestyle='-', label='Pareto Frontier (Original)'))
    legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=3, 
                                    linestyle='-', label='Pareto Frontier (PCA)'))
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=24, frameon=True, 
             fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(f"{save_path}_{distance_metric}_large_font_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Large font comparison plot saved to: {save_path}_{distance_metric}_large_font_comparison.png")
        plt.savefig(f"{save_path}_{distance_metric}_large_font_comparison.svg", format='svg', bbox_inches='tight')
        print(f"Large font comparison plot saved to: {save_path}_{distance_metric}_large_font_comparison.svg")
    else:
        plt.show()


def plot_pca_only(pca_data, distance_metric='euclidean', save_path=None):
    """
    Plot PCA space results only
    
    Parameters:
    -----------
    pca_data : dict
        Results from PCA space
    distance_metric : str
        Distance metric to plot ('euclidean', 'manhattan', 'cosine')
    save_path : str, optional
        Path to save the plot
    """
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    pca_results = pca_data.get(distance_metric, [])
    
    if not pca_results:
        print(f"No data found for {distance_metric} distance metric")
        return
    
    pca_df = pd.DataFrame(pca_results)
    
    all_methods = pca_df['method'].unique().tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_methods)))
    method_colors = {method: colors[i] for i, method in enumerate(all_methods)}
    
    for _, row in pca_df.iterrows():
        method = row['method']
        ax.scatter(row['entropy'], -row['mean_distance'], 
                  c=[method_colors[method]], s=400, alpha=0.8,
                  marker='^', edgecolor='black', linewidth=4,
                  label=f'{method.upper()}')
    
    def add_pareto_frontier(df, linestyle, color):
        entropies = df['entropy'].values
        neg_distances = (-df['mean_distance']).values
        
        pareto_indices = []
        for i in range(len(entropies)):
            is_pareto = True
            for j in range(len(entropies)):
                if i != j:
                    if entropies[j] >= entropies[i] and neg_distances[j] >= neg_distances[i]:
                        if entropies[j] > entropies[i] or neg_distances[j] > neg_distances[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        if len(pareto_indices) > 1:
            pareto_points = [(entropies[i], neg_distances[i]) for i in pareto_indices]
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            
            line = ax.plot(pareto_x, pareto_y, linestyle=linestyle, linewidth=6, 
                   alpha=0.8, color=color, label='Pareto Frontier')
            return line[0]
        return None
    
    add_pareto_frontier(pca_df, '-', 'blue')
    
    texts = []
    for _, row in pca_df.iterrows():
        text = ax.text(row['entropy'], -row['mean_distance'], 
                      row['method'].upper(),
                      fontsize=24, zorder=10,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
        texts.append(text)
    
    adjust_text(texts, ax=ax,
                expand_points=(2.0, 2.0),
                expand_text=(1.5, 1.5),
                force_text=(0.8, 0.8),
                force_points=(0.6, 0.6),
                lim=500,
                arrowprops=dict(arrowstyle='->', color='blue', lw=3, alpha=0.7))
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(5000, 5800)
    ax.set_ylim(-2200, -1300)
    ax.grid(True, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_{distance_metric}_pca_only.png", dpi=300, bbox_inches='tight')
        print(f"PCA only plot saved to: {save_path}_{distance_metric}_pca_only.png")
        plt.savefig(f"{save_path}_{distance_metric}_pca_only.svg", format='svg', bbox_inches='tight')
        print(f"PCA only plot saved to: {save_path}_{distance_metric}_pca_only.svg")
    else:
        plt.show()


def plot_pca_only_consistent_font(pca_data, distance_metric='euclidean', save_path=None):
    """
    Plot PCA space results only with consistent font size as unified comparison
    
    Parameters:
    -----------
    pca_data : dict
        Results from PCA space
    distance_metric : str
        Distance metric to plot ('euclidean', 'manhattan', 'cosine')
    save_path : str, optional
        Path to save the plot
    """
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    pca_results = pca_data.get(distance_metric, [])
    
    if not pca_results:
        print(f"No data found for {distance_metric} distance metric")
        return
    
    pca_df = pd.DataFrame(pca_results)
    
    all_methods = pca_df['method'].unique().tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_methods)))
    method_colors = {method: colors[i] for i, method in enumerate(all_methods)}
    
    for _, row in pca_df.iterrows():
        method = row['method']
        ax.scatter(row['entropy'], -row['mean_distance'], 
                  c=[method_colors[method]], s=200, alpha=0.8,
                  marker='^', edgecolor='black', linewidth=2,
                  label=f'{method.upper()}')
    
    def add_pareto_frontier(df, linestyle, color):
        entropies = df['entropy'].values
        neg_distances = (-df['mean_distance']).values
        
        pareto_indices = []
        for i in range(len(entropies)):
            is_pareto = True
            for j in range(len(entropies)):
                if i != j:
                    if entropies[j] >= entropies[i] and neg_distances[j] >= neg_distances[i]:
                        if entropies[j] > entropies[i] or neg_distances[j] > neg_distances[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        if len(pareto_indices) > 1:
            pareto_points = [(entropies[i], neg_distances[i]) for i in pareto_indices]
            pareto_points.sort(key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            
            line = ax.plot(pareto_x, pareto_y, linestyle=linestyle, linewidth=3, 
                   alpha=0.8, color=color, label='Pareto Frontier')
            return line[0]
        return None
    
    add_pareto_frontier(pca_df, '-', 'blue')
    
    texts = []
    for _, row in pca_df.iterrows():
        text = ax.text(row['entropy'], -row['mean_distance'], 
                      row['method'].upper(),
                      fontsize=14, zorder=10,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
        texts.append(text)
    
    adjust_text(texts, ax=ax,
                expand_points=(2.0, 2.0),
                expand_text=(1.5, 1.5),
                force_text=(0.8, 0.8),
                force_points=(0.6, 0.6),
                lim=500,
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.7))

    ax.set_xlabel('Entropy', fontsize=24, fontweight='bold')
    ax.set_ylabel('Negative MSD', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_xlim(5000, 5800)
    ax.set_ylim(-2200, -1300)
    ax.grid(True, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_{distance_metric}_pca_consistent.png", dpi=300, bbox_inches='tight')
        print(f"PCA consistent font plot saved to: {save_path}_{distance_metric}_pca_consistent.png")
        plt.savefig(f"{save_path}_{distance_metric}_pca_consistent.svg", format='svg', bbox_inches='tight')
        print(f"PCA consistent font plot saved to: {save_path}_{distance_metric}_pca_consistent.svg")
    else:
        plt.show()


def print_comparison_stats(original_data, pca_data):
    """
    Print statistical comparison between original and PCA results
    
    Parameters:
    -----------
    original_data : dict
        Results from original space
    pca_data : dict
        Results from PCA space
    """
    print(f"\n{'='*80}")
    print("Statistical Comparison: Original Space vs PCA Space")
    print(f"{'='*80}")
    
    for distance_metric in ['euclidean', 'manhattan', 'cosine']:
        if distance_metric not in original_data or distance_metric not in pca_data:
            continue
            
        orig_results = original_data[distance_metric]
        pca_results = pca_data[distance_metric]
        
        print(f"\n{distance_metric.upper()} Distance:")
        print(f"{'Method':<15} | {'Orig Entropy':<12} | {'PCA Entropy':<12} | {'Orig Distance':<12} | {'PCA Distance':<12}")
        print(f"{'-'*80}")
        
        # Create lookup dictionaries
        orig_dict = {r['method']: r for r in orig_results}
        pca_dict = {r['method']: r for r in pca_results}
        
        # Find common methods
        common_methods = set(orig_dict.keys()) & set(pca_dict.keys())
        
        for method in sorted(common_methods):
            orig = orig_dict[method]
            pca = pca_dict[method]
            
            print(f"{method:<15} | {orig['entropy']:<12.2f} | {pca['entropy']:<12.2f} | "
                  f"{orig['mean_distance']:<12.2f} | {pca['mean_distance']:<12.2f}")


if __name__ == "__main__":
    print("Parsing log files and generating comparison plots...")
    
    # Parse log files
    print("Parsing sampling_eval.log...")
    original_data = parse_log_data('sampling_eval.log', is_pca=False)
    
    print("Parsing pca_sampling_eval.log...")
    pca_data = parse_log_data('pca_sampling_eval.log', is_pca=True)
    
    if not original_data:
        print("Warning: No data found in sampling_eval.log")
    if not pca_data:
        print("Warning: No data found in pca_sampling_eval.log")
    
    # Print statistical comparison
    print_comparison_stats(original_data, pca_data)
    
    # Generate comparison plots for each distance metric
    distance_metrics = ['euclidean', 'manhattan', 'cosine']
    
    for distance_metric in distance_metrics:
        if distance_metric in original_data and distance_metric in pca_data:
            print(f"\nGenerating comparison plot for {distance_metric.upper()} distance...")
            plot_comparison(original_data, pca_data, distance_metric, 
                          save_path='sampling_comparison')
        else:
            print(f"Warning: Missing data for {distance_metric} distance metric")
    
    for distance_metric in distance_metrics:
        if distance_metric in pca_data:
            print(f"\nGenerating PCA only plot for {distance_metric.upper()} distance...")
            plot_pca_only(pca_data, distance_metric, save_path='sampling_pca')
        else:
            print(f"Warning: Missing PCA data for {distance_metric} distance metric")
    
    # Generate PCA only plots with consistent font sizes
    for distance_metric in distance_metrics:
        if distance_metric in pca_data:
            print(f"\nGenerating PCA only plot (consistent font) for {distance_metric.upper()} distance...")
            plot_pca_only_consistent_font(pca_data, distance_metric, save_path='sampling_pca')
        else:
            print(f"Warning: Missing PCA data for {distance_metric} distance metric")
    
    # Generate unified comparison plots with large font sizes
    for distance_metric in distance_metrics:
        if distance_metric in original_data and distance_metric in pca_data:
            print(f"\nGenerating unified comparison plot (large font) for {distance_metric.upper()} distance...")
            plot_comparison_large_font(original_data, pca_data, distance_metric, 
                          save_path='sampling_comparison')
        else:
            print(f"Warning: Missing data for {distance_metric} distance metric")
    
    print("\nComparison analysis complete!")