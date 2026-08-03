# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def read_cached_results():
    """
    读取1700_exp_sampling.py生成的缓存文件
    """
    results = {}
    
    # 查找所有的采样结果文件
    result_files = glob.glob('*_sampling_results.csv')
    
    for file in result_files:
        method_name = file.replace('_sampling_results.csv', '')
        results[method_name] = pd.read_csv(file)
        
    return results

def plot_single_method_boxplot(data_df, method_name):
    """
    为单个采样方法创建箱型图
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data_df, palette='Set2')
    plt.xlabel('Number of samples', fontsize=30)
    plt.ylabel('Conversion rate', fontsize=30)
    plt.tick_params(axis='both', labelsize=27)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(f'{method_name}_conv_distribution_cached.png', dpi=300)
    plt.savefig(f'{method_name}_conv_distribution_cached.svg')
    plt.show()

def plot_combined_methods(all_results):
    """
    绘制所有方法的合并箱型图
    """
    # 准备数据，只使用采样点数据（排除full列）
    combined_data = []
    method_labels = []
    sample_size_labels = []
    
    for method_name, data_df in all_results.items():
        # 移除full列，只保留采样数据
        sampling_cols = [col for col in data_df.columns if col != 'full']
        
        for col in sampling_cols:
            for value in data_df[col]:
                combined_data.append(value)
                method_labels.append(method_name)
                sample_size_labels.append(col)
    
    # 创建DataFrame
    plot_df = pd.DataFrame({
        'Conversion Rate': combined_data,
        'Method': method_labels,
        'Sample Size': sample_size_labels
    })
    
    # 绘制分面箱型图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    methods = list(all_results.keys())
    
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df['Method'] == method]
        
        # 为每个方法创建子图
        ax = axes[i]
        method_pivot = all_results[method].drop(columns=['full'] if 'full' in all_results[method].columns else [])
        sns.boxplot(data=method_pivot, ax=ax, palette='Set2')
        
        ax.set_xlabel('Sample Size', fontsize=18)
        ax.set_ylabel('Conversion Rate', fontsize=18)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_ylim(-0.1, 1.1)
    
    # 如果方法少于6个，隐藏多余的子图
    for i in range(len(methods), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison_cached.png', dpi=300)
    plt.savefig('all_methods_comparison_cached.svg')
    plt.show()

if __name__ == "__main__":
    # 读取缓存的结果
    print("Reading cached sampling results...")
    all_results = read_cached_results()
    
    if not all_results:
        print("No cached results found. Please run 1700_exp_sampling.py first.")
        exit(1)
    
    print(f"Found results for: {list(all_results.keys())}")
    
    # 为每个方法绘制单独的箱型图
    for method_name, data_df in all_results.items():
        print(f"Plotting {method_name}...")
        plot_single_method_boxplot(data_df, method_name)
    
    # 绘制合并的比较图
    print("Creating combined comparison plot...")
    plot_combined_methods(all_results)
    
    print("All plots have been generated from cached data.")