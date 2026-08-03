# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_rmse_results(log_file):
    """从model_rmse_test.log解析RMSE结果"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取最终结果部分
    final_section = content.split('=== 最终结果 ===')[1].split('=== 实验完成 ===')[0]
    
    # 找到采样方法性能对比表格部分
    table_start = final_section.find('采样方法性能对比:')
    table_section = final_section[table_start:]
    
    # 解析每个方法的RMSE值
    data = []
    lines = table_section.strip().split('\n')
    
    for line in lines:
        if 'weighted_itr_cvt' in line or 'ward_clustering' in line or 'kennard_stone' in line or 'lhs' in line or 'sobol' in line:
            parts = line.split()
            method = parts[0]
            # 提取All RMSEs列中的所有RMSE值
            rmse_start = line.find(parts[3])  # 跳过Method, Mean RMSE, Std RMSE
            rmse_part = line[rmse_start + len(parts[3]):]  # 获取All RMSEs部分
            rmse_match = re.search(r'(\d+\.\d+(?:, \d+\.\d+)*)', rmse_part)
            if rmse_match:
                rmse_values = [float(x.strip()) for x in rmse_match.group(1).split(',')]
                for rmse in rmse_values:
                    data.append({'Method': method, 'RMSE': rmse})
    
    return pd.DataFrame(data)

def create_barplot(log_file='model_rmse_test.log'):
    """创建四种方法的柱形图（不包含Sobol）"""
    # 解析数据
    df = parse_rmse_results(log_file)
    
    # 过滤掉Sobol方法，保持原有的四种方法
    df = df[df['Method'] != 'sobol']
    
    # 重命名方法名称
    method_name_map = {
        'weighted_itr_cvt': 'ScopeMap',
        'lhs': 'LHS',
        'ward_clustering': 'WARD',
        'kennard_stone': 'K_S'
    }
    df['Method'] = df['Method'].map(method_name_map)
    
    # 计算每个方法的平均值和标准差
    stats = df.groupby('Method')['RMSE'].agg(['mean', 'std']).reset_index()
    
    # 设置图形样式
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    # 创建柱形图
    ax = plt.bar(stats['Method'], stats['mean'], 
                 yerr=stats['std'], capsize=5, width=0.6,
                 color=['#D87792', '#2C75E3', '#29ABBA', '#CACACA'],
                 edgecolor=['#CA486B', '#172C51', '#228E9B', '#A7A7A7'], linewidth=3)
    
    # 设置标题和标签
    plt.xlabel('Sampling method', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)
    
    # 旋转x轴标签以便更好显示
    plt.xticks(rotation=0, ha='center', fontsize=24)
    plt.yticks(fontsize=24)
    
    # 使用默认刻度位置，设置y轴范围和刻度
    current_ax = plt.gca()
    current_ax.set_ylim(10, 25)
    current_ax.set_yticks(range(12, 26, 4))
    
    # 保留外边框，设置为黑色
    current_ax.spines['top'].set_visible(True)
    current_ax.spines['right'].set_visible(True)
    current_ax.spines['bottom'].set_visible(True)
    current_ax.spines['left'].set_visible(True)
    current_ax.spines['top'].set_color('black')
    current_ax.spines['right'].set_color('black')
    current_ax.spines['bottom'].set_color('black')
    current_ax.spines['left'].set_color('black')
    current_ax.grid(False, axis='y')
    current_ax.grid(False, axis='x')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('sampling_methods_barplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('sampling_methods_barplot.svg', bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    # 打印统计信息
    print("统计信息（不含Sobol）:")
    print(stats)

def create_barplot_with_sobol(log_file='model_rmse_test.log'):
    """创建包含Sobol在内的五种方法的柱形图"""
    # 解析数据
    df = parse_rmse_results(log_file)
    
    # 重命名方法名称
    method_name_map = {
        'weighted_itr_cvt': 'ScopeMap',
        'lhs': 'LHS',
        'ward_clustering': 'WARD',
        'kennard_stone': 'K_S',
        'sobol': 'Sobol'
    }
    df['Method'] = df['Method'].map(method_name_map)
    
    # 计算每个方法的平均值和标准差
    stats = df.groupby('Method')['RMSE'].agg(['mean', 'std']).reset_index()
    
    # 设置图形样式
    plt.figure(figsize=(9, 6))
    sns.set_style("whitegrid")
    
    # 创建柱形图
    ax = plt.bar(stats['Method'], stats['mean'], 
                 yerr=stats['std'], capsize=5, width=0.6,
                 color=['#D87792', '#2C75E3', '#29ABBA', '#CACACA', '#C5E0B4'],
                 edgecolor=['#CA486B', '#172C51', '#228E9B', '#A7A7A7', '#70AD47'], linewidth=3)
    
    # 设置标题和标签
    plt.xlabel('Sampling method', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)
    
    # 旋转x轴标签以便更好显示
    plt.xticks(rotation=0, ha='center', fontsize=24)
    plt.yticks(fontsize=24)
    
    # 使用默认刻度位置，设置y轴范围和刻度
    current_ax = plt.gca()
    current_ax.set_ylim(10, 25)
    current_ax.set_yticks(range(12, 26, 4))
    
    # 保留外边框，设置为黑色
    current_ax.spines['top'].set_visible(True)
    current_ax.spines['right'].set_visible(True)
    current_ax.spines['bottom'].set_visible(True)
    current_ax.spines['left'].set_visible(True)
    current_ax.spines['top'].set_color('black')
    current_ax.spines['right'].set_color('black')
    current_ax.spines['bottom'].set_color('black')
    current_ax.spines['left'].set_color('black')
    current_ax.grid(False, axis='y')
    current_ax.grid(False, axis='x')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('sampling_methods_barplot_with_sobol.png', dpi=300, bbox_inches='tight')
    plt.savefig('sampling_methods_barplot_with_sobol.svg', bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    # 打印统计信息
    print("统计信息（含Sobol）:")
    print(stats)

if __name__ == "__main__":
    # 创建原有的四种方法对比图
    create_barplot()
    
    # 创建包含Sobol的五种方法对比图
    create_barplot_with_sobol()