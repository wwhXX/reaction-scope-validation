import numpy as np
import pandas as pd
from sampling import (
    cvt_sampling_df_norepeat,
    lhs_sampling_df_norepeat,
    rand_sampling_df_no_repeat,
    sobol_sampling_df_norepeat,
    kennard_stone_sampling_df_norepeat,
    ward_clustering_df_norepeat
)
import seaborn as sns
import matplotlib.pyplot as plt


def test_sampling_method(data, sampling_function, method_name, not_feature_cols):
    """
    测试单个采样方法并返回结果数据
    """
    raw_df = pd.DataFrame()
    
    # 执行采样
    for i in range(5):
        k = (i + 1) * 20 
        sampled_points, unsampled_points = sampling_function(
            data=data,
            k=k,
            not_feature_columns=not_feature_cols
        )
        # 保存采样结果
        sampled_points.to_csv(f'{method_name}_sampled_points_{k}k.csv', index=False)
        
        # 处理conv数据
        input_col = sampled_points['conv'].tolist()
        input_col = [float(x.split('%')[0])/100.0 for x in input_col]
        input_col = pd.DataFrame(input_col, columns=[f'{20*(i+1)}'])
        raw_df = pd.concat([raw_df, input_col], axis=1)
    
    return raw_df

def plot_single_method(data_df, method_name, full_data):
    """
    为单个采样方法创建箱型图
    """
    # 添加完整数据集作为对比
    plot_data = pd.concat([data_df, pd.DataFrame(full_data, columns=['full'])], axis=1)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=plot_data, palette='Set2')
    plt.title(f'{method_name} Sampling - Conversion Rate Distribution', fontsize=20)
    plt.xlabel('Number of samples', fontsize=20)
    plt.ylabel('Conversion rate', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(f'{method_name}_conv_distribution.png', dpi=300)
    plt.savefig(f'{method_name}_conv_distribution.svg')
    plt.show()
    
    return plot_data

if __name__ == "__main__":
    np.random.seed(42)
    data = pd.read_csv('1700_final_norepeat.csv')
    not_feature_cols = ['smiles', 'conv']
    
    # 准备完整数据集的转化率数据
    full_df = pd.read_csv('1700_final_norepeat.csv')
    full_data = full_df['conv'].tolist()
    full_data = [float(x.strip("%"))/100.0 for x in full_data]
    
    # 测试所有采样方法
    sampling_methods = [
        (cvt_sampling_df_norepeat, 'CVT'),
        (lhs_sampling_df_norepeat, 'LHS'),
        (rand_sampling_df_no_repeat, 'Random'),
        (sobol_sampling_df_norepeat, 'Sobol'),
        (kennard_stone_sampling_df_norepeat, 'KennardStone'),
        (ward_clustering_df_norepeat, 'WardClustering')
    ]
    
    all_results = {}
    
    for sampling_func, method_name in sampling_methods:
        print(f"Testing {method_name} sampling...")
        
        try:
            # 测试采样方法
            result_df = test_sampling_method(data, sampling_func, method_name, not_feature_cols)
            
            # 创建单独的图
            plot_data = plot_single_method(result_df, method_name, full_data)
            
            # 保存结果
            all_results[method_name] = plot_data
            plot_data.to_csv(f'{method_name}_sampling_results.csv', index=False)
            
            print(f"{method_name} sampling completed and saved.")
            
        except Exception as e:
            print(f"Error testing {method_name}: {str(e)}")
            continue
    
    print("All sampling methods have been tested and plotted separately.")