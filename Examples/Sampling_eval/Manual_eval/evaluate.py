import numpy as np
import pandas as pd
from utils import calc_entropy, calc_mean_squared_distance
from sampling import cvt_sampling_gold_df_norepeat, cvt_sampling_df_norepeat, kennard_stone_sampling_df_norepeat, rand_sampling_df_no_repeat, ward_clustering_df_norepeat

# ==================== 全局配置 ====================
SUBSTRATE_FILE = 'zinc_aryl_aldehyde.csv'
SUBSTRATE_FP_FILE = 'fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv'
EXPERIMENTAL_FILE = '人工.csv'
EXPERIMENTAL_FP_FILE = 'fp_spoc_morgan41024_Maccs_人工_alcohol.csv'

NOT_FEATURE_COLUMNS = ['smiles']
SAMPLING_SIZE = 20
DISTANCE_METRIC = 'euclidean'
K_NEIGHBORS = 5
RANDOM_SEED = 42
RANDOM_SAMPLING_RUNS = 5  # 随机采样重复次数
# ==================================================


def main():
    np.random.seed(RANDOM_SEED)
    
    print("=" * 80)
    print("开始评估采样质量")
    print("=" * 80)
    
    # ========== 第一部分：CVT和Kennard-Stone采样评估 ==========
    print("\n【第一部分：CVT和Kennard-Stone采样评估】")
    print("-" * 80)
    
    # 1. 读取底物醇数据和特征
    print(f"\n1. 读取数据文件...")
    substrate_df = pd.read_csv(SUBSTRATE_FILE)
    substrate_fp_df = pd.read_csv(SUBSTRATE_FP_FILE)
    print(f"   - 底物醇数据: {substrate_df.shape}")
    print(f"   - 底物醇特征: {substrate_fp_df.shape}")
    
    # 2. 合并数据
    complete_space_df = pd.concat([substrate_df, substrate_fp_df], axis=1)
    print(f"   - 合并后完整空间: {complete_space_df.shape}")
    
    # 3. CVT采样
    print(f"\n2. 进行CVT采样 (采样数量: {SAMPLING_SIZE})...")
    cvt_sampling_df, _ = cvt_sampling_df_norepeat(
        data=complete_space_df,
        k=SAMPLING_SIZE,
        not_feature_columns=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - CVT采样完成，采样点数量: {len(cvt_sampling_df)}")
    
    # 4. 计算CVT采样的熵和MSD
    print(f"\n3. 计算CVT采样的评估指标...")
    cvt_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=cvt_sampling_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        k=K_NEIGHBORS,
        distance_metric=DISTANCE_METRIC
    )
    cvt_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=cvt_sampling_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - CVT采样熵 (Entropy): {cvt_entropy:.6f}")
    print(f"   - CVT采样MSD: {cvt_msd:.6f}")
    
    # 5. Kennard-Stone采样
    print(f"\n4. 进行Kennard-Stone采样 (采样数量: {SAMPLING_SIZE})...")
    ks_sampling_df, _ = kennard_stone_sampling_df_norepeat(
        data=complete_space_df,
        k=SAMPLING_SIZE,
        not_feature_columns=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - Kennard-Stone采样完成，采样点数量: {len(ks_sampling_df)}")
    
    # 6. 计算Kennard-Stone采样的熵和MSD
    print(f"\n5. 计算Kennard-Stone采样的评估指标...")
    ks_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=ks_sampling_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        k=K_NEIGHBORS,
        distance_metric=DISTANCE_METRIC
    )
    ks_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=ks_sampling_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - Kennard-Stone采样熵 (Entropy): {ks_entropy:.6f}")
    print(f"   - Kennard-Stone采样MSD: {ks_msd:.6f}")
    
    # 6.5. Ward聚类采样
    print(f"\n5.5. 进行Ward聚类采样 (采样数量: {SAMPLING_SIZE})...")
    ward_sampling_df, _ = ward_clustering_df_norepeat(
        data=complete_space_df,
        k=SAMPLING_SIZE,
        not_feature_columns=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - Ward聚类采样完成，采样点数量: {len(ward_sampling_df)}")
    
    # 6.6. 计算Ward聚类采样的熵和MSD
    print(f"\n5.6. 计算Ward聚类采样的评估指标...")
    ward_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=ward_sampling_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        k=K_NEIGHBORS,
        distance_metric=DISTANCE_METRIC
    )
    ward_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=ward_sampling_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - Ward聚类采样熵 (Entropy): {ward_entropy:.6f}")
    print(f"   - Ward聚类采样MSD: {ward_msd:.6f}")
    
    # 7. 随机采样（多次取平均）
    print(f"\n6. 进行随机采样 (采样数量: {SAMPLING_SIZE}, 重复{RANDOM_SAMPLING_RUNS}次)...")
    rand_entropies = []
    rand_msds = []
    
    for i in range(RANDOM_SAMPLING_RUNS):
        # 每次使用不同的随机种子
        np.random.seed(RANDOM_SEED + i)
        rand_sampling_df, _ = rand_sampling_df_no_repeat(
            data=complete_space_df,
            k=SAMPLING_SIZE,
            not_feature_columns=NOT_FEATURE_COLUMNS,
            distance_metric=DISTANCE_METRIC
        )
        
        # 计算熵和MSD
        rand_entropy = calc_entropy(
            complete_space_df=complete_space_df,
            sampling_points_df=rand_sampling_df,
            not_feature_col=NOT_FEATURE_COLUMNS,
            k=K_NEIGHBORS,
            distance_metric=DISTANCE_METRIC
        )
        rand_msd = calc_mean_squared_distance(
            complete_space_df=complete_space_df,
            sampling_points_df=rand_sampling_df,
            not_feature_col=NOT_FEATURE_COLUMNS,
            distance_metric=DISTANCE_METRIC
        )
        
        rand_entropies.append(rand_entropy)
        rand_msds.append(rand_msd)
        print(f"   - 第{i+1}次: 熵={rand_entropy:.6f}, MSD={rand_msd:.6f}")
    
    # 计算平均值和标准差
    rand_entropy_mean = np.mean(rand_entropies)
    rand_entropy_std = np.std(rand_entropies, ddof=1)
    rand_msd_mean = np.mean(rand_msds)
    rand_msd_std = np.std(rand_msds, ddof=1)
    
    print(f"\n7. 随机采样统计结果:")
    print(f"   - 平均熵: {rand_entropy_mean:.6f} ± {rand_entropy_std:.6f}")
    print(f"   - 平均MSD: {rand_msd_mean:.6f} ± {rand_msd_std:.6f}")
    
    # ========== 第二部分：实验数据评估 ==========
    print("\n\n【第二部分：实验数据评估】")
    print("-" * 80)
    
    # 8. 读取实验数据
    print(f"\n8. 读取实验数据文件...")
    experimental_df = pd.read_csv(EXPERIMENTAL_FILE)
    experimental_fp_df = pd.read_csv(EXPERIMENTAL_FP_FILE)
    print(f"   - 实验数据: {experimental_df.shape}")
    print(f"   - 实验数据特征: {experimental_fp_df.shape}")
    
    # 9. 合并实验数据
    # 处理列名差异：实验数据中是'SMILES'（大写），需要统一为'smiles'（小写）
    if 'SMILES' in experimental_df.columns:
        experimental_df = experimental_df.rename(columns={'SMILES': 'smiles'})
    experimental_combined_df = pd.concat([experimental_df[NOT_FEATURE_COLUMNS], experimental_fp_df], axis=1)
    print(f"   - 合并后实验数据: {experimental_combined_df.shape}")
    
    # 10. 计算实验采样的熵和MSD
    print(f"\n9. 计算实验采样的评估指标...")
    exp_entropy = calc_entropy(
        complete_space_df=complete_space_df,
        sampling_points_df=experimental_combined_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        k=K_NEIGHBORS,
        distance_metric=DISTANCE_METRIC
    )
    exp_msd = calc_mean_squared_distance(
        complete_space_df=complete_space_df,
        sampling_points_df=experimental_combined_df,
        not_feature_col=NOT_FEATURE_COLUMNS,
        distance_metric=DISTANCE_METRIC
    )
    print(f"   - 实验采样熵 (Entropy): {exp_entropy:.6f}")
    print(f"   - 实验采样MSD: {exp_msd:.6f}")
    
    # ========== 总结输出 ==========
    print("\n\n" + "=" * 80)
    print("评估结果总结")
    print("=" * 80)
    print(f"\n{'方法':<25} {'熵 (Entropy)':<30} {'MSD':<20}")
    print("-" * 80)
    print(f"{'CVT采样':<25} {cvt_entropy:<30.6f} {cvt_msd:<20.6f}")
    print(f"{'Kennard-Stone采样':<25} {ks_entropy:<30.6f} {ks_msd:<20.6f}")
    print(f"{'Ward聚类采样':<25} {ward_entropy:<30.6f} {ward_msd:<20.6f}")
    print(f"{'随机采样(平均)':<25} {rand_entropy_mean:<30.6f} {rand_msd_mean:<20.6f}")
    print(f"{'随机采样(标准差)':<25} {rand_entropy_std:<30.6f} {rand_msd_std:<20.6f}")
    print(f"{'实验采样':<25} {exp_entropy:<30.6f} {exp_msd:<20.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
