import aldehyde_reaction_judger
import utils
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import os
import warnings
warnings.filterwarnings('ignore')

max_k_num = 10

# 读取数据
smi_data = pd.read_csv('zinc_aryl_aldehyde.csv')
fp_all_data = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')
all_data_base = pd.concat([smi_data, fp_all_data], axis=1)

exp_raw_data = pd.read_csv('1700_final_norepeat.csv')
print(exp_raw_data.shape)

all_data_base = pd.concat([all_data_base, exp_raw_data], axis=0, ignore_index=True)
all_data_base['ScreenLabel'] = 'BASE'
print(all_data_base)

# 获取特征列
not_feature_columns = ['smiles', 'conv', 'ScreenLabel']
feature_columns = [col for col in all_data_base.columns if col not in not_feature_columns]
feature_columns = [col for col in feature_columns if all_data_base[col].dtype in [np.float64, np.int64]]
print(f"特征列数量: {len(feature_columns)}")

# 创建输出目录
os.makedirs('./itr', exist_ok=True)
os.makedirs('./tpe_sampling', exist_ok=True)


def compute_scaling_factors(all_data, available_data_df, feature_cols, not_feature_cols, n_presamples=10):
    """
    预采样阶段：计算熵和反应产率的缩放因子，使两者具有可比性。

    参数:
        all_data: 完整数据集
        available_data_df: 可用数据集
        feature_cols: 特征列名列表
        not_feature_cols: 非特征列名列表
        n_presamples: 预采样数量

    返回:
        entropy_scale: 熵的缩放因子
        entropy_mean: 熵的均值
        entropy_reference_df: 所有历史预采样点组成的参照DataFrame
        presampled_ok: 预采样中产率>=10的smiles列表
        presampled_drop: 预采样中被排除的smiles列表
    """
    print(f"\n=== 预采样阶段: 计算缩放因子 (n={n_presamples}) ===")

    # 随机采样n_presamples个点
    presample_indices = np.random.choice(
        len(available_data_df), min(n_presamples, len(available_data_df)), replace=False)

    # 收集预采样点，累积组成整体集合
    presampled_rows = []
    reaction_values = []
    presampled_ok = []
    presampled_drop = []

    for idx in presample_indices:
        row = available_data_df.iloc[idx:idx+1]
        smiles = row['smiles'].values[0]
        presampled_rows.append(row)

        # 计算reaction
        reaction_result = aldehyde_reaction_judger.check_reaction(
            smiles, all_data, 'conv', not_feature_cols)
        if reaction_result == 0.001 or reaction_result is False or reaction_result < 10:
            presampled_drop.append(smiles)
            reaction_values.append(0.0)
        else:
            presampled_ok.append(smiles)
            reaction_values.append(min(reaction_result / 100.0, 1.0))

    # 将所有预采样点拼接为整体DataFrame，统一计算熵
    presampled_df = pd.concat(presampled_rows, ignore_index=True)
    try:
        entropy_mean = utils.calc_entropy(
            available_data_df, presampled_df, not_feature_cols,
            k=5, distance_metric='euclidean'
        )
        entropy_std = 0.0  # 整体集合只返回一个熵值，std归零
    except Exception as e:
        print(f"计算熵出错: {e}")
        entropy_mean = 0.0
        entropy_std = 0.0

    reaction_values = np.array(reaction_values)
    reaction_mean = np.mean(reaction_values)
    reaction_std = np.std(reaction_values)

    print(f"预采样统计:")
    print(f"  熵: mean={entropy_mean:.4f}, std={entropy_std:.4f}")
    print(f"  反应产率: mean={reaction_mean:.4f}, std={reaction_std:.4f}")

    # 计算缩放因子：将熵缩放到与反应产率相同的量级
    if entropy_std > 1e-10:
        entropy_scale = reaction_std / entropy_std if reaction_std > 1e-10 else 1.0
    else:
        entropy_scale = 1.0

    print(f"  熵缩放因子: {entropy_scale:.6f}")
    print(f"  预采样 OK: {len(presampled_ok)}, Drop: {len(presampled_drop)}")

    return entropy_scale, entropy_mean, presampled_df, presampled_ok, presampled_drop


def objective(trial, all_data, available_data_df, feature_cols, not_feature_cols,
              sampled_smiles_set, entropy_scale, entropy_mean, sampled_reference_df):
    """
    TPE采样的目标函数。

    参数:
        trial: Optuna trial对象
        all_data: 完整数据集
        available_data_df: 可用数据集
        feature_cols: 特征列名列表
        not_feature_cols: 非特征列名列表
        sampled_smiles_set: 已采样的smiles集合
        entropy_scale: 熵的缩放因子
        entropy_mean: 熵的均值
        sampled_reference_df: 历史已采样点组成的参照DataFrame

    返回:
        objective_value: 目标函数值（越大越好）
    """
    # 随机选择样本
    sample_idx = trial.suggest_int('sample_idx', 0, len(available_data_df) - 1)

    # 获取采样的数据点
    sampled_point = available_data_df.iloc[sample_idx:sample_idx+1]
    smiles = sampled_point['smiles'].values[0]

    # 确保不重复采样
    if smiles in sampled_smiles_set:
        raise optuna.TrialPruned()

    # 计算reaction反馈（归一化到0-1）
    reaction_result = aldehyde_reaction_judger.check_reaction(
        smiles, all_data, 'conv', not_feature_cols)

    if reaction_result == 0.001 or reaction_result is False:
        reaction_score = 0.0
    else:
        # 假设转化率在0-100之间，归一化
        reaction_score = min(reaction_result / 100.0, 1.0)

    # 计算entropy反馈：以历史已采样点为参照，计算包含新点的整体集合熵
    try:
        combined_df = pd.concat([sampled_reference_df, sampled_point], ignore_index=True)
        entropy_value = utils.calc_entropy(
            combined_df, combined_df, not_feature_cols,
            k=5, distance_metric='euclidean'
        )
        scaled_entropy = (entropy_value - entropy_mean) * entropy_scale + 0.5
        entropy_score = np.clip(scaled_entropy, 0, 1)
    except Exception as e:
        print(f"计算熵出错: {e}")
        entropy_score = 0.5

    # 组合反馈：0.5 * reaction + 0.5 * entropy
    objective_value = 0.5 * reaction_score + 0.5 * entropy_score

    return objective_value


def run_tpe_sampling(out_dir, n_trials_per_iter=10, n_iterations=5, random_seed=42):
    """
    使用TPE采样方法进行分子采样（累积历史信息，接近贝叶斯优化）。

    参数:
        out_dir: 输出目录
        n_trials_per_iter: 每次迭代新增的TPE试验次数
        n_iterations: 迭代次数
        random_seed: 随机种子
    """
    np.random.seed(random_seed)
    os.makedirs(out_dir, exist_ok=True)

    all_data = all_data_base.copy()
    all_data.to_csv(os.path.join(out_dir, 'labeled_points_itr0.csv'), index=False)

    # 预采样阶段：计算缩放因子
    available_data_first = all_data[all_data['ScreenLabel'] == 'BASE'].copy()
    entropy_scale, entropy_mean, sampled_reference_df, presampled_ok, presampled_drop = compute_scaling_factors(
        all_data, available_data_first, feature_columns, not_feature_columns, n_presamples=10)

    # 预采样点标记到 all_data 并重新保存 labeled_points_itr0.csv
    for smiles in presampled_ok:
        all_data.loc[all_data['smiles'] == smiles, 'ScreenLabel'] = 'Sampled'
    for smiles in presampled_drop:
        all_data.loc[all_data['smiles'] == smiles, 'ScreenLabel'] = 'Excluded_Sampled'
    all_data.to_csv(os.path.join(out_dir, 'labeled_points_itr0.csv'), index=False)

    # 创建全局Optuna study，累积历史trials（接近贝叶斯优化）
    sampler = TPESampler(seed=random_seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    sampled_smiles_set = set(presampled_ok + presampled_drop)
    OK_classes = list(presampled_ok)           # 有直接产率的样本
    droped_classes = list(presampled_drop)     # 排除的样本（产率<10或无产率数据）

    # 将预采样点保存为第1次中间结果
    save_counter = 1
    sampling_counter = 10
    pd.DataFrame({'smiles': OK_classes}).to_csv(
        os.path.join(out_dir, f'final_sampling_itr_{save_counter}.csv'), index=False)
    pd.DataFrame({'smiles': droped_classes}).to_csv(
        os.path.join(out_dir, f'final_drop_itr_{save_counter}.csv'), index=False)
    print(f'[SAVE] Itr {save_counter}: saved after {sampling_counter} samplings (pre-sampled)')

    task_itr_id = 1

    # 初始运行一批trials建立先验
    available_data_0 = all_data[all_data['ScreenLabel'] == 'BASE'].copy()
    print(f"\n=== 初始TPE阶段: 运行 {n_trials_per_iter} trials 建立先验 ===")
    study.optimize(
        lambda t: objective(t, all_data, available_data_0, feature_columns,
                            not_feature_columns, sampled_smiles_set, entropy_scale, entropy_mean,
                            sampled_reference_df),
        n_trials=n_trials_per_iter, show_progress_bar=False
    )
    print(f"初始trials完成，总trials数: {len(study.trials)}")


    while task_itr_id <= n_iterations:
        print(f'\n[TPE] Itr {task_itr_id} start')

        # 读取当前数据
        data = pd.read_csv(os.path.join(out_dir, f'labeled_points_itr{task_itr_id-1}.csv'))

        # 获取可用数据（ScreenLabel为BASE）
        available_data_itr = data[data['ScreenLabel'] == 'BASE'].copy()

        if len(available_data_itr) == 0:
            print("没有更多可用数据")
            break

        print(f"可用数据数量: {len(available_data_itr)}")

        # 运行新一批TPE优化（基于累积的历史trials）
        try:
            study.optimize(
                lambda t: objective(t, all_data, available_data_itr, feature_columns,
                                    not_feature_columns, sampled_smiles_set, entropy_scale, entropy_mean,
                                    sampled_reference_df),
                n_trials=n_trials_per_iter, show_progress_bar=False
            )
        except Exception as e:
            print(f"优化过程出错: {e}")
            break

        print(f"累积trials数: {len(study.trials)}")

        # 获取最佳采样点（选择最优的10个trials，确保不重复）
        trials_sorted = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None],
            key=lambda t: t.value,
            reverse=True
        )

        # 收集不重复的索引，最多10个
        selected_indices = []
        for t in trials_sorted:
            idx = t.params['sample_idx']
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= 10:
                break

        sampled_points = available_data_itr.iloc[selected_indices].copy()
        sampled_points = sampled_points.reset_index(drop=True)
        print(f"采样了 {len(sampled_points)} 个点")

        # 评估采样的点
        drop_classes = []
        for i in range(len(sampled_points)):
            smiles = sampled_points['smiles'].iloc[i]
            result = aldehyde_reaction_judger.check_reaction(
                smiles, all_data, 'conv', not_feature_columns)
            if result == 0.001 or result is False or result < 10:
                drop_classes.append(i)
                droped_classes.append(smiles)
                droped_classes = list(set(droped_classes))
            else:
                len_ok = len(OK_classes)
                OK_classes.append(smiles)
                OK_classes = list(set(OK_classes))
                if len(OK_classes) > len_ok:
                    print(f'OK: {sampled_points["conv"].iloc[i]}')

            sampled_smiles_set.add(smiles)
            sampling_counter += 1

            # 每10次采样保存一次中间结果，持续5次（共50次采样）
            if sampling_counter % 10 == 0 and save_counter < 5:
                save_counter += 1
                all_sampled_df_i = pd.DataFrame({'smiles': OK_classes})
                all_sampled_df_i.to_csv(os.path.join(out_dir, f'final_sampling_itr_{save_counter}.csv'), index=False)
                drop_df_i = pd.DataFrame({'smiles': droped_classes})
                drop_df_i.to_csv(os.path.join(out_dir, f'final_drop_itr_{save_counter}.csv'), index=False)
                print(f'[SAVE] Itr {save_counter}: saved after {sampling_counter} samplings')

        # 将本次采样的点累积到参照集合，用于后续熵计算
        sampled_reference_df = pd.concat([sampled_reference_df, sampled_points], ignore_index=True)

        print(f'[TPE] Itr {task_itr_id} finished: drop_classes: {drop_classes}')

        # 更新标签
        for i in range(len(sampled_points)):
            smiles = sampled_points.loc[i, 'smiles']
            if i in drop_classes:
                data.loc[data['smiles'] == smiles, 'ScreenLabel'] = 'Excluded_Sampled'
            else:
                data.loc[data['smiles'] == smiles, 'ScreenLabel'] = 'Sampled'

        data.to_csv(os.path.join(out_dir, f'labeled_points_itr{task_itr_id}.csv'), index=False)
        sampled_points.to_csv(os.path.join(out_dir, f'sampled_points_itr{task_itr_id}.csv'), index=False)

        task_itr_id += 1

    print('\n[TPE] Sampling finished.')
    all_sampled_df = pd.DataFrame({'smiles': OK_classes})
    all_sampled_df.to_csv(os.path.join(out_dir, 'final_sampling.csv'), index=False)
    drop_df = pd.DataFrame({'smiles': droped_classes})
    drop_df.to_csv(os.path.join(out_dir, 'final_drop.csv'), index=False)
    print(f'[TPE] OK samples: {OK_classes}')
    print(f'[TPE] Dropped samples: {len(droped_classes)}')


if __name__ == "__main__":
    # 使用5个不同的随机种子运行
    random_seeds = [0, 1, 2, 3, 4]
    for seed in random_seeds:
        out_dir = f'./tpe_sampling/seed_{seed}'
        print(f'\n========== Running with seed={seed} ==========')
        run_tpe_sampling(out_dir=out_dir, n_trials_per_iter=10, n_iterations=5, random_seed=seed)
        print(f'========== Seed {seed} finished ==========\n')
    print('All done.')
