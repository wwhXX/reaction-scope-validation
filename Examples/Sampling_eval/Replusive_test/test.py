import aldehyde_reaction_judger
import cvt
import numpy as np
import pandas as pd
import os

old_random_seed = 42
max_k_num = 10

# 读取数据（所有运行共享同一份原始数据）
smi_data = pd.read_csv('zinc_aryl_aldehyde.csv')
fp_all_data = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')
all_data_base = pd.concat([smi_data, fp_all_data], axis=1)

exp_raw_data = pd.read_csv('1700_final_norepeat.csv')
print(exp_raw_data.shape)

all_data_base = pd.concat([all_data_base, exp_raw_data], axis=0, ignore_index=True)
all_data_base['ScreenLabel'] = 'BASE'
print(all_data_base)


def run_sampling(repulsion_func, out_dir):
    """
    使用指定排斥函数跑完整采样循环，结果保存到 out_dir。
    """
    np.random.seed(old_random_seed)
    os.makedirs(out_dir, exist_ok=True)

    all_data = all_data_base.copy()
    all_data.to_csv(os.path.join(out_dir, 'labeled_points_itr0.csv'), index=False)

    task_itr_id = 1
    drop_classes = []
    OK_classes = []
    droped_classes = []
    last_labeled_len = len(all_data)

    for task_itr_id in range(1, 6):
        print(f'[{repulsion_func}] Itr {task_itr_id} start')
        data = pd.read_csv(os.path.join(out_dir, f'labeled_points_itr{task_itr_id-1}.csv'))

        k_num = 10
        print(f'k_num: {k_num}')

        sampled_data = data[data['ScreenLabel'] != 'BASE']
        sampled_points, _ = cvt.weighted_itr_cvt_sampling_df_norepeat(
            data=data,
            k=k_num,
            not_feature_columns=['smiles', 'conv', 'ScreenLabel'],
            sampled_data=sampled_data,
            repulsion_func=repulsion_func,
        )
        sampled_points = sampled_points.reset_index(drop=True)

        drop_classes = []
        print(sampled_points)
        for i in range(len(sampled_points)):
            result = aldehyde_reaction_judger.check_reaction(
                sampled_points['smiles'][i], all_data, 'conv', ['smiles', 'conv', 'ScreenLabel'])
            if result is False:
                continue
            if result < 10:
                drop_classes.append(i)
                droped_classes.append(sampled_points['smiles'][i])
                droped_classes = list(set(droped_classes))
            else:
                len_ok = len(OK_classes)
                OK_classes.append(sampled_points['smiles'][i])
                OK_classes = list(set(OK_classes))
                if len(OK_classes) > len_ok:
                    print(f'OK: {sampled_points["conv"][i]}')

        print(f'[{repulsion_func}] Itr {task_itr_id} finished: drop_classes: {drop_classes}')

        for i in range(len(sampled_points)):
            smiles = sampled_points.loc[i, 'smiles']
            label = 'Excluded_Sampled' if i in drop_classes else 'Sampled'
            data.loc[data['smiles'] == smiles, 'ScreenLabel'] = label

        data.to_csv(os.path.join(out_dir, f'labeled_points_itr{task_itr_id}.csv'), index=False)
        sampled_points.to_csv(os.path.join(out_dir, f'sampled_points_itr{task_itr_id}.csv'), index=False)
        last_labeled_len = len(data)

        # 保存历史连带本次采样结果
        ok_df = pd.DataFrame({'smiles': OK_classes})
        ok_df.to_csv(os.path.join(out_dir, f'final_sampling_itr_{task_itr_id}.csv'), index=False)
        drop_df = pd.DataFrame({'smiles': droped_classes})
        drop_df.to_csv(os.path.join(out_dir, f'final_drop_itr_{task_itr_id}.csv'), index=False)
        print(f'Saved final_sampling_itr_{task_itr_id}.csv and final_drop_itr_{task_itr_id}.csv')

    print(f'[{repulsion_func}] Sampling finished.')
    ok_df = pd.DataFrame({'smiles': OK_classes})
    ok_df.to_csv(os.path.join(out_dir, 'final_sampling.csv'), index=False)
    drop_df = pd.DataFrame({'smiles': droped_classes})
    drop_df.to_csv(os.path.join(out_dir, 'final_drop.csv'), index=False)
    print(f'[{repulsion_func}] OK samples: {OK_classes}')


# 依次使用四种排斥函数运行
for func in ['inverse_square', 'inverse', 'gaussian', 'hinge']:
    print(f'\n{"="*60}')
    print(f'Running with repulsion_func = {func}')
    print(f'{"="*60}\n')
    run_sampling(repulsion_func=func, out_dir=f'./itr_{func}')

print('All done.')


