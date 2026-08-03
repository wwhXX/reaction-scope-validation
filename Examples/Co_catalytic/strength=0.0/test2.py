import alcohol_reaction_judger
import cvt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

old_random_seed = 42
np.random.seed(old_random_seed)
max_k_num = 20

# 读取数据
smi_data = pd.read_csv('底物醇去重.csv')
fp_all_data = pd.read_csv('fp_spoc_morgan41024_Maccs_底物醇去重_alcohol.csv')
all_data = pd.concat([smi_data, fp_all_data], axis=1)
print(all_data)

all_data.to_csv('./itr/labeled_points_itr0.csv', index=False)

exp_raw_data = pd.read_csv('experimental_data_results_of_alcohol.csv')
exp_smi_data = exp_raw_data[['SMILES']]
exp_conv_data = exp_raw_data[['condition1_yield']]
exp_fp_data = pd.read_csv('fp_spoc_morgan41024_Maccs_experimental_data_results_of_alcohol_alcohol.csv')
exp_data = pd.concat([exp_smi_data, exp_conv_data, exp_fp_data], axis=1)
print(exp_data.shape)


stop = False
task_itr_id = 1
drop_classes = []
OK_classes = []
droped_classes = []
last_labeled_len = len(all_data)
while(not stop):
    print(f'Itr {task_itr_id} start')
    #动态调整采样数
    k_num = min(max_k_num, int(last_labeled_len/100.0))
    print(f'k_num: {k_num}')
    sampled_points, labeled_points = cvt.get_sampling(task_itr_id=task_itr_id, drop_classes=drop_classes, not_feature_cols=['smiles'], k=k_num)
    last_labeled_len = len(labeled_points)
    sampled_points = sampled_points.reset_index(drop=True)
    drop_classes = []
    print(sampled_points)
    exp_points = cvt.classify_by_centers(data=exp_data, centers=sampled_points, not_feature_columns=['SMILES', 'condition1_yield', 'smiles', 'labels'])
    exp_point_labels = exp_points['labels'].tolist()
    for i in range(len(sampled_points)):
        if alcohol_reaction_judger.check_nucleophilic_atoms(sampled_points['smiles'][i]) == 0 or alcohol_reaction_judger.check_tertiary_alcohols(sampled_points['smiles'][i]) == 0:
            drop_classes.append(i)
            droped_classes.append(sampled_points['smiles'][i])
            droped_classes = list(set(droped_classes))
            continue
        else:
            if i in exp_point_labels:
                matches = exp_points[exp_points['labels'] == i]
                is_OK = False
                for index, row in matches.iterrows():
                    if row['condition1_yield'] != 0:
                        len_ok = len(OK_classes)
                        OK_classes.append(row['SMILES'])
                        OK_classes = list(set(OK_classes))
                        if len(OK_classes) > len_ok:
                            print(f'OK: {row["condition1_yield"]}')
                            is_OK = True
                            break
                if not is_OK:
                    drop_classes.append(i)
                    droped_classes = list(set(droped_classes))
    if len(drop_classes) == k_num:
        print('All droped, restart.')
        random_seed = np.random.randint(1000)
        while(random_seed == old_random_seed):
            random_seed = np.random.randint(1000)
        old_random_seed = random_seed
        print(f'Set new random seed: {random_seed}')
        np.random.seed(random_seed)
        drop_classes = []
        task_itr_id += 1
        continue
    if len(drop_classes) == 0:
        print('No droped, restart.')
        random_seed = np.random.randint(1000)
        while(random_seed == old_random_seed):
            random_seed = np.random.randint(1000)
        old_random_seed = random_seed
        print(f'Set new random seed: {random_seed}')
        np.random.seed(random_seed)
        drop_classes = []
        task_itr_id += 1
        continue
            
    print(f'Itr {task_itr_id} finished: drop_classes: {drop_classes}')
    if len(OK_classes) >= 20:
        stop = True
        break
    task_itr_id += 1

print('Sampling finished.')
ok_df = pd.DataFrame({'smiles': OK_classes})
ok_df.to_csv('final_sampling.csv', index=False)
drop_df = pd.DataFrame({'smiles': droped_classes})
drop_df.to_csv('final_drop.csv', index=False)
print(f'OK samples: {OK_classes}')
print('Done')


