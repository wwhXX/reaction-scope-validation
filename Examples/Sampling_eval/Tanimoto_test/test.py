import aldehyde_reaction_judger
import sampling_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from math import sqrt
from functools import partial
import seaborn as sns

old_random_seed = 42
np.random.seed(old_random_seed)
max_k_num = 10

# 读取数据
smi_data = pd.read_csv('zinc_aryl_aldehyde.csv')
fp_all_data = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')
all_data = pd.concat([smi_data, fp_all_data], axis=1)

exp_raw_data = pd.read_csv('1700_final_norepeat.csv')
exp_data = exp_raw_data
print(exp_data.shape)

all_data = pd.concat([all_data, exp_data], axis=0, ignore_index=True)
all_data['ScreenLabel'] = 'BASE'
print(all_data)

all_data.to_csv('./itr/labeled_points_itr0.csv', index=False)



stop = False
task_itr_id = 1
drop_classes = []
OK_classes = []
droped_classes = []
last_labeled_len = len(all_data)
while(not stop):
    print(f'Itr {task_itr_id} start')
    data = pd.read_csv(f'./itr/labeled_points_itr{task_itr_id-1}.csv')
    #动态调整采样数
    k_num = min(max_k_num, int(last_labeled_len/100.0))
    print(f'k_num: {k_num}')
    sampled_points, labeled_points = sampling_util.get_sampling_weighted(data=data, task_itr_id=task_itr_id, not_feature_cols=['smiles', 'conv', 'ScreenLabel'], k=k_num, distance_metric='tanimoto', n_jobs=32)
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
            
    print(f'Itr {task_itr_id} finished: drop_classes: {drop_classes}')

    for i in range(len(sampled_points)):
        smiles = sampled_points.loc[i, 'smiles']
        if i in drop_classes:
            data.loc[data['smiles'] == smiles, 'ScreenLabel'] = 'Excluded_Sampled'
        else:
            data.loc[data['smiles'] == smiles, 'ScreenLabel'] = 'Sampled'
    
    data.to_csv(f'./itr/labeled_points_itr{task_itr_id}.csv', index=False)
    sampled_points.to_csv(f'./itr/sampled_points_itr{task_itr_id}.csv', index=False)
    last_labeled_len = len(data)

    if len(OK_classes) >= 10:
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


