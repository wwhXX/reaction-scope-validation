import aldehyde_reaction_judger
import sampling_util as cvt
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



data = pd.read_csv('1700_final_norepeat.csv')


X = data.drop(columns=['smiles', 'conv'])
y = data['conv']


y = y.apply(lambda x: 0 if x < 70 else 1)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


xgb_model = XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1     
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train, y_train, 
                           cv=kf, scoring='r2', n_jobs=-1)
print("Cross-Validation R2 Scores:", cv_scores)
print("Mean CV R2 Score:", np.mean(cv_scores))
print("Standard Deviation of CV R2 Scores:", np.std(cv_scores))


def evaluate_performance(model, X, y, set_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='macro')


    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\nPerformance on {set_name} set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return acc, recall, f1


def objective(trial, X_train, y_train, X_valid, y_valid):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 6, 50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': 23
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    acc, recall, f1 = evaluate_performance(model, X_valid, y_valid, "Validation")
    return f1


study = optuna.create_study(direction='maximize')
study.optimize(partial(objective, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid), n_trials=200)

print("\n=== Optuna Optimization Results ===")
print("Best trial:")
trial = study.best_trial
print(f"  R2: {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = trial.params
best_params.update({'random_state': 42, 'n_jobs': -1})
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)


print("\n=== Final Model Evaluation ===")
_ = evaluate_performance(best_model, X_train, y_train, "Training")
_ = evaluate_performance(best_model, X_valid, y_valid, "Validation")
_ = evaluate_performance(best_model, X_test, y_test, "Test")


print("\n=== Feature Importance ===")
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importances.head(10))












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
    k_num = min(max_k_num, int(last_labeled_len/100.0))
    print(f'k_num: {k_num}')
    sampled_points, labeled_points = cvt.get_sampling_weighted(data=data, task_itr_id=task_itr_id, not_feature_cols=['smiles', 'conv', 'ScreenLabel'], k=k_num)
    sampled_points = sampled_points.reset_index(drop=True)
    drop_classes = []
    print(sampled_points)
    for i in range(len(sampled_points)):
        if aldehyde_reaction_judger.check_reaction_by_model(sampled_points['smiles'][i], all_data, ['smiles', 'conv', 'ScreenLabel'], model=best_model) == False:
            drop_classes.append(i)
            droped_classes.append(sampled_points['smiles'][i])
            droped_classes = list(set(droped_classes))
            continue
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


