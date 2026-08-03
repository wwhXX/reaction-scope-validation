from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold

    
def check_reaction_by_model(smi, all_data_df, not_feature_cols, model:XGBClassifier):
    point_data = all_data_df[all_data_df['smiles'] == smi]
    point_data = point_data.drop(not_feature_cols, axis=1)
    all_data_fp = all_data_df.drop(not_feature_cols, axis=1)

    y = model.predict(point_data)
    print(y)
    if y[0] == 1:
        return True
    else:
        return False
    


if __name__ == "__main__":
    data = pd.read_csv('1700_final_norepeat.csv')

