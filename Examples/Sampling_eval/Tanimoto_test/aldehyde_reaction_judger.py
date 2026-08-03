from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold



def check_reaction(smi, all_data_df, conv_col, not_feature_cols):
    point_data = all_data_df[all_data_df['smiles'] == smi]
    point_data = point_data.drop(not_feature_cols, axis=1)
    all_data_fp = all_data_df.drop(not_feature_cols, axis=1)

    # 转换为布尔类型以使用 jaccard 距离
    all_data_fp = all_data_fp.astype(bool)
    point_data = point_data.astype(bool)

    n_neighbors = 10  # 要找的最近邻数量
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='jaccard')
    nn.fit(all_data_fp)

    distances, indices = nn.kneighbors(point_data)
    has_reaction = False
    indices = indices[0]
    for indice in indices:
        conv = all_data_df[conv_col][indice]
        #判断conv NAN
        if pd.isna(conv):
            continue
        else:
            has_reaction = True
            # 处理 conv 可能是浮点数或字符串的情况
            if isinstance(conv, str):
                reaction_conv = int(conv.split('%')[0])
            else:
                reaction_conv = int(conv)
            break
    if has_reaction:
        return reaction_conv
    else:
        # return False
        return 0.001
    
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
    


# 示例用法
if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('1700_final_norepeat.csv')
    # 检查反应
    print(check_reaction('O=Cc1c(F)cc(Cl)cc1F', data, 'conv', ['reactant_aldehyde', 'conv']))
