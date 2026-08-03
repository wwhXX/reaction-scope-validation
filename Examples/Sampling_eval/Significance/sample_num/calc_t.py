# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from scipy import stats



def cohen_dz_ci(v1, v2, alpha=0.05):
    diff = np.asarray(v1) - np.asarray(v2)
    n = len(diff)
    mean_d = diff.mean()
    std_d  = diff.std(ddof=1)
    if std_d == 0:
        return float('nan'), float('nan'), float('nan')
    dz = mean_d / std_d
    se = np.sqrt(1.0 / n + dz ** 2 / (2.0 * (n - 1)))
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lo = dz - t_crit * se
    ci_hi = dz + t_crit * se
    return dz, ci_lo, ci_hi



def parse_cvt_log(log_file='cvt.log'):
    records = []
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 每个块以 "=== Split N, k=K Optuna Results ===" 开头
    block_pat = re.compile(
        r'=== Split (\d+), k=(\d+) Optuna Results ===.*?'
        r'Performance on Test set:\s*\n'
        r'Accuracy:\s*([\d.]+)\s*\n'
        r'Recall:\s*([\d.]+)\s*\n'
        r'F1 Score:\s*([\d.]+)',
        re.DOTALL
    )
    for m in block_pat.finditer(content):
        records.append({
            'split': int(m.group(1)),
            'k':     int(m.group(2)),
            'acc':   float(m.group(3)),
            'recall': float(m.group(4)),
            'f1':    float(m.group(5)),
        })
    return pd.DataFrame(records)


def _print_ttest_table(res_df, label):
    print(f"\n{'Group 1':12s} {'Group 2':12s} {'t统计量':>10s} {'p值':>12s} {'显著性':>6s} {'d_z':>8s} {'95%CI':>18s}")
    print("-" * 80)
    for _, r in res_df.iterrows():
        t_str  = f"{r['t statistic']:10.4f}" if not np.isnan(r['t statistic']) else "       NaN"
        p_str  = f"{r['p value']:12.6f}"     if not np.isnan(r['p value'])      else "         NaN"
        dz_val = r["Cohen's d_z"]
        dz_str = f"{dz_val:8.4f}" if not np.isnan(dz_val) else "     NaN"
        if not np.isnan(r['95% CI lower']):
            ci_str = f"[{r['95% CI lower']:6.4f}, {r['95% CI upper']:6.4f}]"
        else:
            ci_str = "         NaN"
        print(f"{r['Group 1']:12s} {r['Group 2']:12s} {t_str} {p_str} {r['Significance']:>6s} {dz_str} {ci_str:>18s}")
    print("\n显著性标记: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著, NaN = std=0无法检验\n")


def calc_t_test_score_real(log_file='cvt.log'):
    df = parse_cvt_log(log_file)
    if df.empty:
        print(f"警告: 未从 {log_file} 解析到数据")
        return

    k_values = sorted(df['k'].unique())
    metrics   = ['acc', 'recall', 'f1']
    metric_names = {'acc': 'Acc', 'recall': 'Recall', 'f1': 'F1'}

    # 构建 {k: {metric: array(splits)}} 的数据字典
    # 按 split 排序保证配对一致
    score_data = {}
    for k in k_values:
        sub = df[df['k'] == k].sort_values('split')
        score_data[k] = {
            'acc':    sub['acc'].values,
            'recall': sub['recall'].values,
            'f1':     sub['f1'].values,
        }

    n_splits = df['split'].nunique()

    for metric in metrics:
        print("=" * 80)
        print(f"  {metric_names[metric]} 配对 t 检验（真实数据，n={n_splits} splits）")
        print("=" * 80)

        # 各组统计量
        print(f"\n{'k_samples':>12s}  {'Mean':>8s}  {'Std':>8s}  {'Values'}")
        print("-" * 60)
        for k in k_values:
            vals = score_data[k][metric]
            vals_str = ', '.join(f"{v:.4f}" for v in vals)
            print(f"{k:>12d}  {vals.mean():>8.4f}  {vals.std(ddof=1):>8.4f}  [{vals_str}]")

        # 配对 t 检验（所有 k 两两组合）
        results = []
        for i in range(len(k_values)):
            for j in range(i + 1, len(k_values)):
                k1, k2 = k_values[i], k_values[j]
                v1 = score_data[k1][metric]
                v2 = score_data[k2][metric]
                t_stat, p_val = stats.ttest_rel(v1, v2)
                dz, ci_lo, ci_hi = cohen_dz_ci(v1, v2)
                if np.isnan(p_val):
                    sig = 'NaN'
                elif p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                results.append({
                    'Group 1': f'k={k1}',
                    'Group 2': f'k={k2}',
                    't statistic': t_stat,
                    'p value': p_val,
                    'Significance': sig,
                    "Cohen's d_z": dz,
                    '95% CI lower': ci_lo,
                    '95% CI upper': ci_hi,
                })

        _print_ttest_table(pd.DataFrame(results), metric_names[metric])

    return score_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("\n【 Score (Acc/Recall/F1) 配对 t 检验 — 真实数据 (cvt.log) 】\n")
        calc_t_test_score_real()
