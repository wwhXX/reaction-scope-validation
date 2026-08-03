# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats

# 方法名到文件名前缀的映射
METHOD_FILES = {
    'CVT':           'CVT_sampling_results.csv',
    'LHS':           'LHS_sampling_results.csv',
    'Random':        'Random_sampling_results.csv',
    'Sobol':         'Sobol_sampling_results.csv',
    'KennardStone':  'KennardStone_sampling_results.csv',
    'WardClustering': 'WardClustering_sampling_results.csv',
}

# 简称映射
SHORT_NAMES = {
    'CVT':           'CVT',
    'LHS':           'LHS',
    'Random':        'Random',
    'Sobol':         'Sobol',
    'KennardStone':  'K_S',
    'WardClustering': 'WARD',
}


def load_sampling_data(data_dir, sample_sizes=(20, 40)):
    """
    读取各采样方法的 *_sampling_results.csv，提取指定采样数的conv值。

    Returns
    -------
    dict: { method_short: { 20: np.array, 40: np.array }, ... }
    """
    data = {}
    for m, fname in METHOD_FILES.items():
        fpath = f"{data_dir}/{fname}"
        df = pd.read_csv(fpath)
        short = SHORT_NAMES[m]
        data[short] = {}
        for s in sample_sizes:
            col = df[str(s)]
            data[short][s] = col.dropna().values
    return data


def independent_ttest(v1, v2):
    """独立样本t检验（双尾）"""
    t_stat, p_val = stats.ttest_ind(v1, v2)
    return t_stat, p_val


def welch_ttest(v1, v2):
    """Welch t检验（不假设方差齐性，更稳健）"""
    t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
    return t_stat, p_val


def paired_ttest(v1, v2):
    """配对样本t检验（双尾），要求 v1 和 v2 等长"""
    t_stat, p_val = stats.ttest_rel(v1, v2)
    return t_stat, p_val


def cohen_dz(v1, v2):
    """
    配对Cohen's d_z = mean(diff) / std(diff, ddof=1)
    衡量配对设计中的效应量。
    """
    diff = v1 - v2
    return diff.mean() / diff.std(ddof=1)


def paired_ci(v1, v2, alpha=0.05):
    """
    配对差值的 (1-alpha) 置信区间。

    Returns
    -------
    (lower, upper) : tuple of float
    """
    diff = v1 - v2
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    mean_d = diff.mean()
    return mean_d - t_crit * se, mean_d + t_crit * se


def sig_mark(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def calc_t_test(data_dir, exclude_sobol=True, sample_size=20):
    """
    对多种采样方法的转化率进行独立样本t检验。

    Parameters
    ----------
    data_dir : str
        数据文件所在目录
    exclude_sobol : bool
        是否排除Sobol方法
    sample_size : int
        参与比较的采样数（20 或 40）

    Returns
    -------
    pd.DataFrame
        t检验结果表
    """
    all_data = load_sampling_data(data_dir, sample_sizes=(20, 40))

    methods = [k for k in all_data if not (exclude_sobol and k == 'Sobol')]

    # 打印描述统计
    print("=" * 65)
    print(f"各方法 {sample_size} 采样 转化率 描述统计")
    print("=" * 65)
    print(f"{'方法':14s} {'n':>6s} {'mean':>8s} {'std':>8s}")
    print("-" * 65)
    stats_table = []
    for m in methods:
        vals = all_data[m][sample_size]
        mu, sd = vals.mean(), vals.std(ddof=1)
        print(f"{m:14s} {len(vals):6d} {mu:8.4f} {sd:8.4f}")
        stats_table.append({'Method': m, 'n': len(vals), 'mean': mu, 'std': sd})
    print()

    # 两两t检验
    method_list = methods
    results = []

    sep = "=" * 85
    dash = "-" * 85
    header = f"{'方法1':14s} {'方法2':14s} {'t统计量':>10s} {'p值':>12s} {'显著性':>6s} {'d_z':>8s} {'CI_low':>10s} {'CI_high':>10s}"

    print(sep)
    print(f"配对 t 检验结果（{sample_size} 采样，双尾，含 Cohen's d_z 与 95% CI）")
    print(sep)
    print(header)
    print(dash)

    for i in range(len(method_list)):
        for j in range(i + 1, len(method_list)):
            m1, m2 = method_list[i], method_list[j]
            v1, v2 = all_data[m1][sample_size], all_data[m2][sample_size]
            n1, n2 = len(v1), len(v2)

            if n1 == n2:
                t_stat, p_val = paired_ttest(v1, v2)
                dz = cohen_dz(v1, v2)
                ci_lo, ci_hi = paired_ci(v1, v2)
                test_type = 'paired'
            else:
                # 样本量不等时退回 Welch 独立检验，d_z 标记为 NaN
                t_stat, p_val = welch_ttest(v1, v2)
                dz = float('nan')
                ci_lo = ci_hi = float('nan')
                test_type = 'welch'

            sig = sig_mark(p_val)
            print(f"{m1:14s} {m2:14s} {t_stat:10.4f} {p_val:12.6f} {sig:>6s} "
                  f"{dz:8.4f} {ci_lo:10.4f} {ci_hi:10.4f}")
            results.append({
                'Method 1': m1,
                'Method 2': m2,
                'test type': test_type,
                't statistic': t_stat,
                'p value': p_val,
                'Significance': sig,
                "Cohen's d_z": dz,
                'CI 95% low': ci_lo,
                'CI 95% high': ci_hi,
            })

    print()
    print("显著性标记: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")
    print("d_z = mean(v1-v2) / std(v1-v2)；CI 为配对差值均值的 95% 置信区间")

    return pd.DataFrame(results), pd.DataFrame(stats_table)


if __name__ == "__main__":
    import os
    data_dir = os.path.dirname(os.path.abspath(__file__))

    for size in [20, 40]:
        for exclude_sobol in [True, False]:
            label = "（不含 Sobol）" if exclude_sobol else "（含 Sobol）"
            print(f"\n{'='*60}")
            print(f"【{size} 采样数】{label}")
            print("=" * 60)
            results_df, stats_df = calc_t_test(
                data_dir=data_dir,
                exclude_sobol=exclude_sobol,
                sample_size=size
            )
