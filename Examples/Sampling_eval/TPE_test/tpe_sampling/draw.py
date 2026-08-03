import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "/home/ljw/1700/1700/revision1/Bayesian_samesize/tpe_sampling"
scopemap_dir = os.path.join(base_dir, "scopemap")

# ── 读取 seed_* 数据 ──────────────────────────────────────────────────────────
# 存储结构: {itr: {seed: {'entropy': val, 'msd': val}}}
data = {}

for seed in range(5):
    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    for fname in os.listdir(seed_dir):
        if fname.startswith("evaluate_itr_") and fname.endswith(".log"):
            # 提取 itr 编号
            m = re.search(r"evaluate_itr_(\d+)\.log", fname)
            if not m:
                continue
            itr = int(m.group(1))

            fpath = os.path.join(seed_dir, fname)
            entropy_val, msd_val = None, None
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    em = re.search(r"实验采样熵\s*\(Entropy\)\s*:\s*([\d.]+)", line)
                    if em:
                        entropy_val = float(em.group(1))
                    mm = re.search(r"实验采样MSD\s*:\s*([\d.]+)", line)
                    if mm:
                        msd_val = float(mm.group(1))

            if itr not in data:
                data[itr] = {}
            data[itr][seed] = {"entropy": entropy_val, "msd": msd_val}

# ── 读取 scopemap 数据 ────────────────────────────────────────────────────────
# 存储结构: {itr: {'entropy': val, 'msd': val}}
scopemap_data = {}
for fname in sorted(os.listdir(scopemap_dir)):
    if fname.startswith("evaluate_itr_") and fname.endswith(".log"):
        m = re.search(r"evaluate_itr_(\d+)\.log", fname)
        if not m:
            continue
        itr = int(m.group(1))
        fpath = os.path.join(scopemap_dir, fname)
        entropy_val, msd_val = None, None
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                em = re.search(r"实验采样熵\s*\(Entropy\)\s*:\s*([\d.]+)", line)
                if em:
                    entropy_val = float(em.group(1))
                mm = re.search(r"实验采样MSD\s*:\s*([\d.]+)", line)
                if mm:
                    msd_val = float(mm.group(1))
        scopemap_data[itr] = {"entropy": entropy_val, "msd": msd_val}

scopemap_agg = pd.DataFrame([
    {"itr": itr, "entropy": vals["entropy"], "msd": vals["msd"]}
    for itr, vals in sorted(scopemap_data.items())
])

# 构建 DataFrame
records = []
for itr in sorted(data.keys()):
    for seed, vals in data[itr].items():
        records.append({
            "itr": itr,
            "seed": seed,
            "entropy": vals["entropy"],
            "msd": vals["msd"],
        })

df = pd.DataFrame(records)

# 按 itr 聚合均值和标准差
agg = df.groupby("itr").agg(
    entropy_mean=("entropy", "mean"),
    entropy_std=("entropy", "std"),
    msd_mean=("msd", "mean"),
    msd_std=("msd", "std"),
).reset_index()

print(agg)

# 绘图
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Entropy
axes[0].errorbar(
    agg["itr"], agg["entropy_mean"],
    yerr=agg["entropy_std"],
    marker="o", capsize=4, color="steelblue", linewidth=2, markersize=6,
    label="TPE",
)
axes[0].plot(
    scopemap_agg["itr"], scopemap_agg["entropy"],
    marker="s", color="seagreen", linewidth=2, markersize=6,
    label="ScopeMap",
)
axes[0].set_xlabel("Iteration", fontsize=12)
axes[0].set_ylabel("Entropy", fontsize=12)
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xticklabels([10, 20, 30, 40, 50])
axes[0].grid(axis='x')
axes[0].set_title("Entropy vs Iteration", fontsize=13)
axes[0].legend(fontsize=10)

# MSD
axes[1].errorbar(
    agg["itr"], -agg["msd_mean"],
    yerr=agg["msd_std"],
    marker="o", capsize=4, color="steelblue", linewidth=2, markersize=6,
    label="TPE",
)
axes[1].plot(
    scopemap_agg["itr"], -scopemap_agg["msd"],
    marker="s", color="seagreen", linewidth=2, markersize=6,
    label="ScopeMap",
)
axes[1].set_xlabel("Iteration", fontsize=12)
axes[1].set_ylabel("MSD", fontsize=12)
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].set_xticklabels([10, 20, 30, 40, 50])
axes[1].grid(axis='x')
axes[1].set_title("MSD vs Iteration", fontsize=13)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "result.png"), dpi=150)
plt.savefig(os.path.join(base_dir, "result.svg"), format="svg")
plt.show()
