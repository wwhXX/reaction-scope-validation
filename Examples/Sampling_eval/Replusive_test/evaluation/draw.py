import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "/home/ljw/1700/1700/revision1/replusive_test_samesize/evaluation"

# 4个损失函数对应的文件夹
loss_names = ["gaussian", "hinge", "inverse", "inverse_square"]

# 颜色映射
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

# 读取每个文件夹的数据，存储结构: {loss_name: {itr: {'entropy': val, 'msd': val}}}
loss_data = {}

for loss in loss_names:
    folder = os.path.join(base_dir, loss)
    loss_data[loss] = {}
    for fname in sorted(os.listdir(folder)):
        if fname.startswith("evaluate_itr_") and fname.endswith(".log"):
            m = re.search(r"evaluate_itr_(\d+)\.log", fname)
            if not m:
                continue
            itr = int(m.group(1))
            fpath = os.path.join(folder, fname)
            entropy_val, msd_val = None, None
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    em = re.search(r"实验采样熵\s*\(Entropy\)\s*:\s*([\d.]+)", line)
                    if em:
                        entropy_val = float(em.group(1))
                    mm = re.search(r"实验采样MSD\s*:\s*([\d.]+)", line)
                    if mm:
                        msd_val = float(mm.group(1))
            loss_data[loss][itr] = {"entropy": entropy_val, "msd": msd_val}

# 构建 DataFrame
records = []
for loss, itrs in loss_data.items():
    for itr, vals in itrs.items():
        records.append({
            "loss": loss,
            "itr": itr,
            "entropy": vals["entropy"],
            "msd": vals["msd"],
        })

df = pd.DataFrame(records)

# 打印汇总
for loss in loss_names:
    sub = df[df["loss"] == loss].sort_values("itr")
    print(f"\n=== {loss} ===")
    print(sub[["itr", "entropy", "msd"]])

# 绘图
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, loss in enumerate(loss_names):
    sub = df[df["loss"] == loss].sort_values("itr")
    axes[0].plot(
        sub["itr"], sub["entropy"],
        marker="o", color=colors[i], linewidth=2, markersize=6,
        label=loss,
    )
    axes[1].plot(
        sub["itr"], -sub["msd"],
        marker="o", color=colors[i], linewidth=2, markersize=6,
        label=loss,
    )

# Entropy
axes[0].set_xlabel("Iteration", fontsize=12)
axes[0].set_ylabel("Entropy", fontsize=12)
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xticklabels([10, 20, 30, 40, 50])
axes[0].grid(axis='x')
axes[0].set_title("Entropy vs Iteration", fontsize=13)
axes[0].legend(fontsize=10)

# MSD
axes[1].set_xlabel("Iteration", fontsize=12)
axes[1].set_ylabel("MSD", fontsize=12)
axes[1].set_xticks([1, 2, 3, 4, 5])
axes[1].set_xticklabels([10, 20, 30, 40, 50])
axes[1].grid(axis='x')
axes[1].set_title("MSD vs Iteration", fontsize=13)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "result.png"), dpi=150)
plt.savefig(os.path.join(base_dir, "result.svg"))
plt.show()
