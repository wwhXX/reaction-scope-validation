import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

with open('Classification.log', 'r') as f:
    content = f.read()

pattern = r'Accuracy: ([\d.]+)\s+Recall: ([\d.]+)\s+F1 Score: ([\d.]+)'
matches = re.findall(pattern, content)

accuracys = [float(m[0]) for m in matches]
recalls = [float(m[1]) for m in matches]
f1s = [float(m[2]) for m in matches]

print(f"找到 {len(accuracys)} 组数据")

sample_sizes = [20, 40, 60, 80, 100]
n_splits = 5

acc_means, acc_stds = [], []
rec_means, rec_stds = [], []
f1_means, f1_stds = [], []

for i, size in enumerate(sample_sizes):
    acc_group = accuracys[i::len(sample_sizes)]
    rec_group = recalls[i::len(sample_sizes)]
    f1_group  = f1s[i::len(sample_sizes)]

    acc_means.append(np.mean(acc_group))
    acc_stds.append(np.std(acc_group))
    rec_means.append(np.mean(rec_group))
    rec_stds.append(np.std(rec_group))
    f1_means.append(np.mean(f1_group))
    f1_stds.append(np.std(f1_group))

    print(f"{size} samples: Acc={np.mean(acc_group):.4f}±{np.std(acc_group):.4f}, Recall={np.mean(rec_group):.4f}±{np.std(rec_group):.4f}, F1={np.mean(f1_group):.4f}±{np.std(f1_group):.4f}")

# Set up bar positions
samples = ['20', '40', '60', '80', '100']
x = np.arange(len(samples))
width = 0.2

# Create figure and axis
plt.figure(figsize=(12, 8))

# Create grouped bars with error bars
bars1 = plt.bar(x - width, acc_means, width, label='Accuracy', color='#D87792', edgecolor='#CA486B', linewidth=2, yerr=acc_stds, capsize=4)
bars2 = plt.bar(x, rec_means, width, label='Recall', color='#2C75E3', edgecolor='#172C51', linewidth=2, yerr=rec_stds, capsize=4)
bars3 = plt.bar(x + width, f1_means, width, label='F1', color='#29ABBA', edgecolor='#228E9B', linewidth=2, yerr=f1_stds, capsize=4)

# Customize the plot
plt.xlabel('Number of samples', fontsize=30)
plt.ylabel('Score', fontsize=30)
plt.ylim(0.60, 1.05)
plt.xticks(x, samples, fontsize=27)
plt.yticks(fontsize=27)
plt.legend(loc='lower right', fontsize=27)

plt.tight_layout()
plt.savefig('get_score_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('get_score_plot.svg', bbox_inches='tight')
plt.show()
