import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import umap
from scipy.interpolate import griddata

np.random.seed(42)

# Read labeled_points_itr0.csv to get the original data order
labeled_data_itr0 = pd.read_csv('labeled_points_itr0.csv')

# Use labeled_points_itr0 features for UMAP (excluding non-feature columns)
exclude_cols = ['smiles', 'labels', 'ScreenLabel'] 
feature_cols = [col for col in labeled_data_itr0.columns if col not in exclude_cols]
umap_data = labeled_data_itr0[feature_cols].values

print(f"Original labeled data shape: {labeled_data_itr0.shape}")
print(f"UMAP input data shape: {umap_data.shape}")

# Perform UMAP on the original data
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(umap_data)
print('UMAP done')

# Create DataFrame for plotting  
plot_data = pd.DataFrame(X_umap, columns=['x', 'y'])

print(f"UMAP x range: [{X_umap[:, 0].min():.2f}, {X_umap[:, 0].max():.2f}]")
print(f"UMAP y range: [{X_umap[:, 1].min():.2f}, {X_umap[:, 1].max():.2f}]")

# Create mapping for finding indices by smiles
smiles_to_idx = {}
for idx, smiles in enumerate(labeled_data_itr0['smiles']):
    smiles_to_idx[smiles] = idx

def find_sampling_indices(smiles_list, smiles_to_idx):
    """Find indices for given smiles in the labeled_data_itr0"""
    indices = []
    for smiles in smiles_list:
        if smiles in smiles_to_idx:
            indices.append(smiles_to_idx[smiles])
    return indices

def create_energy_contour_kde(X_umap, repulsion_energies, xlim, ylim, bandwidth=0.5):
    """
    使用KDE方法创建排斥能等高线
    
    参数:
        X_umap: UMAP降维后的坐标
        repulsion_energies: 对应的排斥能
        xlim: x轴范围 (xmin, xmax)
        ylim: y轴范围 (ymin, ymax)
        bandwidth: KDE带宽
    
    返回:
        xi, yi, zi: 用于等高线绘制的网格坐标和KDE估计的排斥能
    """
    from sklearn.neighbors import KernelDensity
    
    # 创建网格
    xi = np.linspace(xlim[0], xlim[1], 100)
    yi = np.linspace(ylim[0], ylim[1], 100)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi)
    grid_points = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()]).T
    
    # 使用加权KDE，权重为排斥能
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    
    # 创建加权样本：根据排斥能重复采样点
    weights = repulsion_energies / repulsion_energies.sum()
    n_samples = 1000
    weighted_indices = np.random.choice(len(X_umap), size=n_samples, p=weights, replace=True)
    weighted_points = X_umap[weighted_indices]
    
    kde.fit(weighted_points)
    
    # 计算每个网格点的密度
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)
    
    zi = density.reshape(xi_mesh.shape)
    
    return xi_mesh, yi_mesh, zi

all_sampling_points = []

for i in range(1, 7):
    # 读取能量评估结果
    try:
        energy_data = pd.read_csv(f'energy_eval_itr{i}.csv')
        print(f'Loaded energy data for iteration {i}, shape: {energy_data.shape}')
        
        # 获取排斥能并取对数 - 这里的数据顺序应该与labeled_data_itr0一致
        repulsion_energies = np.log(energy_data['repulsion_energy'].values)
        print(f'Energy range: [{repulsion_energies.min():.6f}, {repulsion_energies.max():.6f}]')
            
    except FileNotFoundError:
        print(f'Warning: energy_eval_itr{i}.csv not found, skipping energy contour for this iteration')
        repulsion_energies = None

    # 读取采样点
    final_sampling = pd.read_csv(f'sampled_points_itr{i}.csv')
    final_sampling = pd.DataFrame(final_sampling.loc[:,'smiles'])
    final_sampling.columns = ['smiles']
    print(final_sampling)

    # 找到采样点在UMAP空间中的索引
    sampling_indices = find_sampling_indices(final_sampling['smiles'].tolist(), smiles_to_idx)
    
    all_sampling_points.append({
        'indices': sampling_indices,
        'iteration': i
    })

    plt.figure(figsize=(14, 10))

    # 绘制排斥能等高线（如果有数据）
    if repulsion_energies is not None:
        print(f'Plotting repulsion energy contour for iteration {i}')
        
        # 设置图形范围
        xlim = (-5, 25)
        ylim = (-15, 25)
        
        # 创建等高线数据 (使用KDE方法)
        xi, yi, zi = create_energy_contour_kde(X_umap, repulsion_energies, xlim, ylim, bandwidth=1.0)
        
        # 绘制等高线填充
        contour_filled = plt.contourf(xi, yi, zi, levels=20, cmap='Greys', alpha=0.6)
        
        # 添加颜色条
        cbar = plt.colorbar(contour_filled, shrink=0.8, aspect=20)
        cbar.ax.tick_params(labelsize=36)
    

    previous_added = False
    current_added = False
    
    for j, prev_sampling in enumerate(all_sampling_points):
        prev_indices = prev_sampling['indices']
        prev_itr = prev_sampling['iteration']
        
        if prev_indices:
            prev_x = [X_umap[idx][0] for idx in prev_indices]
            prev_y = [X_umap[idx][1] for idx in prev_indices]
            
            is_current = (prev_itr == i)
            color = '#4A90E2' if is_current else 'gray'
            
            if is_current:
                label = f'sampling itr {prev_itr}'
                plt.scatter(prev_x, prev_y, c=color, s=300, alpha=0.9, edgecolors='white', linewidth=2, label=label, zorder=10)
                current_added = True
            else:
                if not previous_added:
                    label = 'previous sampling'
                    previous_added = True
                else:
                    label = None
                plt.scatter(prev_x, prev_y, c=color, s=300, alpha=0.8, edgecolors='white', linewidth=1, label=label, zorder=9)

    plt.xlabel('UMAP feature 1', fontsize=36)
    plt.ylabel('UMAP feature 2', fontsize=36)
    plt.tick_params(axis='both', labelsize=36)
    plt.xlim(-5, 25)
    plt.ylim(-15, 25)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'distribution_umap_kde_sampling_repulsion_itr{i}_largepoint_energy.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'distribution_umap_kde_sampling_repulsion_itr{i}_largepoint_energy.svg', bbox_inches='tight')
    plt.show()

    print(f"Labeled dataset shape: {labeled_data_itr0.shape}")
    print(f"UMAP shape: {X_umap.shape}")
    print(f"Iteration {i} - sampling points found: {len(sampling_indices)}")
    if repulsion_energies is not None:
        print(f"Iteration {i} - log(repulsion energy) statistics:")
        print(f"  Mean: {repulsion_energies.mean():.6f}")
        print(f"  Std: {repulsion_energies.std():.6f}")
        print(f"  Min: {repulsion_energies.min():.6f}")
        print(f"  Max: {repulsion_energies.max():.6f}")
    print("="*60)