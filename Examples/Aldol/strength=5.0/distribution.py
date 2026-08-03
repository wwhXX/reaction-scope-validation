import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from umap import UMAP

np.random.seed(42)

# Read 1700_final_norepeat.csv and remove smiles and conv columns
data_1700 = pd.read_csv('1700_final_norepeat.csv')
data_1700_features = data_1700.drop(['smiles', 'conv'], axis=1)

# Read zinc_aryl_aldehyde fingerprints
data_zinc = pd.read_csv('fp_spoc_morgan41024_Maccs_zinc_aryl_aldehyde_al.csv')

# Concatenate the two datasets
combined_data = pd.concat([data_1700_features, data_zinc], axis=0, ignore_index=True)

# Create labels to track data source: 0 for 1700_final_norepeat, 1 for zinc_aryl_aldehyde
data_source_labels = np.concatenate([
    np.zeros(len(data_1700_features)),
    np.ones(len(data_zinc))
])

# Perform UMAP
umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(combined_data.values)
print('UMAP done')

# Create DataFrame for plotting
plot_data = pd.DataFrame(X_umap, columns=['x', 'y'])
plot_data['data_source'] = data_source_labels

# Read sampling files
final_sampling = pd.read_csv('final_sampling.csv')
final_drop = pd.read_csv('final_drop.csv')

# Read reference files with smiles
zinc_ref = pd.read_csv('zinc_aryl_aldehyde.csv')

# Create mapping dictionaries from smiles to features
def create_mapping_1700(df_1700):
    """Create mapping from smiles to row index for 1700 data"""
    mapping = {}
    for idx, row in df_1700.iterrows():
        mapping[row['smiles']] = idx
    return mapping

def create_mapping_zinc(df_zinc_smiles):
    """Create mapping from smiles to row index for zinc data"""
    mapping = {}
    # Assuming the order is the same in both files
    for idx, smiles in enumerate(df_zinc_smiles['smiles']):
        mapping[smiles] = idx
    return mapping

# Create mappings (optimized)
mapping_1700 = create_mapping_1700(data_1700)
print('1700 mapping done')
mapping_zinc = create_mapping_zinc(zinc_ref)
print('Zinc mapping done')

# Find indices in the combined UMAP data for sampling points
def find_umap_indices(smiles_list, mapping_1700, mapping_zinc, offset_zinc):
    """Find UMAP indices for given smiles"""
    indices = []
    colors = []
    
    for smiles in smiles_list:
        if smiles in mapping_1700:
            indices.append(mapping_1700[smiles])
            colors.append('blue')  # 1700_final_norepeat points
        elif smiles in mapping_zinc:
            indices.append(mapping_zinc[smiles] + offset_zinc)
            colors.append('orange')  # zinc_aryl_aldehyde points
            
    return indices, colors

offset_zinc = len(data_1700_features)  # Offset for zinc data in combined dataset

# Get indices and colors for sampling data
sampling_indices, sampling_colors = find_umap_indices(
    final_sampling['smiles'].tolist(), mapping_1700, mapping_zinc, offset_zinc
)

# Get indices for drop data (all gray)
drop_indices, _ = find_umap_indices(
    final_drop['smiles'].tolist(), mapping_1700, mapping_zinc, offset_zinc
)
print('Indices found')

# Create the plot
plt.figure(figsize=(12, 10))

# Create single KDE plot for all data (blue)
sns.kdeplot(data=plot_data, x='x', y='y', fill=True, cmap='Blues', alpha=0.3, levels=8)

# Add scatter plots for final_sampling (blue for 1700, orange for zinc)
if sampling_indices:
    sampling_x = [X_umap[i][0] for i in sampling_indices]
    sampling_y = [X_umap[i][1] for i in sampling_indices]
    
    # Separate by color
    blue_x = [x for x, c in zip(sampling_x, sampling_colors) if c == 'blue']
    blue_y = [y for y, c in zip(sampling_y, sampling_colors) if c == 'blue']
    orange_x = [x for x, c in zip(sampling_x, sampling_colors) if c == 'orange']
    orange_y = [y for y, c in zip(sampling_y, sampling_colors) if c == 'orange']
    
    if blue_x:
        plt.scatter(blue_x, blue_y, c='#79b6da', s=450, alpha=0.8, edgecolors='white', linewidth=1, label='Final sampling (Exp)')
    if orange_x:
        plt.scatter(orange_x, orange_y, c='#ff8040', s=450, alpha=0.8, edgecolors='white', linewidth=1, label='Final sampling (ZINC)')

# Add scatter plots for final_drop (gray)
if drop_indices:
    drop_x = [X_umap[i][0] for i in drop_indices]
    drop_y = [X_umap[i][1] for i in drop_indices]
    plt.scatter(drop_x, drop_y, c='gray', s=450, alpha=0.8, edgecolors='darkgray', linewidth=1, label='Final drop')

# Customize the plot
plt.xlabel('UMAP feature 1', fontsize=37)
plt.ylabel('UMAP feature 2', fontsize=37)
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.tick_params(axis='both', which='major', labelsize=34)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('distribution_umap_kde_sampling.png', dpi=300, bbox_inches='tight')
plt.savefig('distribution_umap_kde_sampling.svg', bbox_inches='tight')
plt.show()

print(f"Combined dataset shape: {combined_data.shape}")
print(f"1700 data points: {len(data_1700_features)}")
print(f"ZINC data points: {len(data_zinc)}")
print(f"Final sampling points found: {len(sampling_indices)}")
print(f"Final drop points found: {len(drop_indices)}")