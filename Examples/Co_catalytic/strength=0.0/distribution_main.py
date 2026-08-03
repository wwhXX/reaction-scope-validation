import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import umap

np.random.seed(42)

# Read 1700_final_norepeat.csv and remove smiles and conv columns
data_1700 = pd.read_csv('experimental_data_results_of_alcohol.csv')
data_1700_features = pd.read_csv('fp_spoc_morgan41024_Maccs_experimental_data_results_of_alcohol_alcohol.csv')

# Read zinc_aryl_aldehyde fingerprints
data_zinc = pd.read_csv('fp_spoc_morgan41024_Maccs_底物醇去重_alcohol.csv')

# Concatenate the two datasets
combined_data = pd.concat([data_1700_features, data_zinc], axis=0, ignore_index=True)

# Create labels to track data source: 0 for 1700_final_norepeat, 1 for zinc_aryl_aldehyde
data_source_labels = np.concatenate([
    np.zeros(len(data_1700_features)),
    np.ones(len(data_zinc))
])

# Perform TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(combined_data.values)
print('TSNE done')

# Create DataFrame for plotting
plot_data = pd.DataFrame(X_tsne, columns=['x', 'y'])
plot_data['data_source'] = data_source_labels

print(f"TSNE x range: [{X_tsne[:, 0].min():.2f}, {X_tsne[:, 0].max():.2f}]")
print(f"TSNE y range: [{X_tsne[:, 1].min():.2f}, {X_tsne[:, 1].max():.2f}]")


# Read reference files with smiles
zinc_ref = pd.read_csv('底物醇去重.csv')

def create_mapping_1700(df_1700):
    """Create mapping from smiles to row index for 1700 data"""
    mapping = {}
    for idx, row in df_1700.iterrows():
        mapping[row['SMILES']] = idx
    return mapping

def create_mapping_zinc(df_zinc_smiles):
    """Create mapping from smiles to row index for zinc data"""
    mapping = {}
    for idx, smiles in enumerate(df_zinc_smiles['smiles']):
        mapping[smiles] = idx
    return mapping

mapping_1700 = create_mapping_1700(data_1700)
print('1700 mapping done')
mapping_zinc = create_mapping_zinc(zinc_ref)
print('Zinc mapping done')

def find_tsne_indices(smiles_list, mapping_1700, mapping_zinc, offset_zinc):
    """Find TSNE indices for given smiles"""
    indices = []
    colors = []
    
    for smiles in smiles_list:
        if smiles in mapping_1700:
            indices.append(mapping_1700[smiles])
            colors.append('blue')
        elif smiles in mapping_zinc:
            indices.append(mapping_zinc[smiles] + offset_zinc)
            colors.append('blue')
            
    return indices, colors

offset_zinc = len(data_1700_features)

final_sampling = pd.read_csv('sampled_points_itr1.csv')
final_sampling = pd.DataFrame(final_sampling.loc[:,'smiles'])
final_sampling.columns = ['smiles']
print(final_sampling)

sampling_indices, sampling_colors = find_tsne_indices(
    final_sampling['smiles'].tolist(), mapping_1700, mapping_zinc, offset_zinc
)

plt.figure(figsize=(12, 10))

sns.kdeplot(data=plot_data, x='x', y='y', fill=True, cmap='Blues', alpha=0.3, levels=8)

if sampling_indices:
    sampling_x = [X_tsne[idx][0] for idx in sampling_indices]
    sampling_y = [X_tsne[idx][1] for idx in sampling_indices]
    
    colors = ['#808080' if i < 7 else '#ff6b35' for i in range(len(sampling_x))]
    
    plt.scatter(sampling_x, sampling_y, c=colors, s=200, alpha=0.8, edgecolors='white', linewidth=1)

plt.xlabel('t-SNE feature 1', fontsize=20)
plt.ylabel('t-SNE feature 2', fontsize=20)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('distribution_tsne_kde_sampling_itr1_largepoint.png', dpi=300, bbox_inches='tight')
plt.savefig('distribution_tsne_kde_sampling_itr1_largepoint.svg', bbox_inches='tight')
plt.show()

print(f"Combined dataset shape: {combined_data.shape}")
print(f"1700 data points: {len(data_1700_features)}")
print(f"ZINC data points: {len(data_zinc)}")
print(f"Iteration 1 - sampling points found: {len(sampling_indices)}")