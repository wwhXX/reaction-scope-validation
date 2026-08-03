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

# Perform UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(combined_data.values)
print('UMAP done')

# Create DataFrame for plotting
plot_data = pd.DataFrame(X_umap, columns=['x', 'y'])
plot_data['data_source'] = data_source_labels

print(f"UMAP x range: [{X_umap[:, 0].min():.2f}, {X_umap[:, 0].max():.2f}]")
print(f"UMAP y range: [{X_umap[:, 1].min():.2f}, {X_umap[:, 1].max():.2f}]")


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

all_sampling_points = []

for i in range(1, 7):
    final_sampling = pd.read_csv(f'sampled_points_itr{i}.csv')
    final_sampling = pd.DataFrame(final_sampling.loc[:,'smiles'])
    final_sampling.columns = ['smiles']
    print(final_sampling)

    sampling_indices, sampling_colors = find_tsne_indices(
        final_sampling['smiles'].tolist(), mapping_1700, mapping_zinc, offset_zinc
    )
    
    all_sampling_points.append({
        'indices': sampling_indices,
        'colors': sampling_colors,
        'iteration': i
    })

    plt.figure(figsize=(12, 10))

    sns.kdeplot(data=plot_data, x='x', y='y', fill=True, cmap='Blues', alpha=0.3, levels=8)

    previous_added = False
    current_added = False
    
    for j, prev_sampling in enumerate(all_sampling_points):
        prev_indices = prev_sampling['indices']
        prev_colors = prev_sampling['colors']
        prev_itr = prev_sampling['iteration']
        
        if prev_indices:
            prev_x = [X_umap[idx][0] for idx in prev_indices]
            prev_y = [X_umap[idx][1] for idx in prev_indices]
            
            is_current = (prev_itr == i)
            color = '#ff6b35' if is_current else '#4A90E2'
            
            if is_current:
                label = f'sampling itr {prev_itr}'
                plt.scatter(prev_x, prev_y, c=color, s=50, alpha=0.8, edgecolors='white', linewidth=1, label=label)
                current_added = True
            else:
                if not previous_added:
                    label = 'previous sampling'
                    previous_added = True
                else:
                    label = None
                plt.scatter(prev_x, prev_y, c=color, s=50, alpha=0.8, edgecolors='white', linewidth=1, label=label)

    plt.xlabel('UMAP feature 1', fontsize=20)
    plt.ylabel('UMAP feature 2', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=18)
    plt.xlim(-5, 25)
    plt.ylim(-15, 25)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'distribution_umap_kde_sampling_itr{i}_largepoint.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'distribution_umap_kde_sampling_itr{i}_largepoint.svg', bbox_inches='tight')
    plt.show()

    print(f"Combined dataset shape: {combined_data.shape}")
    print(f"1700 data points: {len(data_1700_features)}")
    print(f"ZINC data points: {len(data_zinc)}")
    print(f"Iteration {i} - sampling points found: {len(sampling_indices)}")