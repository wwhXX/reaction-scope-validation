import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Load data and prepare UMAP (same as original script)
np.random.seed(42)
data = pd.read_csv('1700_final_norepeat.csv')
data_features = data.drop(['smiles', 'conv'], axis=1)

sampled_points = pd.read_csv('人工.csv')
sampled_points_feature = pd.read_csv('fp_spoc_morgan41024_Maccs_人工_alcohol.csv')
sampled_raw_data = pd.concat([sampled_points, sampled_points_feature], axis=1)
sampled_data = sampled_raw_data.drop(['smiles'], axis=1)
print(sampled_data.head())
data = pd.concat([data_features, sampled_data], axis=0)
labels = np.concatenate([np.zeros(len(data_features)), np.ones(len(sampled_data))])
print(data.head())

# Calculate entropy for sampled_data before dimensionality reduction
def calculate_entropy_multivariate(data, bins=10):
    """Calculate entropy for multivariate data using product of marginal distributions"""
    n_features = data.shape[1]
    total_entropy = 0
    
    for i in range(n_features):
        # Calculate histogram for each feature
        hist, _ = np.histogram(data.iloc[:, i], bins=bins, density=True)
        
        # Normalize to get probabilities
        bin_width = (data.iloc[:, i].max() - data.iloc[:, i].min()) / bins
        prob = hist * bin_width
        prob = prob / prob.sum()  # Ensure normalization
        
        # Remove zero entries to avoid log(0)
        prob_nonzero = prob[prob > 0]
        
        # Calculate entropy for this feature
        if len(prob_nonzero) > 0:
            feature_entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero))
            total_entropy += feature_entropy
    
    return total_entropy

# Calculate entropy for sampled data
sampled_entropy = calculate_entropy_multivariate(sampled_data)
print(f"Entropy of sampled data (original features): {sampled_entropy:.3f} bits")

# Use only initial data for UMAP
X = data_features.values

# Fit UMAP on initial data only
umap_reducer = UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)

# Create dataframe with UMAP coordinates for initial data
data_df = pd.DataFrame(X_umap, columns=['x', 'y'])
data_df['labels'] = 0  # All initial data labeled as 0

# Function to project sampled data onto trained UMAP
def find_closest_umap_coordinates(umap_reducer, sampled_features):
    """
    Project sampled points onto the already trained UMAP model
    """
    # Transform sampled features using the trained UMAP model
    sampled_umap_coords = umap_reducer.transform(sampled_features)
    
    return sampled_umap_coords

# Get UMAP coordinates for sampled points
sampled_umap_coords = find_closest_umap_coordinates(
    umap_reducer, 
    sampled_data.values
)

# Create dataframe for sampled points
plot_data_sampled = pd.DataFrame(sampled_umap_coords, columns=['x', 'y'])
plot_data_sampled['labels'] = 1

# Use initial data for KDE plot
plot_data_1700 = data_df

# Generate the total plot (same as original)
plt.figure(figsize=(10, 8))
sns.kdeplot(data=plot_data_1700, x='x', y='y', fill=True, cmap='Blues', thresh=0, n_levels=8, alpha=0.4)
sns.scatterplot(data=plot_data_sampled, x='x', y='y', hue='labels', palette='Blues', legend=False, alpha=0.8, s=200)
plt.title('Manual', fontsize=16)
plt.xlabel('UMAP feature 1', fontsize=20)
plt.ylabel('UMAP feature 2', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=21)

plt.savefig('umap_morgan_maccs_sampling_cvt_umap.png')
plt.savefig('umap_morgan_maccs_sampling_cvt_umap.svg')
plt.close()
print("Generated total UMAP plot")

# Generate individual plots for each sampled point
for i, (idx, sampled_point) in enumerate(plot_data_sampled.iterrows()):
    plt.figure(figsize=(10, 8))
    
    # Plot the KDE for all 1700 data points (same as original)
    sns.kdeplot(data=plot_data_1700, x='x', y='y', fill=True, cmap='Blues', thresh=0, n_levels=8, alpha=0.4)
    
    # Plot only the current sampled point (use same style as total plot)
    single_point_df = pd.DataFrame({'x': [sampled_point['x']], 'y': [sampled_point['y']], 'labels': [1]})
    sns.scatterplot(data=single_point_df, x='x', y='y', hue='labels', palette='Blues', legend=False, alpha=0.8, s=200)
    
    plt.title('UMAP visualization of sampling')
    plt.xlabel('UMAP feature 1')
    plt.ylabel('UMAP feature 2')
    
    # Get the aldehyde name for this sampled point
    aldehyde_name = sampled_points.iloc[i]['smiles']
    # Clean the aldehyde name for filename (remove special characters)
    clean_name = aldehyde_name.replace('=', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('+', '').replace('-', '').replace('#', '').replace('/', '').replace('\\', '').replace(':', '')
    
    # Save the plot
    filename = f'umap_individual_{clean_name}_umap.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated plot for {aldehyde_name} -> {filename}")

print(f"Generated {len(plot_data_sampled)} individual UMAP plots")