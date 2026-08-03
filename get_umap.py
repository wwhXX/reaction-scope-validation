import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import os
import sys
from pathlib import Path

def plot_umap(csv_file):
    """Plot a UMAP visualization from a labeled CSV file."""
    print(f"=== UMAP Visualization for {csv_file} ===")

    sns.set_style("white")

    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return False

    print(f"Loading {csv_file}...")
    labeled_data = pd.read_csv(csv_file)
    print(f"Loaded {len(labeled_data)} data points")

    if 'ScreenLabel' in labeled_data.columns:
        label_counts = labeled_data['ScreenLabel'].value_counts()
        print("ScreenLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("Warning: No ScreenLabel column found")
        return False

    non_feature_cols = ['Index', 'index', 'smiles', 'SMILES', 'ScreenLabel']
    feature_cols = [col for col in labeled_data.columns if col not in non_feature_cols]

    print(f"Using {len(feature_cols)} feature columns for UMAP")

    if len(feature_cols) == 0:
        print("Error: No feature columns found")
        return False

    X = labeled_data[feature_cols].values
    labels = labeled_data['ScreenLabel'].values

    print("Applying UMAP dimensionality reduction...")

    init_fp_file = 'fp_spoc_morgan41024_Maccs_smiles_for_screen_all.csv'

    if os.path.exists(init_fp_file):
        print("Using original fingerprint file for consistent UMAP initialization...")

        df_init_fp = pd.read_csv(init_fp_file)

        if 'SMILES' in df_init_fp.columns:
            df_init_features = df_init_fp.drop('SMILES', axis=1)
        else:
            df_init_features = df_init_fp

        X_init = df_init_features.values

        print(f"Training UMAP on {len(X_init)} initialization points...")
        umap_reducer = UMAP(n_components=2, random_state=42)
        umap_reducer.fit(X_init)

        X_umap = umap_reducer.transform(X)
        print("Applied fitted UMAP to iteration data for consistent visualization")
    else:
        print(f"Warning: Initialization file {init_fp_file} not found, using direct UMAP on iteration data")
        umap_reducer = UMAP(n_components=2, random_state=42)
        X_umap = umap_reducer.fit_transform(X)

    plt.rcParams['font.size'] = 20  # Scale all font sizes by 2x
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 16

    plt.figure(figsize=(12, 8))

    colors = {
        'BASE': 'lightblue',
        'Sampled': 'red', 
        'Excluded': 'gray',
        'Excluded_Sampled': 'darkgray',
        'Pending': 'orange'
    }

    for label in colors.keys():
        mask = labels == label
        if np.any(mask):
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=colors[label], alpha=0.6, s=50,
                       label=f'{label} ({np.sum(mask)})',
                       edgecolors='none')

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label not in colors:
            mask = labels == label
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c='purple', alpha=0.6, s=50,
                       label=f'{label} ({np.sum(mask)})',
                       edgecolors='none')

    csv_name = Path(csv_file).stem
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title(f'UMAP Visualization - {csv_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_file = f'{csv_name}_umap_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nUMAP visualization completed!")
    print(f"Plot saved as '{output_file}'")
    print(f"Total points: {len(X_umap)}")

    return True

def main():
    if len(sys.argv) != 2:
        print("Error: Expected exactly one CSV file path")
        print("Usage: python get_umap.py <csv_file>")
        return

    csv_file = sys.argv[1]
    success = plot_umap(csv_file)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()