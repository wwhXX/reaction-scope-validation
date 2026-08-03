#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import sampling_util as cvt
import desc_gen_util


def init_sampling(csv_file):
    """
    Initialize sampling process with descriptor calculation
    
    Args:
        csv_file: path to the input CSV file containing SMILES data
    """
    print(f"=== Initializing sampling with {csv_file} ===")
    
    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"Error: Input file '{csv_file}' not found.")
        return False
    
    if not csv_file.endswith('.csv'):
        print("Error: Input file must be a CSV file.")
        return False
    
    # Create itr directory if it doesn't exist
    if not os.path.exists('./itr'):
        os.makedirs('./itr')
        print("Created 'itr' directory")
    
    try:
        # Step 1: Calculate descriptors using desc_gen_util
        print("Step 1: Calculating molecular descriptors...")
        desc_gen_util.spoc_descriptors(csv_file)
        
        # Get the descriptor file name
        name = os.path.splitext(os.path.basename(csv_file))[0]
        descriptor_file = f'fp_spoc_morgan41024_Maccs_{name}.csv'
        
        if not os.path.exists(descriptor_file):
            print(f"Error: Descriptor file '{descriptor_file}' was not created.")
            return False
        
        print(f"Descriptors calculated and saved to: {descriptor_file}")
        
        # Step 2: Load and combine data
        print("Step 2: Loading and combining data...")
        smi_data = pd.read_csv(csv_file)
        fp_data = pd.read_csv(descriptor_file)
        
        print(f"SMILES data shape: {smi_data.shape}")
        print(f"Fingerprint data shape: {fp_data.shape}")
        
        # Combine data
        all_data = pd.concat([smi_data, fp_data], axis=1)
        
        # Add ScreenLabel column
        all_data['ScreenLabel'] = 'BASE'
        
        # Step 3: Save initial iteration file
        all_data.to_csv('./itr/labeled_points_itr0.csv', index=False)
        
        print(f"Step 3: Initialization completed!")
        print(f"  - Total data points: {len(all_data)}")
        print(f"  - Saved to: ./itr/labeled_points_itr0.csv")
        print(f"  - Ready for sampling. Use 'python sampling.py start' to begin.")
        
        return True
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        return False


def find_latest_iteration():
    """
    Find the latest iteration number from existing files
    """
    if not os.path.exists('./itr'):
        return None
    
    max_itr = -1
    for filename in os.listdir('./itr'):
        if filename.startswith('labeled_points_itr') and filename.endswith('.csv'):
            try:
                itr_num = int(filename.replace('labeled_points_itr', '').replace('.csv', ''))
                max_itr = max(max_itr, itr_num)
            except ValueError:
                continue
    
    return max_itr if max_itr >= 0 else None


def update_labels_from_samples(data, samples_file='samples.csv'):
    """
    Update ScreenLabel in data based on samples.csv file
    
    Args:
        data: DataFrame with current labeled points
        samples_file: path to samples.csv file
    
    Returns:
        updated_data: DataFrame with updated ScreenLabels
    """
    if not os.path.exists(samples_file):
        print(f"No previous {samples_file} found, using current labels")
        return data.copy()
    
    try:
        samples_data = pd.read_csv(samples_file)
        print(f"Loaded {len(samples_data)} samples from {samples_file}")
        
        # Show current samples distribution
        if 'ScreenLabel' in samples_data.columns:
            sample_counts = samples_data['ScreenLabel'].value_counts()
            print("Previous samples ScreenLabel distribution:")
            for label, count in sample_counts.items():
                print(f"  {label}: {count}")
        
        updated_data = data.copy()
        
        # Update labels based on samples.csv
        if 'ScreenLabel' in samples_data.columns:
            # Find available identifier column
            id_col = None
            for col in ['smiles', 'SMILES', 'Index', 'index']:
                if col in samples_data.columns and col in updated_data.columns:
                    id_col = col
                    break
            
            if id_col:
                for _, sample_row in samples_data.iterrows():
                    id_value = sample_row[id_col]
                    screen_label = sample_row['ScreenLabel']
                    
                    # Find matching rows in the main data and update ScreenLabel
                    mask = updated_data[id_col] == id_value
                    if mask.any():
                        updated_data.loc[mask, 'ScreenLabel'] = screen_label
                        
                print(f"Labels updated from samples.csv using identifier column: {id_col}")
            else:
                print("Warning: No matching identifier columns found between samples.csv and data")
        else:
            print("Warning: samples.csv missing ScreenLabel column")
            
        return updated_data
        
    except Exception as e:
        print(f"Error reading {samples_file}: {e}")
        print("Using current labels without updates")
        return data.copy()


def start_sampling(sample_num=10):
    """
    Start iterative sampling process
    
    Args:
        sample_num: number of points to sample (default: 10)
    """
    print("=== Starting iterative sampling ===")
    print(f"Sampling number: {sample_num}")
    
    # Check if sampling has been initialized
    latest_itr = find_latest_iteration()
    
    if latest_itr is None:
        print("Error: No initialized data found.")
        print("Please run 'python sampling.py init <csv_file>' first to initialize.")
        return False
    
    print(f"Found existing sampling data up to iteration {latest_itr}")
    
    # Perform next iteration
    success = perform_sampling_iteration(latest_itr, sample_num)
    
    if success:
        print(f"\n=== Iteration {latest_itr + 1} completed successfully! ===")
        print("Run 'python sampling.py start' again to perform the next iteration.")
    else:
        print("\n=== Sampling iteration was not completed ===")
    
    return success


def perform_sampling_iteration(current_itr, sample_num=10):
    """
    Perform one iteration of sampling with updated workflow
    
    Args:
        current_itr: current iteration number
        sample_num: number of points to sample (default: 10)
    """
    next_itr = current_itr + 1
    print(f"\n--- Performing Iteration {next_itr} ---")
    
    # Load data from current iteration
    data_file = f'./itr/labeled_points_itr{current_itr}.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found")
        return False
    
    data = pd.read_csv(data_file)
    print(f"Loaded {len(data)} data points from iteration {current_itr}")
    
    # Step 1: Update labels from previous samples.csv
    print("\nStep 1: Updating labels from previous samples...")
    updated_data = update_labels_from_samples(data)
    
    # Check for pending points that must be resolved before continuing
    pending_points = updated_data[updated_data['ScreenLabel'] == 'Pending']
    if len(pending_points) > 0:
        print(f"\nError: Found {len(pending_points)} points with 'Pending' status.")
        print("All Pending points must be changed to 'Sampled' or 'Excluded' before continuing.")
        print("Please edit samples.csv to update these points, then run sampling again.")
        return False
    
    # Show current ScreenLabel distribution
    label_counts = updated_data['ScreenLabel'].value_counts()
    print("Current ScreenLabel distribution after update:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Step 2: Perform sampling using BASE points
    base_points = len(updated_data[updated_data['ScreenLabel'] == 'BASE'])
    if base_points == 0:
        print("Warning: No BASE points available for sampling")
        return False
        
    # Check if requested sample_num is available
    if sample_num > base_points:
        print(f"Warning: Requested sample_num ({sample_num}) exceeds available BASE points ({base_points})")
        print(f"Using all available BASE points: {base_points}")
        k_num = base_points
    else:
        k_num = sample_num
        
    print(f"\nStep 2: Sampling {k_num} points from {base_points} BASE points")
    
    # Perform sampling
    try:
        # Dynamically determine non-feature columns based on what exists in data
        # Only support Index, index, smiles, SMILES as non-feature columns
        potential_non_feature_cols = ['Index', 'index', 'smiles', 'SMILES', 'ScreenLabel']
        not_feature_cols = [col for col in potential_non_feature_cols if col in updated_data.columns]
        
        print(f"Non-feature columns used: {not_feature_cols}")
        
        sampled_points, labeled_points = cvt.get_sampling_weighted(
            data=updated_data, 
            task_itr_id=next_itr, 
            not_feature_cols=not_feature_cols, 
            k=k_num
        )
        sampled_points = sampled_points.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error during sampling: {e}")
        return False
    
    print(f"\nNewly sampled points for iteration {next_itr}:")
    # Display relevant columns for sampled points
    display_cols = []
    for col in ['Index', 'index', 'smiles', 'SMILES']:
        if col in sampled_points.columns:
            display_cols.append(col)
            
    if display_cols:
        print(sampled_points[display_cols].to_string(index=True))
    else:
        # Fallback: show first few columns if no recognizable identifier columns
        print("No standard identifier columns found, showing first few columns:")
        print(sampled_points.iloc[:, :min(3, sampled_points.shape[1])].to_string(index=True))
    
    # Step 3: Set default ScreenLabel to "Pending" for new samples
    sampled_points['ScreenLabel'] = 'Pending'
    
    # Step 4: Prepare samples.csv with only non-feature columns
    # Extract only identifier columns and ScreenLabel for user viewing
    samples_output_cols = []
    for col in ['Index', 'index', 'smiles', 'SMILES', 'ScreenLabel']:
        if col in sampled_points.columns:
            samples_output_cols.append(col)
    
    # Create user-friendly samples.csv with only non-feature columns
    samples_for_user = sampled_points[samples_output_cols].copy()
    samples_file = 'samples.csv'
    samples_for_user.to_csv(samples_file, index=False)
    print(f"\nStep 3: Saved {len(samples_for_user)} new samples to {samples_file}")
    print(f"Columns in samples.csv: {samples_output_cols}")
    print("Default ScreenLabel set to 'Pending' for all new samples")
    
    # Step 5: Update the main data and save to next iteration
    final_data = updated_data.copy()
    
    # Update ScreenLabel for newly sampled points in the main dataset
    for _, sample_row in sampled_points.iterrows():
        # Try to find matching identifier column
        matched = False
        for id_col in ['smiles', 'SMILES', 'Index', 'index']:
            if id_col in sample_row and id_col in final_data.columns:
                id_value = sample_row[id_col]
                mask = final_data[id_col] == id_value
                if mask.any():
                    final_data.loc[mask, 'ScreenLabel'] = 'Pending'
                    matched = True
                    break
        
        if not matched:
            print(f"Warning: Could not find matching identifier for sample row")
    
    # Save updated labeled points for next iteration
    labeled_file = f'./itr/labeled_points_itr{next_itr}.csv'
    final_data.to_csv(labeled_file, index=False)
    
    # Also save sampled points to itr directory for record keeping
    sampled_itr_file = f'./itr/sampled_points_itr{next_itr}.csv'
    sampled_points.to_csv(sampled_itr_file, index=False)
    
    # Update all_sampled.csv and all_excluded.csv with historical data
    # Read from latest iteration file (which contains all historical labels)
    update_summary_files(final_data, next_itr)
    
    print(f"\nResults saved:")
    print(f"  - {samples_file} (for manual labeling)")
    print(f"  - {labeled_file}")
    print(f"  - {sampled_itr_file}")
    print(f"  - all_sampled.csv (historical sampled points)")
    print(f"  - all_excluded.csv (historical excluded points)")
    
    # Summary
    print(f"\nSummary for iteration {next_itr}:")
    print(f"  - Total sampled: {len(sampled_points)}")
    print(f"  - All samples marked as 'Pending' by default")
    print(f"  - Please manually edit {samples_file} to change 'Pending' to 'Sampled' or 'Excluded'")
    print(f"  - Then run 'python sampling.py start' for next iteration")
    
    # Show final distribution
    final_label_counts = final_data['ScreenLabel'].value_counts()
    print(f"\nFinal ScreenLabel distribution:")
    for label, count in final_label_counts.items():
        print(f"  {label}: {count}")
    
    return True


def update_summary_files(final_data, current_itr):
    """
    Update all_sampled.csv and all_excluded.csv with historical data
    Simply read from the current iteration labeled_points_itr file
    
    Args:
        final_data: DataFrame with current iteration data
        current_itr: current iteration number
    """
    try:
        # Extract non-feature columns for summary files
        potential_non_feature_cols = ['Index', 'index', 'smiles', 'SMILES', 'ScreenLabel']
        summary_cols = [col for col in potential_non_feature_cols if col in final_data.columns]
        
        print(f"Updating summary files from current iteration data...")
        
        # Simply use the current final_data which contains all historical labels
        # Only include Sampled and Excluded points, not Pending points
        all_sampled = final_data[final_data['ScreenLabel'] == 'Sampled']
        all_excluded = final_data[final_data['ScreenLabel'].isin(['Excluded_Sampled', 'Excluded'])]
        pending_points = final_data[final_data['ScreenLabel'] == 'Pending']
        
        print(f"  Found {len(all_sampled)} total Sampled points")
        print(f"  Found {len(all_excluded)} total Excluded points") 
        print(f"  Found {len(pending_points)} total Pending points (not included in summary)")
        if len(all_excluded) > 0:
            print(f"  Excluded labels: {all_excluded['ScreenLabel'].value_counts().to_dict()}")
        
        # Create summary dataframes
        if len(all_sampled) > 0:
            combined_sampled = all_sampled[summary_cols].copy()
            # Remove duplicates
            id_cols = [col for col in summary_cols if col != 'ScreenLabel']
            if id_cols:
                combined_sampled = combined_sampled.drop_duplicates(subset=id_cols, keep='first')
        else:
            combined_sampled = pd.DataFrame(columns=summary_cols)
            
        if len(all_excluded) > 0:
            combined_excluded = all_excluded[summary_cols].copy()
            # Remove duplicates  
            id_cols = [col for col in summary_cols if col != 'ScreenLabel']
            if id_cols:
                combined_excluded = combined_excluded.drop_duplicates(subset=id_cols, keep='first')
        else:
            combined_excluded = pd.DataFrame(columns=summary_cols)
        
        # Save summary files
        combined_sampled.to_csv('all_sampled.csv', index=False)
        combined_excluded.to_csv('all_excluded.csv', index=False)
        
        print(f"Updated summary files:")
        print(f"  - all_sampled.csv: {len(combined_sampled)} sampled points")
        print(f"  - all_excluded.csv: {len(combined_excluded)} excluded points")
        
    except Exception as e:
        print(f"Warning: Failed to update summary files: {e}")


def stop_sampling():
    """
    Stop sampling process and finalize all_sampled.csv and all_excluded.csv with samples.csv
    """
    print("=== Stopping sampling process ===")
    
    # Check if there's a samples.csv to process
    if not os.path.exists('samples.csv'):
        print("No samples.csv found. Nothing to finalize.")
        return False
    
    try:
        # Read the final samples.csv
        samples_data = pd.read_csv('samples.csv')
        print(f"Processing final samples from samples.csv: {len(samples_data)} points")
        
        # Show current samples distribution
        if 'ScreenLabel' in samples_data.columns:
            sample_counts = samples_data['ScreenLabel'].value_counts()
            print("Final samples ScreenLabel distribution:")
            for label, count in sample_counts.items():
                print(f"  {label}: {count}")
        
        # Read existing summary files for reference only (we'll rebuild from iteration files)
        existing_sampled_count = 0
        existing_excluded_count = 0
        
        if os.path.exists('all_sampled.csv'):
            existing_sampled = pd.read_csv('all_sampled.csv')
            existing_sampled_count = len(existing_sampled)
            print(f"Existing all_sampled.csv: {existing_sampled_count} points (will be replaced)")
        else:
            print("No existing all_sampled.csv found")
            
        if os.path.exists('all_excluded.csv'):
            existing_excluded = pd.read_csv('all_excluded.csv')
            existing_excluded_count = len(existing_excluded)
            print(f"Existing all_excluded.csv: {existing_excluded_count} points (will be replaced)")
        else:
            print("No existing all_excluded.csv found")
        
        # Collect ALL historical excluded points from iteration files
        print("\nCollecting historical data from all iteration files...")
        all_historical_excluded = []
        all_historical_sampled = []
        
        # Find all iteration files
        import glob
        import re
        itr_files = []
        for filepath in glob.glob('./itr/labeled_points_itr*.csv'):
            # Extract iteration number using regex
            match = re.search(r'labeled_points_itr(\d+)\.csv', filepath)
            if match:
                itr_num = int(match.group(1))
                itr_files.append((itr_num, filepath))
        
        # Sort by iteration number
        itr_files.sort(key=lambda x: x[0])
        
        for itr_num, itr_file in itr_files:
            print(f"  Checking {itr_file}")
            itr_data = pd.read_csv(itr_file)
            itr_excluded = itr_data[itr_data['ScreenLabel'].isin(['Excluded_Sampled', 'Excluded'])]
            itr_sampled = itr_data[itr_data['ScreenLabel'] == 'Sampled']
            
            print(f"    Total rows: {len(itr_data)}, Excluded: {len(itr_excluded)}, Sampled: {len(itr_sampled)}")
            
            if len(itr_excluded) > 0:
                print(f"    Found excluded points with labels: {itr_excluded['ScreenLabel'].value_counts().to_dict()}")
                # Get the columns that exist in both the data and the expected summary columns
                available_cols = [col for col in ['smiles', 'SMILES', 'Index', 'index', 'ScreenLabel'] 
                                if col in itr_excluded.columns]
                print(f"    Available columns for excluded: {available_cols}")
                if available_cols:
                    all_historical_excluded.append(itr_excluded[available_cols])
            
            if len(itr_sampled) > 0:
                available_cols = [col for col in ['smiles', 'SMILES', 'Index', 'index', 'ScreenLabel'] 
                                if col in itr_sampled.columns]
                if available_cols:
                    all_historical_sampled.append(itr_sampled[available_cols])
        
        # Separate samples by ScreenLabel - check for multiple exclusion label variants
        final_sampled = samples_data[samples_data['ScreenLabel'] == 'Sampled'].copy()
        final_excluded = samples_data[samples_data['ScreenLabel'].isin(['Excluded_Sampled', 'Excluded'])].copy()
        final_pending = samples_data[samples_data['ScreenLabel'] == 'Pending'].copy()
        
        if len(final_pending) > 0:
            print(f"\nWarning: Found {len(final_pending)} points still marked as 'Pending'")
            print("These points will not be included in the final summary files.")
            print("Please change them to 'Sampled' or 'Excluded' if you want them included.")
        
        print(f"\nFinalizing:")
        print(f"  - Adding {len(final_sampled)} 'Sampled' points to all_sampled.csv")
        print(f"  - Adding {len(final_excluded)} 'Excluded' points to all_excluded.csv")
        print(f"  - Skipping {len(final_pending)} 'Pending' points")
        
        # Show what exclusion labels were found
        if len(final_excluded) > 0:
            exclusion_labels = final_excluded['ScreenLabel'].value_counts()
            print(f"  - Exclusion labels found: {dict(exclusion_labels)}")
        
        # Combine historical excluded data from iteration files
        historical_excluded_combined = pd.DataFrame()
        if all_historical_excluded:
            historical_excluded_combined = pd.concat(all_historical_excluded, ignore_index=True)
            # Remove duplicates
            available_id_cols = [col for col in ['smiles', 'SMILES', 'Index', 'index'] 
                               if col in historical_excluded_combined.columns]
            if available_id_cols:
                historical_excluded_combined = historical_excluded_combined.drop_duplicates(subset=available_id_cols, keep='last')
        
        # Combine historical sampled data from iteration files
        historical_sampled_combined = pd.DataFrame()
        if all_historical_sampled:
            historical_sampled_combined = pd.concat(all_historical_sampled, ignore_index=True)
            # Remove duplicates
            available_id_cols = [col for col in ['smiles', 'SMILES', 'Index', 'index'] 
                               if col in historical_sampled_combined.columns]
            if available_id_cols:
                historical_sampled_combined = historical_sampled_combined.drop_duplicates(subset=available_id_cols, keep='last')
        
        # For sampled points: combine historical data from iteration files with samples.csv
        if len(final_sampled) > 0:
            if len(historical_sampled_combined) > 0:
                # Ensure columns match
                common_cols = list(set(historical_sampled_combined.columns) & set(final_sampled.columns))
                combined_sampled = pd.concat([
                    historical_sampled_combined[common_cols], 
                    final_sampled[common_cols]
                ], ignore_index=True)
            else:
                combined_sampled = final_sampled.copy()
                
            # Remove duplicates
            available_id_cols = [col for col in ['smiles', 'SMILES', 'Index', 'index'] 
                               if col in combined_sampled.columns]
            if available_id_cols:
                combined_sampled = combined_sampled.drop_duplicates(subset=available_id_cols, keep='last')
        else:
            combined_sampled = historical_sampled_combined.copy()
        
        # Combine with final excluded from samples.csv
        if len(final_excluded) > 0:
            if len(historical_excluded_combined) > 0:
                # Ensure columns match
                common_cols = list(set(historical_excluded_combined.columns) & set(final_excluded.columns))
                combined_excluded = pd.concat([
                    historical_excluded_combined[common_cols], 
                    final_excluded[common_cols]
                ], ignore_index=True)
            else:
                combined_excluded = final_excluded.copy()
        else:
            combined_excluded = historical_excluded_combined.copy()
            
        # Final deduplication for excluded
        if len(combined_excluded) > 0:
            available_id_cols = [col for col in ['smiles', 'SMILES', 'Index', 'index'] 
                               if col in combined_excluded.columns]
            if available_id_cols:
                combined_excluded = combined_excluded.drop_duplicates(subset=available_id_cols, keep='last')
        
        print(f"\nFinal combination results:")
        print(f"  - Historical excluded from iteration files: {len(historical_excluded_combined)}")
        print(f"  - Historical sampled from iteration files: {len(historical_sampled_combined)}")
        print(f"  - Final excluded from samples.csv: {len(final_excluded)}")
        print(f"  - Final sampled from samples.csv: {len(final_sampled)}")
        print(f"  - Total combined excluded: {len(combined_excluded)}")
        print(f"  - Total combined sampled: {len(combined_sampled)}")
        print(f"\nComparison with previous summary files:")
        print(f"  - Previous all_sampled.csv: {existing_sampled_count} -> New: {len(combined_sampled)}")
        print(f"  - Previous all_excluded.csv: {existing_excluded_count} -> New: {len(combined_excluded)}")
        
        # Validate data before saving
        print(f"\nValidating final data...")
        
        # Check for required columns
        required_cols = ['smiles', 'SMILES', 'Index', 'index']
        sampled_id_cols = [col for col in required_cols if col in combined_sampled.columns] if len(combined_sampled) > 0 else []
        excluded_id_cols = [col for col in required_cols if col in combined_excluded.columns] if len(combined_excluded) > 0 else []
        
        if len(combined_sampled) > 0 and not sampled_id_cols:
            print("Warning: No identifier columns found in combined_sampled data")
        if len(combined_excluded) > 0 and not excluded_id_cols:
            print("Warning: No identifier columns found in combined_excluded data")
            
        # Archive samples.csv first
        import shutil
        if os.path.exists('samples.csv'):
            shutil.move('samples.csv', 'samples_final.csv')
            print(f"  - samples.csv moved to samples_final.csv")
        
        print(f"\nSaving final summary files...")
        
        # Save final summary files (this is the critical operation, done last)
        combined_sampled.to_csv('all_sampled.csv', index=False)
        combined_excluded.to_csv('all_excluded.csv', index=False)
        
        print(f"Successfully saved:")
        print(f"  - all_sampled.csv: {len(combined_sampled)} total sampled points")
        print(f"  - all_excluded.csv: {len(combined_excluded)} total excluded points")
        
        print(f"\nSampling process completed!")
        print(f"Final results are in all_sampled.csv and all_excluded.csv")
        
        return True
        
    except Exception as e:
        print(f"Error during stop process: {e}")
        return False


def show_help():
    """
    Show help information
    """
    print("=== ScopeMap Sampling Tool ===")
    print()
    print("Usage:")
    print("  python sampling.py init <csv_file>       - Initialize sampling with descriptor calculation")
    print("  python sampling.py start [sample_num]    - Start/continue iterative sampling")
    print("  python sampling.py stop                  - Stop sampling and finalize results")
    print("  python sampling.py help                  - Show this help message")
    print()
    print("Examples:")
    print("  python sampling.py init data.csv         - Initialize with data.csv")
    print("  python sampling.py start                 - Start sampling process (default sample_num=10)")
    print("  python sampling.py start 15              - Start sampling process (sample_num=15)")
    print("  python sampling.py stop                  - Finalize sampling and merge samples.csv into summary files")
    print()
    print("Workflow:")
    print("  1. Initialize with 'init' command")
    print("  2. Run 'start' to perform sampling (creates samples.csv with 'Pending' labels)")
    print("  3. Manually edit samples.csv to change 'Pending' to 'Sampled' or 'Excluded'")
    print("  4. Repeat steps 2-3 for multiple iterations")
    print("  5. Run 'stop' to finalize and complete the sampling process")
    print()
    print("Files:")
    print("  - Input CSV must contain identifier column (smiles/SMILES/Index/index)")
    print("  - Results are saved in './itr/' directory")
    print("  - samples.csv contains only identifier columns and ScreenLabel for manual editing")
    print("  - all_sampled.csv contains all historical sampled points (excluding current iteration)")
    print("  - all_excluded.csv contains all historical excluded points (excluding current iteration)")
    print("  - Feature columns are excluded from samples.csv for user convenience")
    print("  - Descriptor files are generated automatically")


def main():
    """
    Main function to handle command line arguments
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'init':
        if len(sys.argv) != 3:
            print("Error: 'init' command requires a CSV file argument")
            print("Usage: python sampling.py init <csv_file>")
            return
        
        csv_file = sys.argv[2]
        success = init_sampling(csv_file)
        
        if not success:
            sys.exit(1)
    
    elif command == 'start':
        # Parse sample_num parameter if provided
        sample_num = 10  # default value
        
        if len(sys.argv) == 3:
            try:
                sample_num = int(sys.argv[2])
                if sample_num <= 0:
                    print("Error: sample_num must be a positive integer")
                    return
            except ValueError:
                print("Error: sample_num must be a valid integer")
                print("Usage: python sampling.py start [sample_num]")
                return
        elif len(sys.argv) > 3:
            print("Error: Too many arguments for 'start' command")
            print("Usage: python sampling.py start [sample_num]")
            return
        
        success = start_sampling(sample_num)
        
        if not success:
            sys.exit(1)
    
    elif command == 'stop':
        success = stop_sampling()
        
        if not success:
            sys.exit(1)
    
    elif command in ['help', '-h', '--help']:
        show_help()
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Use 'python sampling.py help' for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()