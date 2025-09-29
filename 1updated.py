import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq
from aeon.segmentation import GreedyGaussianSegmenter
import warnings
warnings.filterwarnings('ignore')

def process_power_data(parquet_path, output_dir='segmentation_results'):
    """
    Process power data with greedy Gaussian segmentation
    Memory-efficient processing without any data elimination
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define columns
    group_cols = ['unit_id', 'save', 'ofp', 'station', 'test_case', 'test_run']
    voltage_cols = ['voltage_28v_dc1_cal', 'voltage_28v_dc2_cal']
    
    # First pass: identify unique groups
    print("Identifying unique groups...")
    parquet_file = pq.ParquetFile(parquet_path)
    unique_groups = set()
    
    for batch in parquet_file.iter_batches(batch_size=100000, columns=group_cols):
        df_batch = batch.to_pandas()
        groups = df_batch[group_cols].drop_duplicates()
        unique_groups.update([tuple(row) for _, row in groups.iterrows()])
    
    print(f"Found {len(unique_groups)} unique groups to process")
    
    # Process each group individually
    for group_idx, group_values in enumerate(unique_groups, 1):
        print(f"Processing group {group_idx}/{len(unique_groups)}: {dict(zip(group_cols, group_values))}")
        
        # Create filter expression for this group
        filters = [
            (col, '==', val) for col, val in zip(group_cols, group_values)
        ]
        
        # Read only this group's data
        group_df = read_group_data(parquet_path, filters, group_cols, voltage_cols)
        
        if group_df is None or len(group_df) < 10:
            continue
        
        # Sort by timestamp
        group_df = group_df.sort_values('timestamp')
        
        # Create test_case folder with subfolders for each voltage
        test_case_folder = output_path / f"test_case_{group_values[4]}"
        test_case_folder.mkdir(exist_ok=True)
        
        # Process ALL data points
        process_single_group_complete(group_df, group_values, group_cols, voltage_cols, test_case_folder)
        
        # Clear memory
        del group_df

def read_group_data(parquet_path, filters, group_cols, voltage_cols):
    """
    Read data for a specific group using filters
    """
    try:
        columns_to_read = ['timestamp'] + group_cols + voltage_cols
        df = pd.read_parquet(
            parquet_path,
            filters=filters,
            columns=columns_to_read,
            engine='pyarrow'
        )
        return df
    except Exception as e:
        print(f"Error reading group data: {e}")
        return None

def process_single_group_complete(group_df, group_values, group_cols, voltage_cols, test_case_folder):
    """
    Process a single group using ALL data points
    """
    n_points_original = len(group_df)
    print(f"  Original data points: {n_points_original:,}")
    
    # Process each voltage column
    results = {}
    
    for col in voltage_cols:
        if col not in group_df.columns:
            continue
        
        print(f"  Processing {col}...")
        
        # Create subfolder for this voltage column
        voltage_folder = test_case_folder / col.replace('voltage_28v_', '').replace('_cal', '')
        voltage_folder.mkdir(exist_ok=True)
        
        # Extract voltage data and timestamps
        voltage_data = group_df[col].values
        timestamps = group_df['timestamp'].values
        
        # REMOVE ALL NaN VALUES
        valid_mask = ~np.isnan(voltage_data)
        nan_count = np.sum(~valid_mask)
        
        if nan_count > 0:
            print(f"    Removing {nan_count} NaN values from {col}")
            voltage_data = voltage_data[valid_mask]
            timestamps = timestamps[valid_mask]
        
        n_points = len(voltage_data)
        
        if n_points < 10:
            print(f"    Skipping {col} - insufficient data after NaN removal ({n_points} points)")
            continue
        
        print(f"    Processing {n_points:,} valid points (removed {nan_count} NaN values)")
        
        # Calculate features - returns multiple feature columns
        print(f"    Calculating rolling averages and slopes (3, 5, 9 points)...")
        features = calculate_complete_features(voltage_data)
        print(f"    Feature matrix shape: {features.shape}")
        
        # Perform segmentation
        try:
            print(f"    Running segmentation...")
            
            # Initialize with correct parameters
            segmenter = GreedyGaussianSegmenter(k_max=5, max_shuffles=1)
            
            # Transpose features: (n_timepoints, n_features) -> (n_features, n_timepoints)
            features_transposed = features.T
            print(f"    Transposed features shape: {features_transposed.shape}")
            
            # Run segmentation
            segments = segmenter.fit_predict(features_transposed)
            print(f"    Segmentation complete. Found {len(np.unique(segments))} segments")
            
            # Get segment statistics
            stats = get_segment_stats_complete(voltage_data, segments)
            
            results[col] = {
                'segments': segments,
                'stats': stats,
                'features': features,
                'timestamps': timestamps,
                'voltage_data': voltage_data,
                'n_points': n_points,
                'n_segments': len(np.unique(segments)),
                'nan_removed': nan_count
            }
            
            # Create simple, clear visualization
            print(f"    Creating visualization...")
            create_simple_visualization(
                voltage_data, 
                segments, 
                col, 
                n_points,
                voltage_folder,
                group_values,
                group_cols
            )
            
            # Save segment data
            save_segment_data(results[col], voltage_folder, group_values, group_cols, col)
            
        except Exception as e:
            print(f"  Error segmenting {col}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary statistics
    if results:
        save_summary_statistics(results, test_case_folder, group_values, group_cols)
        print(f"  Saved results for group {dict(zip(group_cols, group_values))}")

def calculate_complete_features(data):
    """
    Calculate rolling averages AND slopes for 3, 5, 9 point windows
    Returns a 2D array with shape (n_points, 6) where columns are:
    [avg_3, avg_5, avg_9, slope_3, slope_5, slope_9]
    Assumes no NaN values in input data
    """
    n = len(data)
    
    # Pre-allocate arrays for all features
    avg_3 = np.zeros(n)
    avg_5 = np.zeros(n)
    avg_9 = np.zeros(n)
    slope_3 = np.zeros(n)
    slope_5 = np.zeros(n)
    slope_9 = np.zeros(n)
    
    # Vectorized rolling averages using convolution
    # 3-point average
    kernel_3 = np.ones(3) / 3
    if n >= 3:
        avg_3[1:n-1] = np.convolve(data, kernel_3, mode='valid')
    avg_3[0] = np.mean(data[:min(2, n)])
    avg_3[-1] = np.mean(data[max(-2, -n):])
    
    # 5-point average
    kernel_5 = np.ones(5) / 5
    if n >= 5:
        avg_5[2:n-2] = np.convolve(data, kernel_5, mode='valid')
        for i in range(min(2, n)):
            avg_5[i] = np.mean(data[:min(i+3, n)])
            if n-i-1 >= 0:
                avg_5[-(i+1)] = np.mean(data[max(-(i+3), -n):])
    else:
        avg_5[:] = np.mean(data)
    
    # 9-point average
    kernel_9 = np.ones(9) / 9
    if n >= 9:
        avg_9[4:n-4] = np.convolve(data, kernel_9, mode='valid')
        for i in range(min(4, n)):
            avg_9[i] = np.mean(data[:min(i+5, n)])
            if n-i-1 >= 0:
                avg_9[-(i+1)] = np.mean(data[max(-(i+5), -n):])
    else:
        avg_9[:] = np.mean(data)
    
    # Calculate slopes for 3, 5, 9 point windows
    # 3-point slope
    for i in range(n):
        if i == 0:
            slope_3[i] = (data[min(1, n-1)] - data[0]) if n > 1 else 0
        elif i == n-1:
            slope_3[i] = (data[-1] - data[max(-2, -n)]) if n > 1 else 0
        else:
            slope_3[i] = (data[min(i+1, n-1)] - data[max(i-1, 0)]) / 2
    
    # 5-point slope
    for i in range(n):
        start_idx = max(0, i-2)
        end_idx = min(n, i+3)
        window = data[start_idx:end_idx]
        if len(window) > 1:
            x = np.arange(len(window))
            slope_5[i] = np.polyfit(x, window, 1)[0]
        else:
            slope_5[i] = 0
    
    # 9-point slope
    for i in range(n):
        start_idx = max(0, i-4)
        end_idx = min(n, i+5)
        window = data[start_idx:end_idx]
        if len(window) > 1:
            x = np.arange(len(window))
            slope_9[i] = np.polyfit(x, window, 1)[0]
        else:
            slope_9[i] = 0
    
    # Stack all features as columns
    features = np.column_stack([avg_3, avg_5, avg_9, slope_3, slope_5, slope_9])
    
    return features

def create_simple_visualization(voltage_data, segments, col_name, n_points, 
                               voltage_folder, group_values, group_cols):
    """
    Create a simple, clear visualization with colored points for each segment
    """
    # Create figure with good size for viewing
    plt.figure(figsize=(20, 10))
    
    # Create time index
    time_index = np.arange(n_points)
    
    # Get unique segments and assign distinct colors
    unique_segments = np.unique(segments)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot each segment with its own color
    for i, seg_id in enumerate(unique_segments):
        mask = segments == seg_id
        color = colors[i % len(colors)]
        
        # Plot points for this segment
        # Use larger point size for visibility
        if n_points < 10000:
            # For smaller datasets, use larger points
            plt.scatter(time_index[mask], voltage_data[mask], 
                       c=color, s=10, alpha=0.8, label=f'Segment {seg_id}')
        else:
            # For larger datasets, use smaller points but still visible
            plt.scatter(time_index[mask], voltage_data[mask], 
                       c=color, s=2, alpha=0.6, label=f'Segment {seg_id}')
    
    # Add title and labels
    title = f"{col_name} - Segmentation Results\n"
    title += f"Total Points: {n_points:,} | Segments: {len(unique_segments)}"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time Index', fontsize=14)
    plt.ylabel('Voltage', fontsize=14)
    
    # Add legend
    plt.legend(loc='best', fontsize=12, markerscale=2)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add info box
    info_text = f"Group: {dict(zip(group_cols, group_values))}"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = '_'.join([f"{k}={v}" for k, v in zip(group_cols, group_values)])
    plt.savefig(voltage_folder / f"{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved visualization to {voltage_folder / f'{filename}.png'}")

def get_segment_stats_complete(data, segments):
    """
    Calculate comprehensive statistics for each segment
    """
    stats = {}
    unique_segments = np.unique(segments)
    
    for seg_id in unique_segments:
        mask = segments == seg_id
        seg_data = data[mask]
        seg_indices = np.where(mask)[0]
        
        stats[f'segment_{seg_id}'] = {
            'count': len(seg_data),
            'mean': float(np.mean(seg_data)),
            'std': float(np.std(seg_data)),
            'min': float(np.min(seg_data)),
            'max': float(np.max(seg_data)),
            'range': float(np.max(seg_data) - np.min(seg_data)),
            'q25': float(np.percentile(seg_data, 25)),
            'q50': float(np.percentile(seg_data, 50)),
            'q75': float(np.percentile(seg_data, 75)),
            'iqr': float(np.percentile(seg_data, 75) - np.percentile(seg_data, 25)),
            'cv': float(np.std(seg_data) / (np.mean(seg_data) + 1e-10)),
            'start_idx': int(seg_indices[0]),
            'end_idx': int(seg_indices[-1]),
            'duration': int(seg_indices[-1] - seg_indices[0] + 1),
            'percentage': float(100 * len(seg_data) / len(data))
        }
    
    return stats

def save_segment_data(result, voltage_folder, group_values, group_cols, voltage_col):
    """
    Save segment assignments and statistics for a single voltage column
    """
    filename_base = '_'.join([f"{k}={v}" for k, v in zip(group_cols, group_values)])
    
    # Save segment assignments
    segment_df = pd.DataFrame({
        'timestamp': result['timestamps'],
        'voltage': result['voltage_data'],
        'segment': result['segments']
    })
    segment_df.to_csv(voltage_folder / f"{filename_base}_segments.csv", index=False)
    
    # Save segment statistics
    stats_data = []
    for seg_name, stats in result['stats'].items():
        row = {'segment': seg_name}
        row.update(stats)
        stats_data.append(row)
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(voltage_folder / f"{filename_base}_statistics.csv", index=False)

def save_summary_statistics(results, test_case_folder, group_values, group_cols):
    """
    Save summary statistics for all voltage columns
    """
    filename = '_'.join([f"{k}={v}" for k, v in zip(group_cols, group_values)])
    
    summary = []
    for voltage_col, res in results.items():
        summary.append({
            'voltage_column': voltage_col,
            'n_segments': res['n_segments'],
            'n_points_processed': res['n_points'],
            'nan_values_removed': res['nan_removed'],
            'processing_status': 'complete'
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(test_case_folder / f"{filename}_summary.csv", index=False)

# Main execution
if __name__ == "__main__":
    # Configuration
    PARQUET_FILE = "your_data.parquet"  # Replace with your file path
    OUTPUT_DIR = "segmentation_results"
    
    # Run processing
    print("Starting power data segmentation...")
    print("NaN values will be removed from the data")
    process_power_data(PARQUET_FILE, OUTPUT_DIR)
    print(f"Results saved to {OUTPUT_DIR}/")
