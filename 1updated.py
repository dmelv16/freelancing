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
        
        # Create test_case folder
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
    n_points = len(group_df)
    print(f"  Processing ALL {n_points:,} points...")
    
    # Process each voltage column
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Segmentation Results - {dict(zip(group_cols, group_values))}\nTotal Points: {n_points:,}', 
                 fontsize=10)
    
    for idx, col in enumerate(voltage_cols):
        if col not in group_df.columns or group_df[col].isna().all():
            continue
        
        print(f"  Processing {col}...")
        
        # Extract voltage data as numpy array
        voltage_data = group_df[col].values
        
        # Calculate features - returns multiple feature columns
        print(f"    Calculating rolling averages and slopes (3, 5, 9 points)...")
        features = calculate_complete_features(voltage_data)
        print(f"    Feature matrix shape: {features.shape}")
        
        # Perform segmentation on ALL data
        try:
            print(f"    Running segmentation on all {n_points:,} points...")
            
            # Initialize with correct parameters
            segmenter = GreedyGaussianSegmenter(k_max=5, max_shuffles=1)
            
            # Match what your colleagues do: transpose the features
            # Original features shape: (n_timepoints, n_features)
            # Transposed shape: (n_features, n_timepoints)
            features_transposed = features.T
            
            print(f"    Transposed features shape: {features_transposed.shape}")
            print(f"    Features type: {type(features_transposed)}")
            
            # Run segmentation on transposed features
            segments = segmenter.fit_predict(features_transposed)
            
            print(f"    Segmentation complete. Segments shape: {segments.shape}")
            
            # Get segment statistics using all data
            stats = get_segment_stats_complete(voltage_data, segments)
            
            results[col] = {
                'segments': segments,
                'stats': stats,
                'features': features,
                'n_points': n_points,
                'n_segments': len(np.unique(segments))
            }
            
            # Visualize ALL points
            print(f"    Creating visualization with all {n_points:,} points...")
            visualize_all_points(axes[idx], voltage_data, features, segments, col, n_points)
            
        except Exception as e:
            print(f"  Error segmenting {col}: {e}")
            print(f"  Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if results:
        plt.tight_layout()
        filename = '_'.join([f"{k}={v}" for k, v in zip(group_cols, group_values)])
        
        # Save plot with all points visible
        print(f"  Saving visualization...")
        plt.savefig(test_case_folder / f"{filename}.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        # Save complete statistics
        save_complete_statistics(results, test_case_folder, filename, group_df)
        print(f"  Saved results for group {dict(zip(group_cols, group_values))}")

def calculate_complete_features(data):
    """
    Calculate rolling averages AND slopes for 3, 5, 9 point windows
    Returns a 2D array with shape (n_points, 6) where columns are:
    [avg_3, avg_5, avg_9, slope_3, slope_5, slope_9]
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
    # 3-point slope (centered difference when possible)
    for i in range(n):
        if i == 0:
            slope_3[i] = (data[min(1, n-1)] - data[0]) if n > 1 else 0
        elif i == n-1:
            slope_3[i] = (data[-1] - data[max(-2, -n)]) if n > 1 else 0
        else:
            slope_3[i] = (data[min(i+1, n-1)] - data[max(i-1, 0)]) / 2
    
    # 5-point slope (using linear regression coefficients)
    for i in range(n):
        start_idx = max(0, i-2)
        end_idx = min(n, i+3)
        window = data[start_idx:end_idx]
        if len(window) > 1:
            x = np.arange(len(window))
            slope_5[i] = np.polyfit(x, window, 1)[0]
        else:
            slope_5[i] = 0
    
    # 9-point slope (using linear regression coefficients)
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
    # Shape will be (n_points, 6)
    features = np.column_stack([avg_3, avg_5, avg_9, slope_3, slope_5, slope_9])
    
    return features

def visualize_all_points(axes, voltage_data, features, segments, col_name, n_points):
    """
    Visualization showing ALL data points without downsampling
    Modified to handle multi-dimensional features
    """
    ax1, ax2 = axes
    
    # Create time index for all points
    time_index = np.arange(n_points)
    
    # Get unique segments and colors
    unique_segments = np.unique(segments)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(unique_segments)))
    
    # Plot ALL voltage data points
    ax1.plot(time_index, voltage_data, alpha=0.6, linewidth=0.5, color='black', label='Voltage', zorder=1)
    
    # Overlay segments with transparency to see all points
    for seg_id, color in zip(unique_segments, colors):
        mask = segments == seg_id
        seg_indices = time_index[mask]
        # Use scatter for better visibility of all points
        ax1.scatter(seg_indices, voltage_data[mask], 
                   alpha=0.4, s=0.5, c=[color], label=f'Cluster {seg_id}', zorder=2)
    
    ax1.set_title(f'{col_name} - All {n_points:,} Points with Clusters')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Voltage')
    ax1.legend(loc='best', fontsize=8, markerscale=10)
    ax1.grid(True, alpha=0.3)
    
    # For features, plot the mean of all features or individual features
    # Since we now have 6 features, we can plot them all or just the mean
    feature_names = ['Avg-3', 'Avg-5', 'Avg-9', 'Slope-3', 'Slope-5', 'Slope-9']
    
    # Plot mean of all features
    mean_features = np.mean(features, axis=1)
    ax2.plot(time_index, mean_features, alpha=0.6, linewidth=0.5, 
             color='darkgreen', label='Mean of Features', zorder=1)
    
    # Overlay segments on features
    for seg_id, color in zip(unique_segments, colors):
        mask = segments == seg_id
        seg_indices = time_index[mask]
        ax2.scatter(seg_indices, mean_features[mask],
                   alpha=0.4, s=0.5, c=[color], label=f'Cluster {seg_id}', zorder=2)
    
    ax2.set_title(f'{col_name} - Mean of 6 Features (Avg & Slope: 3, 5, 9 points)')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Mean Feature Value')
    ax2.legend(loc='best', fontsize=8, markerscale=10)
    ax2.grid(True, alpha=0.3)
    
    # Add text showing we're displaying all points
    ax1.text(0.02, 0.98, f'Showing ALL {n_points:,} points', 
             transform=ax1.transAxes, fontsize=9, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.02, 0.98, f'Features: {features.shape[1]} dimensions', 
             transform=ax2.transAxes, fontsize=9, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

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
        
        # Calculate all statistics from complete data
        stats[f'cluster_{seg_id}'] = {
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
            'cv': float(np.std(seg_data) / (np.mean(seg_data) + 1e-10)),  # Coefficient of variation
            'start_idx': int(seg_indices[0]),
            'end_idx': int(seg_indices[-1]),
            'duration': int(seg_indices[-1] - seg_indices[0] + 1),
            'percentage': float(100 * len(seg_data) / len(data)),
            'density': float(len(seg_data) / (seg_indices[-1] - seg_indices[0] + 1)) if seg_indices[-1] > seg_indices[0] else 1.0
        }
    
    return stats

def save_complete_statistics(results, folder, filename, group_df):
    """
    Save comprehensive statistics and segment assignments
    Modified to handle multi-dimensional features
    """
    # Detailed statistics for each cluster
    stats_data = []
    for voltage_col, res in results.items():
        for cluster_id, stats in res['stats'].items():
            row = {
                'voltage_column': voltage_col,
                'cluster': cluster_id,
                'total_points_in_group': res['n_points'],
                'total_clusters': res['n_segments']
            }
            row.update(stats)
            stats_data.append(row)
    
    if stats_data:
        # Save detailed statistics
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(folder / f"{filename}_stats.csv", index=False)
        
        # Save segment assignments for all points
        for voltage_col, res in results.items():
            # For multi-dimensional features, save each feature separately
            features = res['features']
            feature_names = ['avg_3', 'avg_5', 'avg_9', 'slope_3', 'slope_5', 'slope_9']
            
            segment_data = {
                'timestamp': group_df['timestamp'].values,
                'voltage': group_df[voltage_col].values,
                'segment': res['segments']
            }
            
            # Add each feature as a separate column
            for i, feat_name in enumerate(feature_names):
                segment_data[f'feature_{feat_name}'] = features[:, i]
            
            # Also add mean of all features
            segment_data['feature_mean'] = np.mean(features, axis=1)
            
            segment_df = pd.DataFrame(segment_data)
            segment_df.to_csv(folder / f"{filename}_{voltage_col}_segments.csv", index=False)
        
        # Summary statistics
        summary = {
            'voltage_column': list(results.keys()),
            'n_clusters': [res['n_segments'] for res in results.values()],
            'total_points': [res['n_points'] for res in results.values()],
            'n_features': [res['features'].shape[1] for res in results.values()],
            'processing_status': ['complete - all points processed' for _ in results.values()]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(folder / f"{filename}_summary.csv", index=False)

# Main execution
if __name__ == "__main__":
    # Configuration
    PARQUET_FILE = "your_data.parquet"  # Replace with your file path
    OUTPUT_DIR = "segmentation_results"
    
    # Run processing
    print("Starting power data segmentation with ALL points...")
    process_power_data(PARQUET_FILE, OUTPUT_DIR)
    print(f"Results saved to {OUTPUT_DIR}/")
