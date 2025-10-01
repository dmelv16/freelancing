import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from scipy.stats import zscore, linregress
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SimplifiedVoltageAnalyzer:
    """
    Voltage analyzer using actual classification functions while maintaining simplified output.
    """
    
    def __init__(self, input_folder='greedygaussv4', output_folder='voltage_analysis'):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Voltage thresholds for outlier detection
        self.deenergized_max = 2.0  
        self.operational_min = 18.0  
        self.operational_max = 29.0
        
        # Only flag anomalies in steady state
        self.steady_state_thresholds = {
            'max_variance': 1.0,
            'max_std': 1.0,
            'max_slope': 0.05,
            'outlier_threshold': 3  # z-score for steady state only
        }
        
        self.results = []
        self.failed_files = []
    
    def parse_filename(self, filename):
        """Parse filename handling hyphens properly."""
        # Remove .csv and _segments
        base = filename.replace('.csv', '').replace('_segments', '')
        
        # Handle known multi-word keys
        base = base.replace('unit_id=', 'unitid=')
        base = base.replace('test_case=', 'testcase=')
        base = base.replace('test_run=', 'testrun=')
        
        parts = {}
        segments = base.split('_')
        
        current_key = None
        current_value = []
        
        for segment in segments:
            if '=' in segment:
                # Save previous if exists
                if current_key and current_value:
                    parts[current_key] = '_'.join(current_value)
                
                key, value = segment.split('=', 1)
                current_key = key
                current_value = [value] if value else []
            else:
                if current_key:
                    current_value.append(segment)
        
        # Save last pair
        if current_key and current_value:
            parts[current_key] = '_'.join(current_value)
        
        # Restore original names
        if 'unitid' in parts:
            parts['unit_id'] = parts.pop('unitid')
        if 'testcase' in parts:
            parts['test_case'] = parts.pop('testcase')
        if 'testrun' in parts:
            parts['test_run'] = parts.pop('testrun')
        
        return parts
    
    def is_deenergized(self, voltages, labels, timestamps, slope_threshold=0.1, mean_threshold=5):
        """
        Actual is_deenergized function with cluster analysis.
        """
        from scipy.stats import linregress
        
        unique_labels = np.unique(labels)
        deenergized_clusters = np.zeros(len(unique_labels), dtype=bool)
        cluster_mean_voltages = []
        
        for ix, lab in enumerate(unique_labels):
            cluster_mask = labels == lab
            cluster_voltages = voltages[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            
            if len(cluster_voltages) > 1:
                try:
                    cluster_mean_voltage = np.mean(cluster_voltages)
                    cluster_abs_slope = np.abs(
                        linregress(cluster_timestamps, cluster_voltages).slope
                    )
                    
                    if (cluster_abs_slope < slope_threshold and 
                        cluster_mean_voltage < mean_threshold):
                        deenergized_clusters[ix] = True
                    cluster_mean_voltages.append(cluster_mean_voltage)
                except:
                    cluster_mean_voltages.append(np.mean(cluster_voltages))
            else:
                cluster_mean_voltages.append(np.mean(cluster_voltages))
        
        # Check for de-energized clusters between other de-energized clusters
        if len(deenergized_clusters) > 2:
            for i in range(1, len(deenergized_clusters) - 1):
                if (deenergized_clusters[i-1] and deenergized_clusters[i+1] and 
                    cluster_mean_voltages[i] < mean_threshold):
                    deenergized_clusters[i] = True
        
        # Create mask for all points
        deenergized_mask = np.zeros(len(voltages), dtype=bool)
        for ix, lab in enumerate(unique_labels):
            if deenergized_clusters[ix]:
                deenergized_mask[labels == lab] = True
        
        return deenergized_mask
    
    def is_stabilizing(self, voltages, labels, timestamps, slope_cutoff=1):
        """
        Actual is_stabilizing function with cluster analysis.
        """
        from scipy.stats import linregress
        
        unique_labels = np.unique(labels)
        stabilizing_clusters = np.zeros(len(unique_labels), dtype=bool)
        
        for ix, lab in enumerate(unique_labels):
            cluster_mask = labels == lab
            cluster_voltages = voltages[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            
            if len(cluster_voltages) > 1:
                try:
                    cluster_abs_slope = np.abs(
                        linregress(cluster_timestamps, cluster_voltages).slope
                    )
                    if cluster_abs_slope > slope_cutoff:
                        stabilizing_clusters[ix] = True
                except:
                    pass
        
        # Create mask for all points
        stabilizing_mask = np.zeros(len(voltages), dtype=bool)
        for ix, lab in enumerate(unique_labels):
            if stabilizing_clusters[ix]:
                stabilizing_mask[labels == lab] = True
        
        return stabilizing_mask
    
    def is_steadystate(self, voltages, labels, timestamps, slope_threshold=0.1, mean_threshold=20):
        """
        Actual is_steadystate function with cluster analysis.
        """
        from scipy.stats import linregress
        
        unique_labels = np.unique(labels)
        steadystate_clusters = np.zeros(len(unique_labels), dtype=bool)
        cluster_mean_voltages = []
        
        for ix, lab in enumerate(unique_labels):
            cluster_mask = labels == lab
            cluster_voltages = voltages[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            
            if len(cluster_voltages) > 1:
                try:
                    cluster_mean_voltage = np.mean(cluster_voltages)
                    cluster_abs_slope = np.abs(
                        linregress(cluster_timestamps, cluster_voltages).slope
                    )
                    
                    if (cluster_abs_slope < slope_threshold and 
                        cluster_mean_voltage > mean_threshold):
                        steadystate_clusters[ix] = True
                    cluster_mean_voltages.append(cluster_mean_voltage)
                except:
                    cluster_mean_voltages.append(np.mean(cluster_voltages))
            else:
                cluster_mean_voltages.append(np.mean(cluster_voltages))
        
        # Check for steady state clusters between other steady state clusters
        if len(steadystate_clusters) > 2:
            for i in range(1, len(steadystate_clusters) - 1):
                if (steadystate_clusters[i-1] and steadystate_clusters[i+1] and 
                    cluster_mean_voltages[i] > mean_threshold):
                    steadystate_clusters[i] = True
        
        # Create mask for all points
        steadystate_mask = np.zeros(len(voltages), dtype=bool)
        for ix, lab in enumerate(unique_labels):
            if steadystate_clusters[ix]:
                steadystate_mask[labels == lab] = True
        
        return steadystate_mask
    
    def classify_segments(self, df):
        """
        Use the actual classification functions to label segments.
        """
        # Get arrays
        voltages = df['voltage'].to_numpy()
        labels = df['segment'].to_numpy()
        timestamps = df['timestamp'].to_numpy()
        
        # Get masks from classification functions
        deenergized_mask = self.is_deenergized(voltages, labels, timestamps)
        stabilizing_mask = self.is_stabilizing(voltages, labels, timestamps)
        steadystate_mask = self.is_steadystate(voltages, labels, timestamps)
        
        # Apply labels based on masks
        df['label'] = 'unidentified'
        df.loc[deenergized_mask, 'label'] = 'de-energized'
        df.loc[stabilizing_mask, 'label'] = 'stabilizing'
        df.loc[steadystate_mask, 'label'] = 'steady_state'
        
        return df
    
    def analyze_csv(self, csv_path):
        """Analyze a single CSV file with comprehensive statistics."""
        try:
            filename = csv_path.name
            grouping = self.parse_filename(filename)
            
            # Add folder info
            parent_path = csv_path.parent
            grouping['dc_folder'] = parent_path.name  # dc1 or dc2
            grouping['test_case_folder'] = parent_path.parent.name
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check required columns
            if not all(col in df.columns for col in ['voltage', 'timestamp', 'segment']):
                self.failed_files.append((filename, "Missing required columns"))
                return None
            
            # Classify segments using actual functions
            df = self.classify_segments(df)
            
            # Calculate metrics for each label type
            results = []
            for label in df['label'].unique():
                label_data = df[df['label'] == label]
                voltage_values = label_data['voltage'].values
                
                if len(voltage_values) == 0:
                    continue
                
                # Comprehensive metrics for all
                metrics = {
                    **grouping,
                    'label': label,
                    'n_points': len(voltage_values),
                    'mean_voltage': np.mean(voltage_values),
                    'median_voltage': np.median(voltage_values),
                    'std': np.std(voltage_values),
                    'variance': np.var(voltage_values),
                    'min_voltage': np.min(voltage_values),
                    'max_voltage': np.max(voltage_values),
                    'range': np.max(voltage_values) - np.min(voltage_values),
                    'q1': np.percentile(voltage_values, 25),
                    'q3': np.percentile(voltage_values, 75),
                    'iqr': np.percentile(voltage_values, 75) - np.percentile(voltage_values, 25),
                    'cv': (np.std(voltage_values) / np.mean(voltage_values) * 100) if np.mean(voltage_values) != 0 else 0
                }
                
                # Calculate slope and r-squared if enough points
                if len(label_data) > 1:
                    try:
                        result = linregress(range(len(voltage_values)), voltage_values)
                        metrics['slope'] = result.slope
                        metrics['abs_slope'] = abs(result.slope)
                        metrics['r_squared'] = result.rvalue ** 2
                    except:
                        metrics['slope'] = 0
                        metrics['abs_slope'] = 0
                        metrics['r_squared'] = 0
                else:
                    metrics['slope'] = 0
                    metrics['abs_slope'] = 0
                    metrics['r_squared'] = 0
                
                # Calculate skewness and kurtosis
                if len(voltage_values) > 3:
                    metrics['skewness'] = stats.skew(voltage_values)
                    metrics['kurtosis'] = stats.kurtosis(voltage_values)
                else:
                    metrics['skewness'] = 0
                    metrics['kurtosis'] = 0
                
                # Only flag anomalies for steady state
                if label == 'steady_state':
                    flags = []
                    reasons = []
                    
                    # CHECK 1: Voltage below 18V for steady state
                    if metrics['mean_voltage'] < self.operational_min:
                        flags.append('low_voltage')
                        reasons.append(f"Mean voltage {metrics['mean_voltage']:.2f}V < 18V")
                    
                    # CHECK 2: High variance
                    if metrics['variance'] > self.steady_state_thresholds['max_variance']:
                        flags.append('high_variance')
                        reasons.append(f"Variance {metrics['variance']:.3f} > {self.steady_state_thresholds['max_variance']}")
                    
                    # CHECK 3: High std
                    if metrics['std'] > self.steady_state_thresholds['max_std']:
                        flags.append('high_std')
                        reasons.append(f"Std {metrics['std']:.3f} > {self.steady_state_thresholds['max_std']}")
                    
                    # CHECK 4: Excessive slope
                    if metrics['abs_slope'] > self.steady_state_thresholds['max_slope']:
                        flags.append('excessive_slope')
                        reasons.append(f"Slope {metrics['abs_slope']:.4f} > {self.steady_state_thresholds['max_slope']}")
                    
                    # CHECK 5: Individual outlier points within steady state
                    if len(voltage_values) > 3:
                        z_scores = np.abs(zscore(voltage_values))
                        outlier_indices = np.where(z_scores > self.steady_state_thresholds['outlier_threshold'])[0]
                        n_outliers = len(outlier_indices)
                        metrics['n_outliers_zscore'] = n_outliers
                        metrics['max_zscore'] = np.max(z_scores)
                        metrics['outlier_indices'] = outlier_indices.tolist()
                        
                        if n_outliers > 0:
                            flags.append('outlier_points')
                            reasons.append(f"{n_outliers} points with z-score > {self.steady_state_thresholds['outlier_threshold']}")
                    else:
                        metrics['n_outliers_zscore'] = 0
                        metrics['max_zscore'] = 0
                        metrics['outlier_indices'] = []
                    
                    # IQR outliers
                    if metrics['iqr'] > 0:
                        lower_bound = metrics['q1'] - 1.5 * metrics['iqr']
                        upper_bound = metrics['q3'] + 1.5 * metrics['iqr']
                        n_outliers_iqr = np.sum((voltage_values < lower_bound) | (voltage_values > upper_bound))
                        metrics['n_outliers_iqr'] = n_outliers_iqr
                    else:
                        metrics['n_outliers_iqr'] = 0
                    
                    metrics['flagged'] = len(flags) > 0
                    metrics['flags'] = ', '.join(flags) if flags else ''
                    metrics['flag_reasons'] = '; '.join(reasons) if reasons else ''
                    
                else:
                    # Other labels are just tracked with statistics
                    metrics['flagged'] = False
                    metrics['flags'] = ''
                    metrics['flag_reasons'] = ''
                    metrics['n_outliers_zscore'] = 0
                    metrics['n_outliers_iqr'] = 0
                    metrics['max_zscore'] = 0
                    metrics['outlier_indices'] = []
                
                results.append(metrics)
            
            return results, df, grouping
            
        except Exception as e:
            self.failed_files.append((csv_path.name, str(e)))
            return None
    
    def create_simple_plot(self, df, grouping, output_path):
        """Create a detailed plot showing WHY something was flagged."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        colors = {
            'de-energized': 'gray',
            'stabilizing': 'orange',
            'steady_state': 'green',
            'unidentified': 'purple'
        }
        
        # Main voltage plot
        for label in df['label'].unique():
            label_data = df[df['label'] == label]
            ax1.scatter(label_data['timestamp'], label_data['voltage'],
                       color=colors.get(label, 'black'),
                       label=label, s=15, alpha=0.7, edgecolors='none')
        
        # Add reference lines
        ax1.axhline(y=self.operational_min, color='red', linestyle='--', alpha=0.5, 
                   label='18V threshold', linewidth=2)
        ax1.axhline(y=self.deenergized_max, color='gray', linestyle='--', alpha=0.3)
        
        # Mark specific outliers in steady state
        flagged_info = []
        if 'steady_state' in df['label'].values:
            ss_data = df[df['label'] == 'steady_state']
            ss_voltage = ss_data['voltage'].values
            
            # Get flag reasons from our results
            for result in self.results:
                if (result.get('label') == 'steady_state' and 
                    result.get('unit_id') == grouping.get('unit_id') and
                    result.get('test_run') == grouping.get('test_run')):
                    
                    if result.get('flagged'):
                        flagged_info.append(f"FLAGGED: {result.get('flag_reasons', 'Unknown reason')}")
                        
                        # Show outlier points if they exist
                        if len(ss_voltage) > 3:
                            z_scores = np.abs(zscore(ss_voltage))
                            outlier_mask = z_scores > self.steady_state_thresholds['outlier_threshold']
                            if np.any(outlier_mask):
                                outlier_data = ss_data[outlier_mask]
                                ax1.scatter(outlier_data['timestamp'], outlier_data['voltage'],
                                          color='red', s=100, marker='x', label='Outlier points', 
                                          zorder=5, linewidths=2)
                    
                    # Add statistics text
                    stats_text = (f"Mean: {result.get('mean_voltage', 0):.2f}V | "
                                f"Std: {result.get('std', 0):.3f} | "
                                f"Var: {result.get('variance', 0):.3f} | "
                                f"Slope: {result.get('abs_slope', 0):.4f}")
                    flagged_info.append(stats_text)
                    break
        
        # Title with all info
        title = (f"Unit: {grouping.get('unit_id', 'NA')} | "
                f"Test Case: {grouping.get('test_case', 'NA')} | "
                f"Run: {grouping.get('test_run', 'NA')} | "
                f"DC: {grouping.get('dc_folder', 'NA')}")
        
        if flagged_info:
            title += f"\n{' | '.join(flagged_info)}"
        
        ax1.set_title(title, fontsize=11, color='red' if flagged_info else 'black')
        ax1.set_xlabel('Timestamp', fontsize=10)
        ax1.set_ylabel('Voltage (V)', fontsize=10)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Segment visualization at bottom
        segment_colors = plt.cm.tab10(np.linspace(0, 1, len(df['segment'].unique())))
        for i, seg in enumerate(df['segment'].unique()):
            seg_data = df[df['segment'] == seg]
            ax2.scatter(seg_data['timestamp'], [seg]*len(seg_data), 
                       color=segment_colors[i], s=10, alpha=0.7, label=f'Seg {seg}')
        
        ax2.set_xlabel('Timestamp', fontsize=10)
        ax2.set_ylabel('Segment ID', fontsize=10)
        ax2.set_title('Original Segment Clustering', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def create_summary_plot(self, all_results_df):
        """Create a single summary plot showing all flagged items."""
        if all_results_df.empty or 'flagged' not in all_results_df.columns:
            print("No data to create summary plot")
            return
            
        if not any(all_results_df['flagged']):
            print("No flagged items to plot")
            return
        
        flagged = all_results_df[all_results_df['flagged']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Flagged items by test case
        ax = axes[0, 0]
        if 'test_case' in flagged.columns:
            test_cases = flagged['test_case'].value_counts()
            ax.bar(range(len(test_cases)), test_cases.values, color='coral')
            ax.set_xticks(range(len(test_cases)))
            ax.set_xticklabels(test_cases.index, rotation=45, ha='right', fontsize=8)
            ax.set_title('Flagged Items by Test Case')
            ax.set_ylabel('Count')
        
        # Plot 2: Flagged items by label type
        ax = axes[0, 1]
        label_counts = flagged['label'].value_counts()
        colors_map = {'steady_state': 'green', 'voltage_outlier': 'red', 'unidentified': 'purple'}
        colors = [colors_map.get(x, 'gray') for x in label_counts.index]
        ax.bar(range(len(label_counts)), label_counts.values, color=colors)
        ax.set_xticks(range(len(label_counts)))
        ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
        ax.set_title('Flagged Items by Label Type')
        ax.set_ylabel('Count')
        
        # Plot 3: Voltage distribution of flagged steady states
        ax = axes[1, 0]
        ss_flagged = flagged[flagged['label'] == 'steady_state']
        if not ss_flagged.empty:
            ax.hist(ss_flagged['mean_voltage'], bins=20, edgecolor='black', color='green', alpha=0.7)
            ax.axvline(x=self.operational_min, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.operational_max, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Voltage Distribution - Flagged Steady States')
            ax.set_xlabel('Mean Voltage (V)')
            ax.set_ylabel('Count')
        
        # Plot 4: Variance distribution
        ax = axes[1, 1]
        if 'variance' in flagged.columns:
            ax.hist(flagged['variance'], bins=20, edgecolor='black', color='blue', alpha=0.7)
            ax.axvline(x=self.steady_state_thresholds['max_variance'], 
                      color='red', linestyle='--', alpha=0.5, 
                      label=f"Threshold: {self.steady_state_thresholds['max_variance']}")
            ax.set_title('Variance Distribution - Flagged Items')
            ax.set_xlabel('Variance')
            ax.set_ylabel('Count')
            ax.legend()
        
        plt.suptitle('Summary of Flagged Anomalies', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_folder / 'summary_flagged.png'
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"Summary plot saved to: {output_path}")
    
    def run_analysis(self):
        """Run analysis on all CSV files."""
        print("="*60)
        print("VOLTAGE ANALYSIS WITH CLUSTER CLASSIFICATION")
        print("="*60)
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        
        # Collect all CSV files
        csv_files = []
        for test_case_folder in self.input_folder.iterdir():
            if test_case_folder.is_dir():
                for dc_folder in ['dc1', 'dc2']:
                    dc_path = test_case_folder / dc_folder
                    if dc_path.exists():
                        found_csvs = list(dc_path.glob('*_segments.csv'))
                        if found_csvs:
                            csv_files.extend(found_csvs)
                            print(f"  Found {len(found_csvs)} files in {test_case_folder.name}/{dc_folder}")
        
        if not csv_files:
            print("No CSV files found!")
            return pd.DataFrame()
        
        print(f"\nTotal CSV files to process: {len(csv_files)}")
        
        # Process each CSV
        all_results = []
        flagged_files = []
        
        for csv_path in tqdm(csv_files, desc="Processing"):
            result = self.analyze_csv(csv_path)
            if result:
                file_results, df, grouping = result
                all_results.extend(file_results)
                
                # Check if any segment is flagged
                if any(r.get('flagged', False) for r in file_results):
                    flagged_files.append((df, grouping, csv_path))
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        
        # Create output folders - SIMPLIFIED STRUCTURE
        plots_folder = self.output_folder / 'flagged_plots'
        plots_folder.mkdir(exist_ok=True)
        
        # Generate plots only for flagged files
        if flagged_files:
            print(f"\nGenerating plots for {len(flagged_files)} flagged files...")
            for df, grouping, csv_path in tqdm(flagged_files, desc="Creating plots"):
                # Simple filename including DC folder
                plot_name = (f"{grouping.get('unit_id', 'NA')}_"
                           f"{grouping.get('test_case', 'NA')}_"
                           f"run{grouping.get('test_run', 'NA')}_"
                           f"{grouping.get('dc_folder', 'NA')}.png")
                plot_path = plots_folder / plot_name
                self.create_simple_plot(df, grouping, plot_path)
        else:
            print("\nNo flagged files found - no plots to generate")
        
        # Save Excel report
        if not results_df.empty:
            excel_path = self.output_folder / 'analysis_results.xlsx'
            with pd.ExcelWriter(excel_path) as writer:
                # All results
                results_df.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Only flagged
                if 'flagged' in results_df.columns:
                    flagged_df = results_df[results_df['flagged'] == True]
                    if not flagged_df.empty:
                        flagged_df.to_excel(writer, sheet_name='Flagged_Only', index=False)
                        
                        # Summary statistics by test case and label
                        summary = results_df.groupby(['test_case', 'label']).agg({
                            'mean_voltage': ['mean', 'std', 'min', 'max'],
                            'variance': ['mean', 'max'],
                            'n_points': 'sum',
                            'flagged': 'sum'
                        }).round(3)
                        summary.to_excel(writer, sheet_name='Summary_Stats')
                        
                        # DC comparison
                        if 'dc_folder' in results_df.columns:
                            dc_summary = results_df.groupby(['dc_folder', 'label']).agg({
                                'mean_voltage': ['mean', 'std'],
                                'flagged': 'sum',
                                'n_points': 'sum'
                            }).round(3)
                            dc_summary.to_excel(writer, sheet_name='DC_Comparison')
            
            print(f"\nExcel report saved to: {excel_path}")
            
            # Create summary plot
            self.create_summary_plot(results_df)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total segments analyzed: {len(results_df)}")
        
        if 'flagged' in results_df.columns:
            n_flagged = results_df['flagged'].sum()
            print(f"Flagged segments: {n_flagged}")
            
            if n_flagged > 0:
                print("\nFlagged breakdown by label:")
                for label in results_df[results_df['flagged']]['label'].unique():
                    count = len(results_df[(results_df['flagged']) & (results_df['label'] == label)])
                    print(f"  {label}: {count}")
                
                print("\nFlagged breakdown by test case:")
                if 'test_case' in results_df.columns:
                    for tc in results_df[results_df['flagged']]['test_case'].unique():
                        count = len(results_df[(results_df['flagged']) & (results_df['test_case'] == tc)])
                        print(f"  {tc}: {count}")
        
        if self.failed_files:
            print(f"\n{len(self.failed_files)} files failed to process")
            for filename, error in self.failed_files[:5]:
                print(f"  {filename}: {error}")
        
        print(f"\nOutputs:")
        print(f"  Excel: {self.output_folder}/analysis_results.xlsx")
        print(f"  Plots: {self.output_folder}/flagged_plots/")
        print(f"  Summary: {self.output_folder}/summary_flagged.png")
        
        return results_df


def main():
    """Main function to run simplified analysis."""
    analyzer = SimplifiedVoltageAnalyzer(
        input_folder='greedygaussv4',
        output_folder='voltage_analysis'
    )
    
    results = analyzer.run_analysis()
    
    return results


if __name__ == "__main__":
    main()
