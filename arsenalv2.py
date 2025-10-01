import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import zscore
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VoltageSegmentAnalyzer:
    """Analyze voltage segments and flag anomalies using dynamic thresholds."""
    
    def __init__(self, dc_folder, voltage_range=(20, 28), 
                 zscore_threshold=3, iqr_multiplier=1.5, mad_threshold=3.5,
                 percentile_steady=95, percentile_stabilizing=97, percentile_deenergized=95,
                 cache_thresholds=True, output_base_dir=None):
        """
        Initialize the analyzer with configurable parameters.
        
        Args:
            dc_folder: Path to DC folder containing CSV files
            voltage_range: Tuple of (min, max) expected voltage
            zscore_threshold: Threshold for z-score outlier detection
            iqr_multiplier: Multiplier for IQR outlier detection
            mad_threshold: Threshold for MAD-based outlier detection
            percentile_steady: Percentile threshold for steady state (default 95)
            percentile_stabilizing: Percentile threshold for stabilizing (default 97)
            percentile_deenergized: Percentile threshold for de-energized (default 95)
            cache_thresholds: Whether to cache calculated thresholds to disk
            output_base_dir: Base directory for all outputs (if None, uses current directory)
        """
        self.dc_folder = Path(dc_folder)
        self.voltage_min, self.voltage_max = voltage_range
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.mad_threshold = mad_threshold
        self.cache_thresholds = cache_thresholds
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path('analysis_output')
        self.failed_files = []
        self.dynamic_thresholds = {}
        
        # Percentile levels for dynamic thresholds
        self.percentile_levels = {
            'steady_state': {
                'variance': percentile_steady,
                'abs_slope': percentile_steady,
                'cv': percentile_steady,
                'n_outliers_zscore': percentile_steady,
                'max_zscore': percentile_steady,
                'iqr': percentile_steady,
                'skewness': percentile_steady,
                'max_rolling_std': percentile_steady
            },
            'stabilizing': {
                'variance': percentile_stabilizing,
                'abs_slope': percentile_stabilizing,
                'cv': percentile_stabilizing,
                'n_outliers_zscore': percentile_stabilizing,
                'max_zscore': percentile_stabilizing,
                'iqr': percentile_stabilizing,
                'max_rolling_std': percentile_stabilizing
            },
            'de-energized': {
                'variance': percentile_deenergized,
                'max_zscore': percentile_deenergized,
                'mean_voltage': percentile_deenergized
            }
        }
        
        # Anomaly checkers for each label type
        self.anomaly_checkers = {
            'steady_state': self._check_steady_state_anomalies,
            'stabilizing': self._check_stabilizing_anomalies,
            'de-energized': self._check_de_energized_anomalies
        }
    
    def is_deenergized(self, voltage, segment, timestamp):
        """
        Your function to identify de-energized segments.
        Returns a boolean mask array.
        """
        # PLACEHOLDER: Replace with your actual is_deenergized function
        # This should be your actual implementation that returns a mask
        deenergized_mask = voltage < 3  # Simple placeholder
        return deenergized_mask
    
    def is_stabilizing(self, voltage, segment, timestamp):
        """
        Your function to identify stabilizing segments.
        Returns a boolean mask array.
        """
        # PLACEHOLDER: Replace with your actual is_stabilizing function
        # This should be your actual implementation that returns a mask
        stabilizing_mask = (voltage > 3) & (voltage < 20)  # Simple placeholder
        return stabilizing_mask
    
    def is_steadystate(self, voltage, segment, timestamp):
        """
        Your function to identify steady state segments.
        Returns a boolean mask array.
        """
        # PLACEHOLDER: Replace with your actual is_steadystate function
        # This should be your actual implementation that returns a mask
        steadystate_mask = voltage >= 20  # Simple placeholder
        return steadystate_mask
    
    def apply_status_labels(self, df):
        """
        Apply status labels using mask functions.
        """
        # Get masks from your functions
        deenergized_mask = self.is_deenergized(
            df['voltage'].to_numpy(),
            df['segment'].to_numpy(),
            df['timestamp'].to_numpy()
        )
        
        stabilizing_mask = self.is_stabilizing(
            df['voltage'].to_numpy(),
            df['segment'].to_numpy(),
            df['timestamp'].to_numpy()
        )
        
        steadystate_mask = self.is_steadystate(
            df['voltage'].to_numpy(),
            df['segment'].to_numpy(),
            df['timestamp'].to_numpy()
        )
        
        # Apply labels based on masks
        df['predicted_status'] = "unidentified"
        df.loc[deenergized_mask, 'predicted_status'] = "de-energized"
        df.loc[stabilizing_mask, 'predicted_status'] = "stabilizing"
        df.loc[steadystate_mask, 'predicted_status'] = "steady_state"
        
        return df
    
    def parse_filename(self, filename):
        """Extract grouping information from filename."""
        # Remove .csv extension and _segments suffix
        base = filename.replace('.csv', '').replace('_segments', '')
        
        # Parse components more carefully to handle hyphens in values
        parts = {}
        
        # Find all positions where we have 'key=' pattern
        import re
        
        # Find all key= patterns
        pattern = r'(\w+)='
        matches = list(re.finditer(pattern, base))
        
        if not matches:
            print(f"Warning: No key=value pairs found in filename: {filename}")
            return parts
        
        # Extract key-value pairs
        for i, match in enumerate(matches):
            key = match.group(1)
            start = match.end()  # Position after the '='
            
            # Find where this value ends (at the next key= or end of string)
            if i + 1 < len(matches):
                # Value ends at the underscore before the next key
                next_match = matches[i + 1]
                # Find the underscore right before the next key
                end = base.rfind('_', start, next_match.start())
                if end == -1:  # No underscore found, shouldn't happen
                    end = next_match.start()
            else:
                # This is the last key-value pair
                end = len(base)
            
            value = base[start:end]
            parts[key] = value
        
        return parts
    
    def calculate_metrics(self, voltage_data):
        """Calculate comprehensive statistical metrics for voltage data."""
        metrics = {
            'mean': np.mean(voltage_data),
            'std': np.std(voltage_data),
            'variance': np.var(voltage_data),
            'cv': (np.std(voltage_data) / np.mean(voltage_data) * 100) if np.mean(voltage_data) != 0 else 0,
            'min': np.min(voltage_data),
            'max': np.max(voltage_data),
            'range': np.max(voltage_data) - np.min(voltage_data),
            'median': np.median(voltage_data),
            'q1': np.percentile(voltage_data, 25),
            'q3': np.percentile(voltage_data, 75),
            'iqr': np.percentile(voltage_data, 75) - np.percentile(voltage_data, 25)
        }
        
        # Calculate slope using linear regression
        if len(voltage_data) > 1:
            x = np.arange(len(voltage_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, voltage_data)
            metrics['slope'] = slope
            metrics['abs_slope'] = abs(slope)
            metrics['r_squared'] = r_value ** 2
            metrics['slope_pvalue'] = p_value
        else:
            metrics['slope'] = 0
            metrics['abs_slope'] = 0
            metrics['r_squared'] = 0
            metrics['slope_pvalue'] = 1
            
        # Z-score analysis
        z_scores = zscore(voltage_data)
        metrics['max_zscore'] = np.max(np.abs(z_scores))
        metrics['mean_zscore'] = np.mean(np.abs(z_scores))
        metrics['n_outliers_zscore'] = np.sum(np.abs(z_scores) > self.zscore_threshold)
        metrics['outlier_indices'] = np.where(np.abs(z_scores) > self.zscore_threshold)[0]
        
        # IQR-based outlier detection
        Q1 = metrics['q1']
        Q3 = metrics['q3']
        IQR = metrics['iqr']
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        metrics['n_outliers_iqr'] = np.sum((voltage_data < lower_bound) | (voltage_data > upper_bound))
        
        # Modified Z-score (MAD-based)
        median = metrics['median']
        mad = np.median(np.abs(voltage_data - median))
        if mad == 0:
            mad = 1.4826 * np.std(voltage_data)
        modified_z_scores = 0.6745 * (voltage_data - median) / mad if mad != 0 else np.zeros_like(voltage_data)
        metrics['n_outliers_mad'] = np.sum(np.abs(modified_z_scores) > self.mad_threshold)
        metrics['max_modified_zscore'] = np.max(np.abs(modified_z_scores))
        
        # Skewness and Kurtosis for distribution shape
        metrics['skewness'] = stats.skew(voltage_data)
        metrics['kurtosis'] = stats.kurtosis(voltage_data)
        
        # Rolling statistics (window = 10% of data length, min 5)
        window_size = max(5, len(voltage_data) // 10)
        voltage_series = pd.Series(voltage_data)
        rolling_std = voltage_series.rolling(window=window_size).std()
        metrics['max_rolling_std'] = np.nanmax(rolling_std)
        metrics['mean_rolling_std'] = np.nanmean(rolling_std)
        
        return metrics
    
    def _process_csv(self, csv_path, callback):
        """Common CSV processing logic for both passes."""
        try:
            filename = csv_path.name
            grouping = self.parse_filename(filename)
            
            # Debug: Print what we parsed
            if not grouping:
                print(f"Warning: Could not parse filename: {filename}")
                self.failed_files.append((str(csv_path), "Failed to parse filename"))
                return [], None, None
            
            df = pd.read_csv(csv_path)
            
            # Check if required columns exist
            required_cols = ['segment', 'voltage', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
                print(f"Error in {filename}: {error_msg}")
                self.failed_files.append((str(csv_path), error_msg))
                return [], None, None
            
            # Check if dataframe is empty
            if df.empty:
                self.failed_files.append((str(csv_path), "Empty dataframe"))
                return [], None, None
            
            # Apply status labels using mask functions
            df = self.apply_status_labels(df)
            
            results = []
            for label in df['predicted_status'].unique():
                if label == 'unidentified':
                    continue
                
                label_data = df[df['predicted_status'] == label]
                voltage_data = label_data['voltage'].values
                
                if len(voltage_data) == 0:
                    continue
                
                result = callback(label, voltage_data, grouping, filename)
                if result:
                    results.append(result)
            
            return results, df, grouping
            
        except Exception as e:
            print(f"Error processing {csv_path.name}: {str(e)}")
            self.failed_files.append((str(csv_path), str(e)))
            return [], None, None
    
    def process_csv_first_pass(self, csv_path):
        """First pass: collect metrics for threshold calculation."""
        def callback(label, voltage_data, grouping, filename):
            metrics = self.calculate_metrics(voltage_data)
            metrics['test_case'] = grouping.get('test_case', 'unknown')
            metrics['label'] = label
            metrics['filename'] = filename
            return metrics
        
        metrics_list, _, _ = self._process_csv(csv_path, callback)
        return metrics_list
    
    def process_csv_second_pass(self, csv_path):
        """Second pass: analyze with dynamic thresholds and flag anomalies."""
        def callback(label, voltage_data, grouping, filename):
            # Calculate metrics
            metrics = self.calculate_metrics(voltage_data)
            
            # Check for anomalies using dynamic thresholds
            test_case = grouping.get('test_case', 'unknown')
            flags, reasons = self.check_anomalies_dynamic(test_case, label, metrics)
            
            # Store results
            result = {
                **grouping,
                'label': label,
                'n_points': len(voltage_data),
                'mean_voltage': metrics['mean'],
                'std': metrics['std'],
                'variance': metrics['variance'],
                'cv': metrics['cv'],
                'slope': metrics['slope'],
                'abs_slope': metrics['abs_slope'],
                'r_squared': metrics['r_squared'],
                'min_voltage': metrics['min'],
                'max_voltage': metrics['max'],
                'range': metrics['range'],
                'median': metrics['median'],
                'iqr': metrics['iqr'],
                'skewness': metrics['skewness'],
                'kurtosis': metrics['kurtosis'],
                'n_outliers_zscore': metrics['n_outliers_zscore'],
                'n_outliers_iqr': metrics['n_outliers_iqr'],
                'n_outliers_mad': metrics['n_outliers_mad'],
                'max_zscore': metrics['max_zscore'],
                'mean_zscore': metrics['mean_zscore'],
                'max_rolling_std': metrics['max_rolling_std'],
                'flagged': len(flags) > 0,
                'flags': ', '.join(flags) if flags else '',
                'reasons': '; '.join(reasons) if reasons else '',
                'csv_file': filename
            }
            
            return result
        
        return self._process_csv(csv_path, callback)
    
    def calculate_dynamic_thresholds(self, all_metrics_df):
        """Calculate dynamic thresholds based on percentiles within each test_case and label."""
        thresholds = {}
        
        # Group by test_case and label
        for (test_case, label), group_df in all_metrics_df.groupby(['test_case', 'label']):
            if label == 'unknown':
                continue
                
            key = f"{test_case}_{label}"
            thresholds[key] = {}
            
            # Get percentile levels for this label type
            if label in self.percentile_levels:
                percentile_config = self.percentile_levels[label]
                
                for metric, percentile in percentile_config.items():
                    if metric in group_df.columns:
                        valid_data = group_df[metric].dropna()
                        if not valid_data.empty:
                            threshold_value = np.percentile(valid_data, percentile)
                            thresholds[key][metric] = threshold_value
            
            # Special handling for de-energized expected voltage
            if label == 'de-energized':
                thresholds[key]['max_mean'] = 2.0
        
        return thresholds
    
    def check_anomalies_dynamic(self, test_case, label, metrics):
        """Check for anomalies using dynamic thresholds."""
        threshold_key = f"{test_case}_{label}"
        if threshold_key not in self.dynamic_thresholds:
            return [], []
        
        if label in self.anomaly_checkers:
            return self.anomaly_checkers[label](test_case, metrics)
        
        return [], []
    
    def _check_steady_state_anomalies(self, test_case, metrics):
        """Check anomalies for steady state segments."""
        flags = []
        reasons = []
        threshold_key = f"{test_case}_steady_state"
        thresholds = self.dynamic_thresholds.get(threshold_key, {})
        
        if 'variance' in thresholds and metrics['variance'] > thresholds['variance']:
            flags.append('high_variance')
            reasons.append(f"Variance {metrics['variance']:.3f} > {thresholds['variance']:.3f} (95th percentile)")
        
        if 'abs_slope' in thresholds and metrics['abs_slope'] > thresholds['abs_slope']:
            flags.append('excessive_slope')
            reasons.append(f"Slope {metrics['abs_slope']:.3f} > {thresholds['abs_slope']:.3f} (95th percentile)")
        
        if 'cv' in thresholds and metrics['cv'] > thresholds['cv']:
            flags.append('high_cv')
            reasons.append(f"CV {metrics['cv']:.1f}% > {thresholds['cv']:.1f}% (95th percentile)")
        
        if 'n_outliers_zscore' in thresholds and metrics['n_outliers_zscore'] > thresholds['n_outliers_zscore']:
            flags.append('excessive_outliers')
            reasons.append(f"{metrics['n_outliers_zscore']} z-score outliers > {thresholds['n_outliers_zscore']:.0f} (95th percentile)")
        
        if 'max_zscore' in thresholds and metrics['max_zscore'] > thresholds['max_zscore']:
            flags.append('extreme_zscore')
            reasons.append(f"Max z-score {metrics['max_zscore']:.2f} > {thresholds['max_zscore']:.2f} (95th percentile)")
        
        if 'iqr' in thresholds and metrics['iqr'] > thresholds['iqr']:
            flags.append('high_iqr')
            reasons.append(f"IQR {metrics['iqr']:.3f} > {thresholds['iqr']:.3f} (95th percentile)")
        
        if 'skewness' in thresholds and abs(metrics['skewness']) > abs(thresholds['skewness']):
            flags.append('high_skewness')
            reasons.append(f"Skewness {abs(metrics['skewness']):.2f} > {abs(thresholds['skewness']):.2f} (95th percentile)")
        
        # Check absolute voltage range
        if metrics['mean'] < self.voltage_min or metrics['mean'] > self.voltage_max:
            flags.append('voltage_out_of_range')
            reasons.append(f"Mean voltage {metrics['mean']:.1f}V outside {self.voltage_min}-{self.voltage_max}V")
        
        return flags, reasons
    
    def _check_stabilizing_anomalies(self, test_case, metrics):
        """Check anomalies for stabilizing segments."""
        flags = []
        reasons = []
        threshold_key = f"{test_case}_stabilizing"
        thresholds = self.dynamic_thresholds.get(threshold_key, {})
        
        if 'variance' in thresholds and metrics['variance'] > thresholds['variance']:
            flags.append('excessive_variance')
            reasons.append(f"Variance {metrics['variance']:.3f} > {thresholds['variance']:.3f} (97th percentile)")
        
        if 'abs_slope' in thresholds and metrics['abs_slope'] > thresholds['abs_slope']:
            flags.append('excessive_slope')
            reasons.append(f"Slope {metrics['abs_slope']:.3f} > {thresholds['abs_slope']:.3f} (97th percentile)")
        
        if 'n_outliers_zscore' in thresholds and metrics['n_outliers_zscore'] > thresholds['n_outliers_zscore']:
            flags.append('excessive_outliers')
            reasons.append(f"{metrics['n_outliers_zscore']} outliers > {thresholds['n_outliers_zscore']:.0f} (97th percentile)")
        
        return flags, reasons
    
    def _check_de_energized_anomalies(self, test_case, metrics):
        """Check anomalies for de-energized segments."""
        flags = []
        reasons = []
        threshold_key = f"{test_case}_de-energized"
        thresholds = self.dynamic_thresholds.get(threshold_key, {})
        
        if 'max_mean' in thresholds and metrics['mean'] > thresholds['max_mean']:
            flags.append('voltage_too_high')
            reasons.append(f"Mean voltage {metrics['mean']:.1f}V > {thresholds['max_mean']}V for de-energized")
        
        if 'variance' in thresholds and metrics['variance'] > thresholds['variance']:
            flags.append('high_variance')
            reasons.append(f"Variance {metrics['variance']:.3f} > {thresholds['variance']:.3f} (95th percentile)")
        
        return flags, reasons
    
    def load_or_calculate_thresholds(self, csv_files):
        """Load cached thresholds or calculate new ones."""
        cache_file = self.dc_folder / 'thresholds_cache.pkl'
        
        if self.cache_thresholds and cache_file.exists():
            try:
                print("Loading cached thresholds...")
                with open(cache_file, 'rb') as f:
                    self.dynamic_thresholds = pickle.load(f)
                print(f"Loaded {len(self.dynamic_thresholds)} cached thresholds")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recalculating...")
        
        # Calculate thresholds
        print("\nFirst pass: Collecting metrics for dynamic thresholds...")
        all_metrics = []
        
        for csv_path in tqdm(csv_files, desc="Collecting metrics"):
            metrics_list = self.process_csv_first_pass(csv_path)
            all_metrics.extend(metrics_list)
        
        # Calculate dynamic thresholds
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            self.dynamic_thresholds = self.calculate_dynamic_thresholds(metrics_df)
            print(f"Calculated dynamic thresholds for {len(self.dynamic_thresholds)} test_case/label combinations")
            
            # Cache thresholds
            if self.cache_thresholds:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.dynamic_thresholds, f)
                    print(f"Cached thresholds to {cache_file}")
                except Exception as e:
                    print(f"Failed to cache thresholds: {e}")
    
    def create_plot(self, df, grouping, output_path):
        """Create plot showing all labels with different colors."""
        # Apply status labels if not already present
        if 'predicted_status' not in df.columns:
            df = self.apply_status_labels(df)
        
        color_map = {
            'steady_state': 'blue',
            'stabilizing': 'orange',
            'de-energized': 'green',
            'unidentified': 'gray'
        }
        
        fig = go.Figure()
        
        # Plot each label with different color
        for label in df['predicted_status'].unique():
            label_data = df[df['predicted_status'] == label]
            
            fig.add_trace(go.Scatter(
                x=label_data['timestamp'],
                y=label_data['voltage'],
                mode='lines+markers',
                name=label,
                marker=dict(color=color_map.get(label, 'gray'), size=4),
                line=dict(color=color_map.get(label, 'gray'))
            ))
        
        # Detect and highlight outliers
        for label in df['predicted_status'].unique():
            if label == 'unidentified':
                continue
                
            label_data = df[df['predicted_status'] == label]
            voltage_data = label_data['voltage'].values
            
            if len(voltage_data) > 0:
                # Find outliers
                z_scores = np.abs(zscore(voltage_data))
                outlier_mask = z_scores > self.zscore_threshold
                
                if np.any(outlier_mask):
                    outlier_data = label_data[outlier_mask]
                    
                    fig.add_trace(go.Scatter(
                        x=outlier_data['timestamp'],
                        y=outlier_data['voltage'],
                        mode='markers',
                        name=f'{label} outliers',
                        marker=dict(color='red', size=10, symbol='x'),
                        showlegend=True
                    ))
        
        # Add horizontal lines for voltage range
        fig.add_hline(y=self.voltage_min, line_dash="dash", line_color="gray", 
                     annotation_text=f"Min: {self.voltage_min}V")
        fig.add_hline(y=self.voltage_max, line_dash="dash", line_color="gray",
                     annotation_text=f"Max: {self.voltage_max}V")
        
        # Update layout
        title = f"Unit: {grouping.get('unit_id', 'N/A')} | " \
               f"Test: {grouping.get('test_case', 'N/A')} | " \
               f"Run: {grouping.get('test_run', 'N/A')}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Timestamp",
            yaxis_title="Voltage (V)",
            height=600,
            width=1200,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        # Save as PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path))
        
        return fig
    
    def create_threshold_overview(self, summary_df, threshold_df, output_path):
        """Create comprehensive overview plot showing all test cases and thresholds."""
        from plotly.subplots import make_subplots
        
        # Get unique test cases and labels
        test_cases = sorted(summary_df['test_case'].unique())
        labels = ['steady_state', 'stabilizing', 'de-energized']
        
        # Create subplots - one row per metric
        metrics_to_plot = ['variance', 'abs_slope', 'cv', 'max_zscore']
        fig = make_subplots(
            rows=len(metrics_to_plot), 
            cols=1,
            subplot_titles=[f'{metric.replace("_", " ").title()} Analysis' for metric in metrics_to_plot],
            vertical_spacing=0.08,
            specs=[[{"secondary_y": True}] for _ in metrics_to_plot]
        )
        
        colors = {
            'steady_state': 'blue',
            'stabilizing': 'orange', 
            'de-energized': 'green'
        }
        
        # For each metric, create grouped box plots with threshold lines
        for idx, metric in enumerate(metrics_to_plot, 1):
            for label in labels:
                label_data = summary_df[summary_df['label'] == label]
                
                if metric in label_data.columns:
                    # Add box plot for each test case
                    for test_case in test_cases:
                        test_data = label_data[label_data['test_case'] == test_case]
                        if not test_data.empty and metric in test_data.columns:
                            fig.add_trace(
                                go.Box(
                                    y=test_data[metric],
                                    name=f"{test_case}_{label}",
                                    marker_color=colors[label],
                                    boxmean='sd',
                                    opacity=0.7,
                                    legendgroup=label,
                                    showlegend=(idx == 1)
                                ),
                                row=idx, col=1
                            )
                            
                            # Add threshold line if it exists
                            threshold_key = f"{test_case}_{label}"
                            if not threshold_df.empty:
                                thresh_data = threshold_df[
                                    (threshold_df['test_case'] == test_case) & 
                                    (threshold_df['label'] == label) & 
                                    (threshold_df['metric'] == metric)
                                ]
                                if not thresh_data.empty:
                                    threshold_val = thresh_data['threshold'].iloc[0]
                                    fig.add_hline(
                                        y=threshold_val,
                                        line_dash="dash",
                                        line_color=colors[label],
                                        opacity=0.5,
                                        annotation_text=f"{test_case} {label} threshold",
                                        annotation_position="right",
                                        row=idx, col=1
                                    )
            
            # Add flagged points as scatter
            flagged_data = summary_df[summary_df['flagged'] == True]
            for label in labels:
                label_flagged = flagged_data[flagged_data['label'] == label]
                if not label_flagged.empty and metric in label_flagged.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=[f"{row['test_case']}_{row['label']}" for _, row in label_flagged.iterrows()],
                            y=label_flagged[metric],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='x'),
                            name=f"Flagged {label}",
                            legendgroup=f"flagged_{label}",
                            showlegend=(idx == 1)
                        ),
                        row=idx, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=300 * len(metrics_to_plot),
            title_text=f"Comprehensive Threshold Analysis - {self.dc_folder.name}",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axes labels
        for idx, metric in enumerate(metrics_to_plot, 1):
            fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=idx, col=1)
        
        # Save as HTML for interactivity
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved threshold overview to: {output_path}")
        
        return fig
    
    def create_distribution_plots(self, summary_df, threshold_df, output_path):
        """Create distribution plots showing where thresholds fall for each metric."""
        from plotly.subplots import make_subplots
        
        # Metrics to visualize
        metrics = ['variance', 'abs_slope', 'cv', 'max_zscore', 'n_outliers_zscore', 'iqr']
        
        # Create subplot for each label type
        fig = make_subplots(
            rows=3, cols=len(metrics)//2 + len(metrics)%2,
            subplot_titles=metrics,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = {
            'steady_state': 'blue',
            'stabilizing': 'orange',
            'de-energized': 'green'
        }
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            for label in ['steady_state', 'stabilizing', 'de-energized']:
                label_data = summary_df[summary_df['label'] == label]
                
                if metric in label_data.columns:
                    values = label_data[metric].dropna()
                    
                    if len(values) > 0:
                        # Add histogram
                        fig.add_trace(
                            go.Histogram(
                                x=values,
                                name=label,
                                marker_color=colors[label],
                                opacity=0.5,
                                nbinsx=30,
                                legendgroup=label,
                                showlegend=(i == 0)
                            ),
                            row=row, col=col
                        )
                        
                        # Add threshold lines for each test case
                        if not threshold_df.empty:
                            for test_case in summary_df['test_case'].unique():
                                thresh_data = threshold_df[
                                    (threshold_df['test_case'] == test_case) & 
                                    (threshold_df['label'] == label) & 
                                    (threshold_df['metric'] == metric)
                                ]
                                if not thresh_data.empty:
                                    threshold_val = thresh_data['threshold'].iloc[0]
                                    fig.add_vline(
                                        x=threshold_val,
                                        line_dash="dash",
                                        line_color=colors[label],
                                        opacity=0.7,
                                        annotation_text=f"{test_case}",
                                        row=row, col=col
                                    )
        
        fig.update_layout(
            height=900,
            title_text=f"Metric Distributions with Dynamic Thresholds - {self.dc_folder.name}",
            showlegend=True,
            barmode='overlay'
        )
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved distribution plots to: {output_path}")
        
        return fig
    
    def run_analysis(self, return_only=False, test_case_name=None):
        """Run two-pass analysis with dynamic threshold calculation."""
        # Find all segment CSV files
        csv_files = list(self.dc_folder.glob('*_segments.csv'))
        
        if not csv_files:
            print(f"No segment CSV files found in {self.dc_folder}")
            return pd.DataFrame() if return_only else None
        
        print(f"Found {len(csv_files)} CSV files to process")
        
        # Load or calculate thresholds
        self.load_or_calculate_thresholds(csv_files)
        
        # SECOND PASS: Analyze with dynamic thresholds
        print("\nSecond pass: Analyzing with dynamic thresholds...")
        all_results = []
        flagged_files = []
        
        for csv_path in tqdm(csv_files, desc="Analyzing files"):
            file_results, df, grouping = self.process_csv_second_pass(csv_path)
            
            if file_results:
                # Add DC folder to results
                for result in file_results:
                    result['dc_folder'] = self.dc_folder.name
                
                all_results.extend(file_results)
                
                # Check if any labels were flagged
                if any(r['flagged'] for r in file_results):
                    flagged_files.append((df, grouping))
        
        # If return_only, just return the DataFrame without saving
        if return_only:
            return pd.DataFrame(all_results), flagged_files
        
        # Create results directory structure in output folder
        dc_name = self.dc_folder.name
        
        # Build output path based on test case if provided
        if test_case_name:
            results_dir = self.output_base_dir / test_case_name / dc_name / 'results'
            plots_dir = results_dir / 'plots'
        else:
            results_dir = self.output_base_dir / dc_name / 'results'
            plots_dir = results_dir / 'plots'
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(all_results)
        
        # Save threshold report
        threshold_df = []
        for key, thresholds in self.dynamic_thresholds.items():
            test_case, label = key.rsplit('_', 1)
            for metric, value in thresholds.items():
                threshold_df.append({
                    'test_case': test_case,
                    'label': label,
                    'metric': metric,
                    'threshold': value
                })
        threshold_df = pd.DataFrame(threshold_df)
        
        # Create comprehensive overview plots
        if not summary_df.empty:
            print("\nGenerating comprehensive overview plots...")
            overview_path = results_dir / f'{dc_name}_threshold_overview.html'
            self.create_threshold_overview(summary_df, threshold_df, overview_path)
            
            distribution_path = results_dir / f'{dc_name}_distributions.html'
            self.create_distribution_plots(summary_df, threshold_df, distribution_path)
        
        # Generate individual plots for flagged files (optional - can be disabled)
        if flagged_files:
            print(f"\nGenerating individual plots for {len(flagged_files)} flagged groupings...")
            for df, grouping in tqdm(flagged_files, desc="Creating plots"):
                if df is None:
                    continue
                    
                # Create unit_id subfolder, then test_case subfolder
                unit_id = grouping.get('unit_id', 'unknown')
                test_case = grouping.get('test_case', 'unknown')
                test_folder = plots_dir / f"unit_id={unit_id}" / f"test_case={test_case}"
                
                # Create plot filename
                plot_name = f"save={grouping.get('save', 'NA')}_" \
                           f"ofp={grouping.get('ofp', 'NA')}_" \
                           f"station={grouping.get('station', 'NA')}_" \
                           f"run={grouping.get('test_run', 'NA')}.png"
                
                plot_path = test_folder / plot_name
                
                try:
                    self.create_plot(df, grouping, plot_path)
                except Exception as e:
                    print(f"  Error creating plot for {plot_name}: {e}")
        
        # Print failed files if any
        if self.failed_files:
            print(f"\n{len(self.failed_files)} files failed to process:")
            for filepath, error in self.failed_files:
                print(f"  - {Path(filepath).name}: {error}")
        
        print(f"\nAnalysis complete for {dc_name}!")
        print(f"Total groupings analyzed: {len(summary_df)}")
        if not summary_df.empty:
            print(f"Flagged groupings: {len(summary_df[summary_df['flagged']])}")
        
        return summary_df, threshold_df


def main(input_folder='greedygaussv4', output_folder='analysis_output'):
    """
    Main function to run the analysis.
    
    Args:
        input_folder: Path to the folder containing test case folders
        output_folder: Path where all output should be saved
    """
    import os
    
    base_folder = Path(input_folder)
    output_base = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_base.mkdir(exist_ok=True)
    
    all_results = []
    all_thresholds = []
    
    # Find all test case folders
    test_case_folders = [f for f in base_folder.iterdir() if f.is_dir()]
    
    print(f"Found {len(test_case_folders)} test case folders in {base_folder}")
    print(f"Output will be saved to: {output_base.absolute()}")
    
    # Process each test case folder
    for test_case_folder in test_case_folders:
        test_case_name = test_case_folder.name
        print(f"\n{'='*60}")
        print(f"Processing Test Case: {test_case_name}")
        print(f"{'='*60}")
        
        # Process DC1 within this test case
        dc1_path = test_case_folder / 'dc1'
        if dc1_path.exists():
            print(f"\nProcessing {test_case_name}/dc1...")
            dc1_analyzer = VoltageSegmentAnalyzer(
                dc1_path, 
                cache_thresholds=True,
                output_base_dir=output_base
            )
            dc1_results, dc1_thresholds = dc1_analyzer.run_analysis(test_case_name=test_case_name)
            
            if not dc1_results.empty:
                dc1_results['test_case_folder'] = test_case_name
                dc1_results['dc_folder'] = 'dc1'
                all_results.append(dc1_results)
            
            if not dc1_thresholds.empty:
                dc1_thresholds['test_case_folder'] = test_case_name
                dc1_thresholds['dc_folder'] = 'dc1'
                all_thresholds.append(dc1_thresholds)
        else:
            print(f"  No dc1 folder found in {test_case_name}")
        
        # Process DC2 within this test case
        dc2_path = test_case_folder / 'dc2'
        if dc2_path.exists():
            print(f"\nProcessing {test_case_name}/dc2...")
            dc2_analyzer = VoltageSegmentAnalyzer(
                dc2_path,
                cache_thresholds=True,
                output_base_dir=output_base
            )
            dc2_results, dc2_thresholds = dc2_analyzer.run_analysis(test_case_name=test_case_name)
            
            if not dc2_results.empty:
                dc2_results['test_case_folder'] = test_case_name
                dc2_results['dc_folder'] = 'dc2'
                all_results.append(dc2_results)
            
            if not dc2_thresholds.empty:
                dc2_thresholds['test_case_folder'] = test_case_name
                dc2_thresholds['dc_folder'] = 'dc2'
                all_thresholds.append(dc2_thresholds)
        else:
            print(f"  No dc2 folder found in {test_case_name}")
    
    # Check if we have results
    if not all_results:
        print("\nNo data to analyze. Exiting.")
        return
    
    # Combine all results and thresholds
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_thresholds = pd.concat(all_thresholds, ignore_index=True) if all_thresholds else pd.DataFrame()
    
    # Create combined overview plots in output directory
    if not combined_df.empty:
        print("\n" + "="*60)
        print("Creating combined overview visualizations...")
        print("="*60)
        
        overview_dir = output_base / 'combined_analysis'
        overview_dir.mkdir(exist_ok=True)
        
        # Create overview for each DC type
        for dc_type in ['dc1', 'dc2']:
            dc_data = combined_df[combined_df['dc_folder'] == dc_type]
            if not dc_data.empty:
                dc_thresh = combined_thresholds[combined_thresholds['dc_folder'] == dc_type] if not combined_thresholds.empty else pd.DataFrame()
                
                # Create a temporary analyzer just for plotting
                temp_analyzer = VoltageSegmentAnalyzer(
                    Path('.'), 
                    cache_thresholds=False,
                    output_base_dir=output_base
                )
                
                overview_path = overview_dir / f'{dc_type}_all_testcases_overview.html'
                temp_analyzer.create_threshold_overview(dc_data, dc_thresh, overview_path)
                
                distribution_path = overview_dir / f'{dc_type}_all_testcases_distributions.html'
                temp_analyzer.create_distribution_plots(dc_data, dc_thresh, distribution_path)
    
    # Save combined Excel report in output directory
    combined_excel_path = output_base / 'combined_flagged_analysis.xlsx'
    
    with pd.ExcelWriter(combined_excel_path, engine='openpyxl') as writer:
        # All results
        combined_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Only flagged results
        flagged_df = combined_df[combined_df['flagged']]
        if not flagged_df.empty:
            flagged_df.to_excel(writer, sheet_name='Flagged_Only', index=False)
        
        # Dynamic thresholds used
        if not combined_thresholds.empty:
            combined_thresholds.to_excel(writer, sheet_name='Dynamic_Thresholds', index=False)
        
        # Statistics by test case folder, DC folder and label
        if 'test_case_folder' in combined_df.columns:
            stats_by_all = combined_df.groupby(['test_case_folder', 'dc_folder', 'label']).agg({
                'mean_voltage': ['mean', 'std'],
                'variance': ['mean', 'std'],
                'abs_slope': ['mean', 'std'],
                'n_outliers_zscore': 'sum',
                'flagged': 'sum'
            }).round(3)
            stats_by_all.to_excel(writer, sheet_name='Statistics_by_TestCase_DC_Label')
        
        # Separate sheets for each test case folder
        for test_case_folder in combined_df['test_case_folder'].unique():
            test_case_data = flagged_df[flagged_df['test_case_folder'] == test_case_folder]
            if not test_case_data.empty:
                sheet_name = f"Flagged_{test_case_folder[:20]}"  # Limit sheet name length
                test_case_data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("\n" + "=" * 60)
    print("Combined Analysis Complete!")
    print("=" * 60)
    print(f"Output directory: {output_base.absolute()}")
    print(f"Combined Excel: {combined_excel_path}")
    print(f"Total groupings analyzed: {len(combined_df)}")
    
    if not flagged_df.empty:
        print(f"Total flagged groupings: {len(flagged_df)}")
        
        # Summary by test case folder
        print("\nFlagged summary by test case:")
        for test_case_folder in combined_df['test_case_folder'].unique():
            tc_flagged = flagged_df[flagged_df['test_case_folder'] == test_case_folder]
            if not tc_flagged.empty:
                dc1_count = len(tc_flagged[tc_flagged['dc_folder'] == 'dc1'])
                dc2_count = len(tc_flagged[tc_flagged['dc_folder'] == 'dc2'])
                print(f"  {test_case_folder}: DC1={dc1_count}, DC2={dc2_count}")
    
    if not combined_thresholds.empty:
        print("\nDynamic Threshold Summary (first 20 rows):")
        summary = combined_thresholds.groupby(['test_case_folder', 'dc_folder', 'label', 'metric'])['threshold'].mean().round(3)
        print(summary.head(20))
    
    print("\nAll processing complete!")
    print(f"Check {output_base}/combined_analysis/ for overview visualizations")


if __name__ == "__main__":
    # You can customize the input and output folders here
    # main(input_folder='greedygaussv4', output_folder='my_custom_output')
    main()  # Uses default folders
