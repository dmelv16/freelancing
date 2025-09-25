"""
Power Data Clustering Pipeline using GreedyGaussianSegmenter
Comprehensive analysis with multi-level grouping and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from aeon.segmentation import GreedyGaussianSegmenter
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PowerDataClusteringPipeline:
    """
    Complete pipeline for power data clustering with multi-level analysis
    """
    
    def __init__(self, data_path: str, output_dir: str = "./clustering_results"):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the parquet file
            output_dir: Directory to save results and models
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.stats_dir = self.output_dir / "stats"
        
        for dir in [self.models_dir, self.plots_dir, self.stats_dir]:
            dir.mkdir(exist_ok=True)
        
        self.df = None
        self.test_case_models = {}
        self.cluster_mappings = {}
        
    def load_and_prepare_data(self):
        """Load parquet file and prepare feature columns"""
        print("Loading data from parquet file...")
        self.df = pd.read_parquet(self.data_path)
        
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # Identify grouping columns
        self.grouping_cols = ['test_case', 'test_run', 'save', 'unit_id', 'station', 'aircraft']
        existing_grouping_cols = [col for col in self.grouping_cols if col in self.df.columns]
        
        if not existing_grouping_cols:
            print("Warning: No standard grouping columns found. Using all data as single group.")
            self.df['test_case'] = 'default'
            existing_grouping_cols = ['test_case']
        
        self.grouping_cols = existing_grouping_cols
        
        # Create rolling features if they don't exist
        self._create_rolling_features()
        
        return self.df
    
    def _create_rolling_features(self):
        """Create rolling features for both dc1 and dc2"""
        print("Creating rolling features...")
        
        # Define the base columns to work with
        dc_channels = ['dc1', 'dc2']
        
        for dc in dc_channels:
            voltage_col = f'voltage_28v_{dc}_cal'
            
            if voltage_col in self.df.columns:
                # Rolling averages
                self.df[f'average_3v_{dc}'] = self.df[voltage_col].rolling(window=3, center=True).mean()
                self.df[f'average_5v_{dc}'] = self.df[voltage_col].rolling(window=5, center=True).mean()
                
                # Rolling slopes (difference over window)
                self.df[f'slope_3_V_{dc}'] = self.df[voltage_col].diff(3) / 3
                self.df[f'slope_5_V_{dc}'] = self.df[voltage_col].diff(5) / 5
                
                # Additional features for better characterization
                self.df[f'std_3v_{dc}'] = self.df[voltage_col].rolling(window=3, center=True).std()
                self.df[f'range_5v_{dc}'] = (self.df[voltage_col].rolling(window=5, center=True).max() - 
                                             self.df[voltage_col].rolling(window=5, center=True).min())
        
        # Fill NaN values from rolling operations
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for clustering"""
        feature_patterns = [
            'voltage_28v_dc1_cal', 'voltage_28v_dc2_cal',
            'average_3v_dc1', 'average_3v_dc2',
            'average_5v_dc1', 'average_5v_dc2',
            'slope_3_V_dc1', 'slope_3_V_dc2',
            'slope_5_V_dc1', 'slope_5_V_dc2',
            'std_3v_dc1', 'std_3v_dc2',
            'range_5v_dc1', 'range_5v_dc2'
        ]
        
        return [col for col in feature_patterns if col in self.df.columns]
    
    def train_test_case_models(self, k_max: int = 5, max_shuffles: int = 1):
        """
        Train a separate model for each test case
        
        Args:
            k_max: Maximum number of clusters
            max_shuffles: Maximum shuffles for GreedyGaussianSegmenter
        """
        test_cases = self.df['test_case'].unique() if 'test_case' in self.df.columns else ['all_data']
        
        print(f"\nTraining models for {len(test_cases)} test cases...")
        
        for test_case in test_cases:
            print(f"\n{'='*50}")
            print(f"Processing Test Case: {test_case}")
            print(f"{'='*50}")
            
            # Filter data for this test case
            if test_case == 'all_data':
                test_case_data = self.df
            else:
                test_case_data = self.df[self.df['test_case'] == test_case].copy()
            
            if len(test_case_data) < 100:
                print(f"Skipping {test_case}: insufficient data ({len(test_case_data)} points)")
                continue
            
            # Get features
            feature_cols = self.get_feature_columns()
            features = test_case_data[feature_cols].values
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train GreedyGaussianSegmenter from aeon
            print(f"Training GreedyGaussianSegmenter with k_max={k_max}")
            ggs = GreedyGaussianSegmenter(k_max=k_max, max_shuffles=max_shuffles)
            
            # GreedyGaussianSegmenter expects shape (n_channels, n_timepoints)
            # So we transpose our features
            predicted_labels = ggs.fit_predict(features_scaled.T)
            
            # Store results
            test_case_data['cluster'] = predicted_labels
            
            # Analyze clusters and create mapping
            cluster_mapping = self._analyze_clusters(test_case_data, test_case)
            
            # Store model and mapping
            self.test_case_models[test_case] = {
                'model': ggs,
                'scaler': scaler,
                'features': feature_cols,
                'data': test_case_data,
                'cluster_mapping': cluster_mapping,
                'n_clusters': len(np.unique(predicted_labels))
            }
            
            # Save model
            model_path = self.models_dir / f"model_{test_case}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.test_case_models[test_case], f)
            
            # Create visualizations
            self._create_test_case_visualizations(test_case_data, test_case)
    
    def _analyze_clusters(self, data: pd.DataFrame, test_case: str) -> Dict:
        """
        Analyze clusters and create detailed characterization WITHOUT business assumptions
        """
        print(f"\nAnalyzing clusters for {test_case}...")
        
        cluster_stats = []
        feature_cols = self.get_feature_columns()
        
        for cluster in sorted(data['cluster'].unique()):
            cluster_data = data[data['cluster'] == cluster]
            
            stats = {
                'cluster': cluster,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
            }
            
            # Calculate statistics for voltage columns
            for dc in ['dc1', 'dc2']:
                voltage_col = f'voltage_28v_{dc}_cal'
                slope_col = f'slope_3_V_{dc}'
                
                if voltage_col in data.columns:
                    stats[f'avg_voltage_{dc}'] = cluster_data[voltage_col].mean()
                    stats[f'std_voltage_{dc}'] = cluster_data[voltage_col].std()
                    stats[f'min_voltage_{dc}'] = cluster_data[voltage_col].min()
                    stats[f'max_voltage_{dc}'] = cluster_data[voltage_col].max()
                    stats[f'median_voltage_{dc}'] = cluster_data[voltage_col].median()
                
                if slope_col in data.columns:
                    stats[f'avg_slope_{dc}'] = cluster_data[slope_col].mean()
                    stats[f'std_slope_{dc}'] = cluster_data[slope_col].std()
                    stats[f'max_slope_{dc}'] = cluster_data[slope_col].max()
                    stats[f'min_slope_{dc}'] = cluster_data[slope_col].min()
            
            # Temporal position analysis - just for information, not for business logic
            positions = cluster_data.index.values if isinstance(cluster_data.index.values[0], int) else np.arange(len(cluster_data))
            relative_positions = positions / len(data)
            stats['avg_position'] = np.mean(relative_positions)
            stats['first_occurrence'] = np.min(relative_positions)
            stats['last_occurrence'] = np.max(relative_positions)
            stats['position_spread'] = np.std(relative_positions)
            
            # Continuity analysis - does this cluster appear in continuous blocks?
            cluster_runs = (data['cluster'] == cluster).astype(int)
            runs = cluster_runs.diff().ne(0).cumsum()
            run_lengths = cluster_runs.groupby(runs).sum()
            run_lengths = run_lengths[run_lengths > 0]
            
            stats['num_continuous_blocks'] = len(run_lengths)
            stats['avg_block_length'] = run_lengths.mean()
            stats['max_block_length'] = run_lengths.max()
            
            cluster_stats.append(stats)
        
        # Create DataFrame for easy viewing
        stats_df = pd.DataFrame(cluster_stats)
        
        # Save detailed statistics
        stats_path = self.stats_dir / f"cluster_stats_{test_case}.csv"
        stats_df.to_csv(stats_path, index=False)
        
        print("\nCluster Statistics:")
        print("-" * 100)
        
        # Print formatted statistics for better readability
        for _, row in stats_df.iterrows():
            print(f"\nCluster {row['cluster']}:")
            print(f"  Size: {row['count']} points ({row['percentage']:.1f}%)")
            print(f"  DC1 Voltage: {row.get('avg_voltage_dc1', 0):.1f}V ± {row.get('std_voltage_dc1', 0):.1f}V (range: {row.get('min_voltage_dc1', 0):.1f}-{row.get('max_voltage_dc1', 0):.1f}V)")
            print(f"  DC1 Slope: {row.get('avg_slope_dc1', 0):.3f} ± {row.get('std_slope_dc1', 0):.3f}")
            print(f"  Temporal: First at {row['first_occurrence']*100:.1f}%, Last at {row['last_occurrence']*100:.1f}%, Avg at {row['avg_position']*100:.1f}%")
            print(f"  Continuity: {row['num_continuous_blocks']} blocks, avg length {row['avg_block_length']:.0f} points")
        
        # Create descriptive mapping (not business mapping)
        mapping = self._create_physical_state_mapping(stats_df)
        
        print("\nCluster Descriptive Labels:")
        for cluster, label in mapping.items():
            print(f"  Cluster {cluster}: {label}")
        
        return {
            'stats': stats_df,
            'descriptive_mapping': mapping
        }
    
    def _create_physical_state_mapping(self, stats_df: pd.DataFrame) -> Dict:
        """
        Create descriptive mapping from clusters based on their characteristics
        WITHOUT making assumptions about business meaning
        """
        mapping = {}
        
        # Use dc1 as primary reference (adjust if dc2 is primary)
        if 'avg_voltage_dc1' in stats_df.columns:
            voltage_col = 'avg_voltage_dc1'
            slope_col = 'avg_slope_dc1'
        else:
            # Fallback to dc2 or any available
            voltage_col = stats_df.columns[stats_df.columns.str.contains('avg_voltage')][0] if any(stats_df.columns.str.contains('avg_voltage')) else None
            slope_col = stats_df.columns[stats_df.columns.str.contains('avg_slope')][0] if any(stats_df.columns.str.contains('avg_slope')) else None
        
        if voltage_col and slope_col:
            for _, row in stats_df.iterrows():
                cluster = row['cluster']
                avg_voltage = row[voltage_col]
                avg_slope = row[slope_col]
                position = row.get('avg_position', 0.5)
                
                # Create descriptive label based on characteristics
                voltage_desc = ""
                if avg_voltage < 1.0:
                    voltage_desc = "zero_V"
                elif avg_voltage < 10:
                    voltage_desc = "low_V"
                elif avg_voltage < 20:
                    voltage_desc = "mid_V"
                else:
                    voltage_desc = "high_V"
                
                slope_desc = ""
                if abs(avg_slope) < 0.1:
                    slope_desc = "stable"
                elif avg_slope > 0.5:
                    slope_desc = "fast_increasing"
                elif avg_slope > 0.1:
                    slope_desc = "slow_increasing"
                elif avg_slope < -0.5:
                    slope_desc = "fast_decreasing"
                else:
                    slope_desc = "slow_decreasing"
                
                # Position descriptor
                pos_desc = ""
                if position < 0.2:
                    pos_desc = "early"
                elif position > 0.8:
                    pos_desc = "late"
                else:
                    pos_desc = "middle"
                
                # Combine descriptors
                mapping[cluster] = f"{voltage_desc}_{slope_desc}_{pos_desc}"
        
        return mapping
    
    def _create_test_case_visualizations(self, data: pd.DataFrame, test_case: str):
        """Create comprehensive visualizations for each test case"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Test Case: {test_case} - Clustering Analysis', fontsize=16)
        
        # Plot 1: Time series with clusters
        ax = axes[0, 0]
        if 'voltage_28v_dc1_cal' in data.columns:
            for cluster in sorted(data['cluster'].unique()):
                cluster_data = data[data['cluster'] == cluster]
                ax.scatter(range(len(cluster_data)), 
                          cluster_data['voltage_28v_dc1_cal'],
                          label=f'Cluster {cluster}',
                          alpha=0.6, s=1)
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Voltage DC1 (V)')
            ax.set_title('Voltage Time Series by Cluster')
            ax.legend()
        
        # Plot 2: Slope vs Voltage scatter
        ax = axes[0, 1]
        if 'voltage_28v_dc1_cal' in data.columns and 'slope_3_V_dc1' in data.columns:
            scatter = ax.scatter(data['voltage_28v_dc1_cal'], 
                                data['slope_3_V_dc1'],
                                c=data['cluster'],
                                cmap='viridis',
                                alpha=0.6)
            ax.set_xlabel('Voltage DC1 (V)')
            ax.set_ylabel('Slope DC1 (V/sample)')
            ax.set_title('Voltage vs Slope Clustering')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Plot 3: Cluster distribution over time
        ax = axes[1, 0]
        cluster_counts = data.groupby('cluster').size()
        ax.bar(cluster_counts.index, cluster_counts.values)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Cluster Size Distribution')
        
        # Plot 4: DC1 vs DC2 comparison (if both exist)
        ax = axes[1, 1]
        if 'voltage_28v_dc1_cal' in data.columns and 'voltage_28v_dc2_cal' in data.columns:
            scatter = ax.scatter(data['voltage_28v_dc1_cal'],
                                data['voltage_28v_dc2_cal'],
                                c=data['cluster'],
                                cmap='viridis',
                                alpha=0.6)
            ax.set_xlabel('Voltage DC1 (V)')
            ax.set_ylabel('Voltage DC2 (V)')
            ax.set_title('DC1 vs DC2 Voltage')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Plot 5: Temporal cluster sequence
        ax = axes[2, 0]
        ax.plot(data['cluster'].values, linewidth=0.5)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Cluster ID')
        ax.set_title('Cluster Sequence Over Time')
        
        # Plot 6: Feature importance heatmap
        ax = axes[2, 1]
        feature_cols = [col for col in self.get_feature_columns() if col in data.columns][:6]  # Top 6 features
        if feature_cols:
            cluster_means = data.groupby('cluster')[feature_cols].mean()
            sns.heatmap(cluster_means.T, annot=True, fmt='.2f', ax=ax, cmap='coolwarm')
            ax.set_title('Mean Feature Values by Cluster')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'test_case_{test_case}_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for {test_case}")
    
    def create_manual_mapping_interface(self):
        """
        After reviewing cluster statistics, manually assign business meanings
        This gives YOU control over what each cluster means for each test case
        """
        print("\n" + "="*60)
        print("Manual Cluster Mapping Interface")
        print("="*60)
        
        manual_mappings = {}
        
        for test_case, model_data in self.test_case_models.items():
            stats_df = model_data['cluster_mapping']['stats']
            
            print(f"\n\nTest Case: {test_case}")
            print("-" * 40)
            print("\nBased on the statistics, assign each cluster to one of:")
            print("  - de_energized")
            print("  - ramp_up")
            print("  - steady_state") 
            print("  - ramp_down")
            print("  - transient (for noise/brief transitions)")
            
            # Show summary for decision making
            for _, row in stats_df.iterrows():
                print(f"\nCluster {row['cluster']}:")
                print(f"  {row['percentage']:.1f}% of data")
                print(f"  Voltage: {row.get('avg_voltage_dc1', 0):.1f}V (slope: {row.get('avg_slope_dc1', 0):.3f})")
                print(f"  Appears at: {row.get('avg_position', 0.5)*100:.0f}% through test")
                print(f"  Suggestion: {model_data['cluster_mapping']['descriptive_mapping'][row['cluster']]}")
            
            # In production, you'd load these from a config file
            # For now, showing the structure
            test_case_mapping = {}
            print(f"\n→ Define mappings for {test_case} in config file")
            
            manual_mappings[test_case] = test_case_mapping
        
        return manual_mappings
    
    def create_multilevel_summary(self):
        """
        Create comprehensive summary statistics at multiple levels
        """
        print("\n" + "="*60)
        print("Multi-Level Summary Statistics")
        print("="*60)
        
        all_summaries = []
        
        for test_case, model_data in self.test_case_models.items():
            data = model_data['data']
            
            # Get all grouping levels available
            available_groups = [col for col in self.grouping_cols if col in data.columns]
            
            for group_level in range(1, len(available_groups) + 1):
                group_cols = available_groups[:group_level]
                
                if 'business_state' in data.columns:
                    summary = data.groupby(group_cols + ['business_state']).agg({
                        'voltage_28v_dc1_cal': ['mean', 'std', 'min', 'max'],
                        'slope_3_V_dc1': ['mean', 'std']
                    }).round(2)
                    
                    summary_flat = summary.reset_index()
                    summary_flat['grouping_level'] = '_'.join(group_cols)
                    all_summaries.append(summary_flat)
        
        # Save comprehensive summary
        if all_summaries:
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            summary_path = self.stats_dir / "multilevel_summary.csv"
            combined_summary.to_csv(summary_path, index=False)
            print(f"Saved multi-level summary to {summary_path}")
    
    def analyze_aircraft_anomalies(self):
        """
        Comprehensive aircraft-level analysis to identify problematic aircraft
        """
        print("\n" + "="*60)
        print("Aircraft-Level Anomaly Analysis")
        print("="*60)
        
        # Collect all data across test cases
        all_aircraft_data = []
        
        for test_case, model_data in self.test_case_models.items():
            data = model_data['data'].copy()
            data['test_case'] = test_case
            all_aircraft_data.append(data)
        
        if not all_aircraft_data:
            print("No data available for aircraft analysis")
            return
        
        combined_data = pd.concat(all_aircraft_data, ignore_index=True)
        
        # Check if aircraft column exists
        if 'aircraft' not in combined_data.columns:
            print("No aircraft column found in data")
            return
        
        aircraft_metrics = []
        
        for aircraft in combined_data['aircraft'].unique():
            aircraft_data = combined_data[combined_data['aircraft'] == aircraft]
            
            metrics = {
                'aircraft': aircraft,
                'total_tests': aircraft_data['test_case'].nunique() if 'test_case' in aircraft_data.columns else 1,
                'total_runs': aircraft_data['test_run'].nunique() if 'test_run' in aircraft_data.columns else 1,
                'total_data_points': len(aircraft_data)
            }
            
            # Power quality metrics
            if 'voltage_28v_dc1_cal' in aircraft_data.columns:
                steady_state_data = aircraft_data[aircraft_data['business_state'] == 'steady_state'] if 'business_state' in aircraft_data.columns else aircraft_data
                
                if len(steady_state_data) > 0:
                    metrics['avg_steady_voltage'] = steady_state_data['voltage_28v_dc1_cal'].mean()
                    metrics['voltage_stability'] = steady_state_data['voltage_28v_dc1_cal'].std()
                    metrics['voltage_drops'] = len(aircraft_data[aircraft_data['voltage_28v_dc1_cal'] < 25])  # Count voltage drops
                    metrics['voltage_drop_rate'] = metrics['voltage_drops'] / len(aircraft_data) * 100
                
                # Ramp characteristics
                ramp_up_data = aircraft_data[aircraft_data['business_state'] == 'ramp_up'] if 'business_state' in aircraft_data.columns else pd.DataFrame()
                ramp_down_data = aircraft_data[aircraft_data['business_state'] == 'ramp_down'] if 'business_state' in aircraft_data.columns else pd.DataFrame()
                
                if len(ramp_up_data) > 0:
                    metrics['avg_ramp_up_time'] = len(ramp_up_data) / metrics['total_runs'] if metrics['total_runs'] > 0 else 0
                    metrics['avg_ramp_up_slope'] = ramp_up_data['slope_3_V_dc1'].mean() if 'slope_3_V_dc1' in ramp_up_data.columns else 0
                
                if len(ramp_down_data) > 0:
                    metrics['avg_ramp_down_time'] = len(ramp_down_data) / metrics['total_runs'] if metrics['total_runs'] > 0 else 0
                    metrics['avg_ramp_down_slope'] = ramp_down_data['slope_3_V_dc1'].mean() if 'slope_3_V_dc1' in ramp_down_data.columns else 0
                
                # Anomaly indicators
                metrics['de_energized_percentage'] = len(aircraft_data[aircraft_data['business_state'] == 'de_energized']) / len(aircraft_data) * 100 if 'business_state' in aircraft_data.columns else 0
                
                # Check for unusual patterns
                if 'cluster' in aircraft_data.columns:
                    cluster_transitions = aircraft_data['cluster'].diff().fillna(0).abs()
                    metrics['cluster_transitions'] = cluster_transitions.sum()
                    metrics['transition_rate'] = metrics['cluster_transitions'] / len(aircraft_data) * 100
                    
                    # High transition rate might indicate instability
                    metrics['stability_score'] = 100 - min(metrics['transition_rate'] * 10, 100)
            
            aircraft_metrics.append(metrics)
        
        # Create DataFrame and calculate anomaly scores
        aircraft_df = pd.DataFrame(aircraft_metrics)
        
        # Calculate composite anomaly score
        if len(aircraft_df) > 1:
            # Normalize metrics for scoring
            for col in ['voltage_stability', 'voltage_drop_rate', 'de_energized_percentage', 'transition_rate']:
                if col in aircraft_df.columns:
                    # Higher values = more problematic
                    aircraft_df[f'{col}_zscore'] = (aircraft_df[col] - aircraft_df[col].mean()) / aircraft_df[col].std()
            
            # Create composite anomaly score
            zscore_cols = [col for col in aircraft_df.columns if col.endswith('_zscore')]
            if zscore_cols:
                aircraft_df['anomaly_score'] = aircraft_df[zscore_cols].mean(axis=1)
                aircraft_df['anomaly_rank'] = aircraft_df['anomaly_score'].rank(ascending=False)
        
        # Sort by anomaly score
        if 'anomaly_score' in aircraft_df.columns:
            aircraft_df = aircraft_df.sort_values('anomaly_score', ascending=False)
        
        # Save aircraft analysis
        aircraft_path = self.stats_dir / "aircraft_analysis.csv"
        aircraft_df.to_csv(aircraft_path, index=False)
        
        print("\nAircraft Anomaly Rankings:")
        print("-" * 60)
        
        # Print top problematic aircraft
        if 'anomaly_score' in aircraft_df.columns:
            top_issues = aircraft_df.head(5)
            for _, row in top_issues.iterrows():
                print(f"\n{row['aircraft']}:")
                print(f"  Anomaly Score: {row['anomaly_score']:.2f}")
                if 'voltage_stability' in row:
                    print(f"  Voltage Stability (std): {row['voltage_stability']:.2f}V")
                if 'voltage_drop_rate' in row:
                    print(f"  Voltage Drop Rate: {row['voltage_drop_rate']:.1f}%")
                if 'transition_rate' in row:
                    print(f"  State Transition Rate: {row['transition_rate']:.1f}%")
                if 'stability_score' in row:
                    print(f"  Stability Score: {row['stability_score']:.1f}/100")
        
        # Create aircraft comparison visualizations
        self._create_aircraft_comparison_plots(aircraft_df, combined_data)
        
        return aircraft_df
    
    def _create_aircraft_comparison_plots(self, aircraft_df: pd.DataFrame, combined_data: pd.DataFrame):
        """
        Create visualizations comparing aircraft performance
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Aircraft Performance Comparison', fontsize=16)
        
        # Plot 1: Anomaly scores by aircraft
        ax = axes[0, 0]
        if 'anomaly_score' in aircraft_df.columns:
            aircraft_sorted = aircraft_df.sort_values('anomaly_score', ascending=False).head(10)
            ax.barh(range(len(aircraft_sorted)), aircraft_sorted['anomaly_score'])
            ax.set_yticks(range(len(aircraft_sorted)))
            ax.set_yticklabels(aircraft_sorted['aircraft'])
            ax.set_xlabel('Anomaly Score (Higher = More Issues)')
            ax.set_title('Top 10 Aircraft by Anomaly Score')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Voltage stability comparison
        ax = axes[0, 1]
        if 'voltage_stability' in aircraft_df.columns:
            aircraft_sorted = aircraft_df.sort_values('voltage_stability', ascending=False).head(10)
            ax.barh(range(len(aircraft_sorted)), aircraft_sorted['voltage_stability'])
            ax.set_yticks(range(len(aircraft_sorted)))
            ax.set_yticklabels(aircraft_sorted['aircraft'])
            ax.set_xlabel('Voltage Stability (Std Dev)')
            ax.set_title('Aircraft with Highest Voltage Instability')
        
        # Plot 3: State distribution by aircraft
        ax = axes[1, 0]
        if 'business_state' in combined_data.columns:
            top_aircraft = aircraft_df.head(5)['aircraft'].tolist() if len(aircraft_df) > 5 else aircraft_df['aircraft'].tolist()
            state_dist = combined_data[combined_data['aircraft'].isin(top_aircraft)].groupby(['aircraft', 'business_state']).size().unstack(fill_value=0)
            state_dist_pct = state_dist.div(state_dist.sum(axis=1), axis=0) * 100
            state_dist_pct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_xlabel('Aircraft')
            ax.set_ylabel('State Distribution (%)')
            ax.set_title('Power State Distribution by Aircraft')
            ax.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Voltage drop frequency
        ax = axes[1, 1]
        if 'voltage_drop_rate' in aircraft_df.columns:
            ax.scatter(aircraft_df['total_tests'], aircraft_df['voltage_drop_rate'], 
                      s=aircraft_df['total_data_points']/100, alpha=0.6)
            ax.set_xlabel('Number of Tests')
            ax.set_ylabel('Voltage Drop Rate (%)')
            ax.set_title('Voltage Drop Rate vs Test Count\n(bubble size = data points)')
            
            # Annotate outliers
            for _, row in aircraft_df.iterrows():
                if row['voltage_drop_rate'] > aircraft_df['voltage_drop_rate'].mean() + 2*aircraft_df['voltage_drop_rate'].std():
                    ax.annotate(row['aircraft'], (row['total_tests'], row['voltage_drop_rate']),
                              fontsize=8, alpha=0.7)
        
        # Plot 5: Ramp characteristics
        ax = axes[2, 0]
        if 'avg_ramp_up_slope' in aircraft_df.columns and 'avg_ramp_down_slope' in aircraft_df.columns:
            ax.scatter(aircraft_df['avg_ramp_up_slope'], aircraft_df['avg_ramp_down_slope'],
                      s=50, alpha=0.6)
            ax.set_xlabel('Avg Ramp Up Slope')
            ax.set_ylabel('Avg Ramp Down Slope')
            ax.set_title('Ramp Characteristics by Aircraft')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Annotate problematic aircraft
            if 'anomaly_score' in aircraft_df.columns:
                top_anomalies = aircraft_df.nlargest(3, 'anomaly_score')
                for _, row in top_anomalies.iterrows():
                    if pd.notna(row.get('avg_ramp_up_slope', np.nan)) and pd.notna(row.get('avg_ramp_down_slope', np.nan)):
                        ax.annotate(row['aircraft'], 
                                  (row['avg_ramp_up_slope'], row['avg_ramp_down_slope']),
                                  fontsize=8, color='red', alpha=0.7)
        
        # Plot 6: Timeline of issues
        ax = axes[2, 1]
        if 'test_run' in combined_data.columns and 'voltage_28v_dc1_cal' in combined_data.columns:
            # Show voltage drops over time for top problematic aircraft
            if 'anomaly_score' in aircraft_df.columns:
                worst_aircraft = aircraft_df.iloc[0]['aircraft']
                worst_data = combined_data[combined_data['aircraft'] == worst_aircraft]
                
                if len(worst_data) > 0:
                    ax.plot(worst_data['voltage_28v_dc1_cal'].values, alpha=0.7, linewidth=0.5)
                    ax.set_xlabel('Time Index')
                    ax.set_ylabel('Voltage (V)')
                    ax.set_title(f'Voltage Profile: {worst_aircraft}\n(Most Problematic Aircraft)')
                    ax.axhline(y=28, color='g', linestyle='--', alpha=0.5, label='Target')
                    ax.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Low Threshold')
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'aircraft_comparison_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved aircraft comparison plots")
    
    def run_full_pipeline(self, custom_mappings: Dict = None):
        """
        Execute the complete clustering pipeline
        
        Args:
            custom_mappings: Optional dictionary of test_case -> cluster -> business_state mappings
                           e.g., {'TestA': {0: 'de_energized', 1: 'ramp_up', 2: 'steady_state'}}
        """
        
        # Step 1: Load and prepare data
        print("\n" + "="*60)
        print("STEP 1: Loading and Preparing Data")
        print("="*60)
        self.load_and_prepare_data()
        
        # Step 2: Train models for each test case
        print("\n" + "="*60)
        print("STEP 2: Training Models")
        print("="*60)
        self.train_test_case_models()
        
        # Step 3: Apply custom mappings if provided
        if custom_mappings:
            print("\n" + "="*60)
            print("STEP 3: Applying Custom Business Mappings")
            print("="*60)
            self.apply_custom_mappings(custom_mappings)
        else:
            print("\n" + "="*60)
            print("STEP 3: Review Statistics to Create Mappings")
            print("="*60)
            self.create_manual_mapping_interface()
        
        # Step 4: Create multi-level summaries
        print("\n" + "="*60)
        print("STEP 4: Creating Multi-Level Summaries")
        print("="*60)
        self.create_multilevel_summary()
        
        # Step 5: Aircraft anomaly analysis
        print("\n" + "="*60)
        print("STEP 5: Aircraft Anomaly Detection")
        print("="*60)
        self.analyze_aircraft_anomalies()
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        return self.test_case_models
    
    def predict_new_data(self, new_data_path: str, test_case: str) -> pd.DataFrame:
        """
        Use a trained model to predict on new data
        
        Args:
            new_data_path: Path to new parquet file
            test_case: Which test case model to use
            
        Returns:
            DataFrame with predictions
        """
        if test_case not in self.test_case_models:
            raise ValueError(f"No model found for test case: {test_case}")
        
        # Load new data
        new_df = pd.read_parquet(new_data_path)
        
        # Create rolling features
        self._create_rolling_features_for_df(new_df)
        
        # Get model components
        model_data = self.test_case_models[test_case]
        ggs = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['features']
        
        # Prepare features
        features = new_df[feature_cols].values
        features_scaled = scaler.transform(features)
        
        # Predict using the trained GreedyGaussianSegmenter
        predicted_labels = ggs.predict(features_scaled.T)
        
        # Apply mappings
        new_df['cluster'] = predicted_labels
        new_df['descriptive_state'] = new_df['cluster'].map(model_data['cluster_mapping']['descriptive_mapping'])
        
        # Apply business mapping if it exists
        if 'business_mapping' in model_data:
            new_df['business_state'] = new_df['cluster'].map(model_data['business_mapping'])
        
        return new_df
    
    def _create_rolling_features_for_df(self, df: pd.DataFrame):
        """Create rolling features for a dataframe (used for prediction on new data)"""
        dc_channels = ['dc1', 'dc2']
        
        for dc in dc_channels:
            voltage_col = f'voltage_28v_{dc}_cal'
            
            if voltage_col in df.columns:
                df[f'average_3v_{dc}'] = df[voltage_col].rolling(window=3, center=True).mean()
                df[f'average_5v_{dc}'] = df[voltage_col].rolling(window=5, center=True).mean()
                df[f'slope_3_V_{dc}'] = df[voltage_col].diff(3) / 3
                df[f'slope_5_V_{dc}'] = df[voltage_col].diff(5) / 5
                df[f'std_3v_{dc}'] = df[voltage_col].rolling(window=3, center=True).std()
                df[f'range_5v_{dc}'] = (df[voltage_col].rolling(window=5, center=True).max() - 
                                        df[voltage_col].rolling(window=5, center=True).min())
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        """Execute the complete clustering pipeline"""
        
        # Step 1: Load and prepare data
        print("\n" + "="*60)
        print("STEP 1: Loading and Preparing Data")
        print("="*60)
        self.load_and_prepare_data()
        
        # Step 2: Train models for each test case
        print("\n" + "="*60)
        print("STEP 2: Training Models")
        print("="*60)
        self.train_test_case_models()
        
        # Step 3: Apply business logic
        print("\n" + "="*60)
        print("STEP 3: Applying Business Logic")
        print("="*60)
        self.create_business_state_mapping()
        
        # Step 4: Create multi-level summaries
        print("\n" + "="*60)
        print("STEP 4: Creating Multi-Level Summaries")
        print("="*60)
        self.create_multilevel_summary()
        
        # Step 5: Aircraft anomaly analysis
        print("\n" + "="*60)
        print("STEP 5: Aircraft Anomaly Detection")
        print("="*60)
        self.analyze_aircraft_anomalies()
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        return self.test_case_models


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PowerDataClusteringPipeline(
        data_path="your_data.parquet",
        output_dir="./power_clustering_results"
    )
    
    # Run full pipeline
    models = pipeline.run_full_pipeline()
    
    # The models dictionary now contains everything needed for deployment
    # Each test case has:
    # - Trained model
    # - Scaler
    # - Feature list
    # - Cluster mappings
    # - Processed data with business states
