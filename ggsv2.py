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
        self.grouping_cols = ['test_case', 'test_run', 'save', 'unit_id', 'station', 'ofp']
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
    
    def analyze_grouping_granularity(self):
        """
        Analyze data to determine optimal grouping level
        Shows data distribution across different grouping combinations
        """
        print("\n" + "="*60)
        print("Grouping Granularity Analysis")
        print("="*60)
        
        # Test different grouping levels
        grouping_levels = [
            ['test_case'],
            ['test_case', 'ofp'],
            ['test_case', 'test_run'],
            ['test_case', 'ofp', 'test_run'],
            ['test_case', 'ofp', 'test_run', 'station'],
            ['test_case', 'test_run', 'save', 'unit_id', 'station', 'ofp']  # Full granularity
        ]
        
        granularity_stats = []
        
        for group_cols in grouping_levels:
            # Check if all columns exist
            existing_cols = [col for col in group_cols if col in self.df.columns]
            if not existing_cols:
                continue
                
            # Group and analyze
            grouped = self.df.groupby(existing_cols).size()
            
            stats = {
                'grouping': ' + '.join(existing_cols),
                'num_groups': len(grouped),
                'avg_points_per_group': grouped.mean(),
                'min_points': grouped.min(),
                'max_points': grouped.max(),
                'std_points': grouped.std(),
                'groups_under_100': (grouped < 100).sum(),
                'groups_under_1000': (grouped < 1000).sum()
            }
            
            granularity_stats.append(stats)
            
            print(f"\nGrouping: {stats['grouping']}")
            print(f"  Number of groups: {stats['num_groups']}")
            print(f"  Avg points per group: {stats['avg_points_per_group']:.0f}")
            print(f"  Range: {stats['min_points']:.0f} - {stats['max_points']:.0f}")
            print(f"  Groups with <100 points: {stats['groups_under_100']}")
            print(f"  Groups with <1000 points: {stats['groups_under_1000']}")
        
        return pd.DataFrame(granularity_stats)
    
    def train_hierarchical_models(self, 
                                 primary_grouping: List[str] = ['test_case', 'ofp'],
                                 min_points_for_model: int = 500,
                                 k_max: int = 5, 
                                 max_shuffles: int = 1,
                                 split_by_dc: bool = False):
        """
        Train models at multiple granularity levels with fallback logic
        
        Args:
            primary_grouping: Primary grouping level for models
            min_points_for_model: Minimum data points needed to train a specific model
            k_max: Maximum clusters for GreedyGaussianSegmenter
            max_shuffles: Maximum shuffles for optimization
            split_by_dc: If True, train separate models for DC1 and DC2
        """
        print(f"\nTraining hierarchical models with primary grouping: {' + '.join(primary_grouping)}")
        if split_by_dc:
            print("SPLITTING BY DC CHANNEL - Training separate models for DC1 and DC2")
        
        # Create primary groups
        existing_primary = [col for col in primary_grouping if col in self.df.columns]
        
        # Store models at different levels
        self.hierarchical_models = {
            'primary': {},
            'fallback': {},
            'test_case': {}  # Always keep test_case level as ultimate fallback
        }
        
        # Track which groups use which model
        self.model_assignments = {}
        
        # If splitting by DC, we'll process each channel separately
        dc_channels = ['dc1', 'dc2'] if split_by_dc else ['combined']
        
        for dc_channel in dc_channels:
            print(f"\n{'='*40}")
            print(f"Processing DC Channel: {dc_channel}")
            print(f"{'='*40}")
            
            # First, train test_case level models as fallback
            if 'test_case' in self.df.columns:
                test_cases = self.df['test_case'].unique()
                for test_case in test_cases:
                    test_data = self.df[self.df['test_case'] == test_case].copy()
                    if len(test_data) >= min_points_for_model:
                        model_key = f"test_{test_case}_{dc_channel}" if split_by_dc else f"test_{test_case}"
                        print(f"\nTraining fallback model: {model_key}")
                        model_info = self._train_single_model_dc(test_data, model_key, dc_channel, k_max, max_shuffles)
                        if model_info:
                            self.hierarchical_models['test_case'][model_key] = model_info
            
            # Train primary level models
            if len(existing_primary) > 0:
                grouped = self.df.groupby(existing_primary)
                
                for group_key, group_data in grouped:
                    group_key_str = '_'.join(map(str, group_key if isinstance(group_key, tuple) else [group_key]))
                    
                    if split_by_dc:
                        group_key_str = f"{group_key_str}_{dc_channel}"
                    
                    if len(group_data) >= min_points_for_model:
                        print(f"\nTraining primary model for: {group_key_str}")
                        print(f"  Data points: {len(group_data)}")
                        
                        model_info = self._train_single_model_dc(group_data.copy(), group_key_str, dc_channel, k_max, max_shuffles)
                        if model_info:
                            self.hierarchical_models['primary'][group_key_str] = model_info
                            self.model_assignments[group_key_str] = 'primary'
                    else:
                        # Not enough data - mark for fallback
                        print(f"\nInsufficient data for: {group_key_str} ({len(group_data)} points)")
                        self.model_assignments[group_key_str] = 'fallback'
        
        print(f"\n" + "="*50)
        print(f"Model Training Summary:")
        print(f"  Primary models trained: {len(self.hierarchical_models['primary'])}")
        print(f"  Test case fallback models: {len(self.hierarchical_models['test_case'])}")
        print(f"  Groups using fallback: {sum(1 for v in self.model_assignments.values() if v == 'fallback')}")
        if split_by_dc:
            print(f"  Total models (with DC split): {len(self.hierarchical_models['primary']) + len(self.hierarchical_models['test_case'])}")
        
        return self.hierarchical_models
    
    def _train_single_model_dc(self, data: pd.DataFrame, identifier: str, dc_channel: str, k_max: int = 5, max_shuffles: int = 1) -> Dict:
        """
        Train a single GreedyGaussianSegmenter model for a specific DC channel
        
        Args:
            data: Data to train on
            identifier: Unique identifier for this model
            dc_channel: 'dc1', 'dc2', or 'combined'
            k_max: Maximum clusters
            max_shuffles: Maximum shuffles for optimization
            
        Returns:
            Dictionary with model info or None if training fails
        """
        try:
            # Get appropriate feature columns based on DC channel
            if dc_channel == 'dc1':
                feature_cols = [col for col in self.get_feature_columns() if 'dc1' in col or 'dc2' not in col]
            elif dc_channel == 'dc2':
                feature_cols = [col for col in self.get_feature_columns() if 'dc2' in col or 'dc1' not in col]
            else:  # combined
                feature_cols = self.get_feature_columns()
            
            # Make sure we have features
            feature_cols = [col for col in feature_cols if col in data.columns]
            if not feature_cols:
                print(f"  No features found for {dc_channel}")
                return None
            
            features = data[feature_cols].values
            
            # Check if we have valid features
            if features.shape[0] < 100:
                print(f"  Skipping - insufficient data points: {features.shape[0]}")
                return None
            
            # Standardize
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train GreedyGaussianSegmenter
            ggs = GreedyGaussianSegmenter(k_max=k_max, max_shuffles=max_shuffles)
            predicted_labels = ggs.fit_predict(features_scaled.T)
            
            # Add predictions to data
            data[f'cluster_{dc_channel}'] = predicted_labels
            data['cluster'] = predicted_labels  # Keep compatible with existing code
            
            # Analyze clusters
            cluster_mapping = self._analyze_clusters_dc(data, identifier, dc_channel)
            
            # Create visualizations
            self._create_test_case_visualizations_dc(data, identifier, dc_channel)
            
            # Store model info
            model_info = {
                'model': ggs,
                'scaler': scaler,
                'features': feature_cols,
                'data': data,
                'cluster_mapping': cluster_mapping,
                'n_clusters': len(np.unique(predicted_labels)),
                'n_points': len(data),
                'identifier': identifier,
                'dc_channel': dc_channel
            }
            
            # Save model
            model_path = self.models_dir / f"model_{identifier}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            print(f"  Successfully trained {dc_channel} model with {model_info['n_clusters']} clusters")
            return model_info
            
        except Exception as e:
            print(f"  Error training model for {identifier}: {str(e)}")
            return None
    
    def _analyze_clusters_dc(self, data: pd.DataFrame, identifier: str, dc_channel: str) -> Dict:
        """
        Analyze clusters for a specific DC channel
        """
        print(f"  Analyzing clusters for {identifier} ({dc_channel})...")
        
        cluster_stats = []
        
        # Use the appropriate voltage column for this DC channel
        if dc_channel == 'dc1':
            voltage_col = 'voltage_28v_dc1_cal'
            slope_col = 'slope_3_V_dc1'
        elif dc_channel == 'dc2':
            voltage_col = 'voltage_28v_dc2_cal'
            slope_col = 'slope_3_V_dc2'
        else:  # combined - use dc1 as primary
            voltage_col = 'voltage_28v_dc1_cal'
            slope_col = 'slope_3_V_dc1'
        
        cluster_col = f'cluster_{dc_channel}' if f'cluster_{dc_channel}' in data.columns else 'cluster'
        
        for cluster in sorted(data[cluster_col].unique()):
            cluster_data = data[data[cluster_col] == cluster]
            
            stats = {
                'cluster': cluster,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'dc_channel': dc_channel
            }
            
            if voltage_col in data.columns:
                stats['avg_voltage'] = cluster_data[voltage_col].mean()
                stats['std_voltage'] = cluster_data[voltage_col].std()
                stats['min_voltage'] = cluster_data[voltage_col].min()
                stats['max_voltage'] = cluster_data[voltage_col].max()
            
            if slope_col in data.columns:
                stats['avg_slope'] = cluster_data[slope_col].mean()
                stats['std_slope'] = cluster_data[slope_col].std()
            
            # Temporal analysis
            positions = cluster_data.index.values if isinstance(cluster_data.index.values[0], int) else np.arange(len(cluster_data))
            relative_positions = positions / len(data)
            stats['avg_position'] = np.mean(relative_positions)
            stats['first_occurrence'] = np.min(relative_positions)
            stats['last_occurrence'] = np.max(relative_positions)
            
            cluster_stats.append(stats)
        
        stats_df = pd.DataFrame(cluster_stats)
        
        # Save statistics with DC channel in filename
        stats_path = self.stats_dir / f"cluster_stats_{identifier}.csv"
        stats_df.to_csv(stats_path, index=False)
        
        # Create descriptive mapping
        mapping = self._create_physical_state_mapping(stats_df)
        
        return {
            'stats': stats_df,
            'descriptive_mapping': mapping
        }
    
    def _create_test_case_visualizations_dc(self, data: pd.DataFrame, identifier: str, dc_channel: str):
        """Create visualizations for a specific DC channel model"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{identifier} - DC Channel: {dc_channel} - Clustering Analysis', fontsize=16)
        
        # Determine which voltage/slope columns to use
        if dc_channel == 'dc1':
            voltage_col = 'voltage_28v_dc1_cal'
            slope_col = 'slope_3_V_dc1'
        elif dc_channel == 'dc2':
            voltage_col = 'voltage_28v_dc2_cal'
            slope_col = 'slope_3_V_dc2'
        else:  # combined
            voltage_col = 'voltage_28v_dc1_cal'
            slope_col = 'slope_3_V_dc1'
        
        cluster_col = f'cluster_{dc_channel}' if f'cluster_{dc_channel}' in data.columns else 'cluster'
        
        # Plot 1: Time series with clusters
        ax = axes[0, 0]
        if voltage_col in data.columns:
            for cluster in sorted(data[cluster_col].unique()):
                cluster_data = data[data[cluster_col] == cluster]
                ax.scatter(range(len(cluster_data)), 
                          cluster_data[voltage_col],
                          label=f'Cluster {cluster}',
                          alpha=0.6, s=1)
            ax.set_xlabel('Time Index')
            ax.set_ylabel(f'Voltage {dc_channel.upper()} (V)')
            ax.set_title(f'{dc_channel.upper()} Voltage Time Series by Cluster')
            ax.legend()
        
        # Plot 2: Slope vs Voltage
        ax = axes[0, 1]
        if voltage_col in data.columns and slope_col in data.columns:
            scatter = ax.scatter(data[voltage_col], 
                                data[slope_col],
                                c=data[cluster_col],
                                cmap='viridis',
                                alpha=0.6)
            ax.set_xlabel(f'Voltage {dc_channel.upper()} (V)')
            ax.set_ylabel(f'Slope {dc_channel.upper()} (V/sample)')
            ax.set_title(f'{dc_channel.upper()} Feature Space Clustering')
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Continue with other plots...
        # (Rest of visualization code similar to original but using appropriate DC columns)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{identifier}_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization for {identifier}")
    
    def _train_single_model(self, data: pd.DataFrame, identifier: str) -> Dict:
        """
        Train a single GreedyGaussianSegmenter model
        
        Returns:
            Dictionary with model info or None if training fails
        """
        try:
            # Get features
            feature_cols = self.get_feature_columns()
            features = data[feature_cols].values
            
            # Check if we have valid features
            if features.shape[0] < 100:
                print(f"  Skipping - insufficient data points: {features.shape[0]}")
                return None
            
            # Standardize
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train GreedyGaussianSegmenter
            ggs = GreedyGaussianSegmenter(k_max=5, max_shuffles=1)
            predicted_labels = ggs.fit_predict(features_scaled.T)
            
            # Add predictions to data
            data_copy = data.copy()
            data_copy['cluster'] = predicted_labels
            
            # Analyze clusters
            cluster_mapping = self._analyze_clusters(data_copy, identifier)
            
            # Create visualizations
            self._create_test_case_visualizations(data_copy, identifier)
            
            # Store model info
            model_info = {
                'model': ggs,
                'scaler': scaler,
                'features': feature_cols,
                'data': data_copy,
                'cluster_mapping': cluster_mapping,
                'n_clusters': len(np.unique(predicted_labels)),
                'n_points': len(data),
                'identifier': identifier
            }
            
            # Save model
            model_path = self.models_dir / f"model_{identifier}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            print(f"  Successfully trained model with {model_info['n_clusters']} clusters")
            return model_info
            
        except Exception as e:
            print(f"  Error training model for {identifier}: {str(e)}")
            return None
    
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
    
    def run_full_pipeline(self, 
                         custom_mappings: Dict = None,
                         analyze_granularity: bool = True,
                         primary_grouping: List[str] = None,
                         min_points_for_model: int = 500,
                         plot_all_groupings: bool = True,
                         split_by_dc: bool = False):
        """
        Execute the complete clustering pipeline with smart granularity handling
        
        Args:
            custom_mappings: Optional dictionary of group -> cluster -> business_state mappings
            analyze_granularity: Whether to analyze and recommend grouping level
            primary_grouping: Grouping columns to use (e.g., ['test_case', 'ofp'])
                            If None, will recommend based on analysis
            min_points_for_model: Minimum points needed to train a dedicated model
            plot_all_groupings: Whether to create plots for every full grouping combination
            split_by_dc: If True, train separate models for DC1 and DC2 channels
        """
        
        # Step 1: Load and prepare data
        print("\n" + "="*60)
        print("STEP 1: Loading and Preparing Data")
        print("="*60)
        self.load_and_prepare_data()
        
        # Step 2: Analyze granularity if requested
        if analyze_granularity:
            print("\n" + "="*60)
            print("STEP 2: Analyzing Grouping Granularity")
            print("="*60)
            granularity_df = self.analyze_grouping_granularity()
            
            # Recommend grouping if not provided
            if primary_grouping is None:
                print("\n" + "-"*40)
                print("RECOMMENDATION:")
                # Find sweet spot - enough groups but not too many tiny ones
                for _, row in granularity_df.iterrows():
                    if row['num_groups'] < 50 and row['avg_points_per_group'] > 1000:
                        primary_grouping = row['grouping'].split(' + ')
                        print(f"Recommended grouping: {row['grouping']}")
                        print(f"This will create {row['num_groups']} models")
                        print(f"Average {row['avg_points_per_group']:.0f} points per model")
                        break
                
                if primary_grouping is None:
                    primary_grouping = ['test_case', 'ofp']
                    print(f"Using default: test_case + ofp")
        
        # Step 3: Train hierarchical models
        print("\n" + "="*60)
        print("STEP 3: Training Hierarchical Models")
        if split_by_dc:
            print("*** SPLITTING BY DC CHANNEL ***")
        print("="*60)
        self.train_hierarchical_models(
            primary_grouping=primary_grouping or ['test_case', 'ofp'],
            min_points_for_model=min_points_for_model,
            split_by_dc=split_by_dc
        )
        
        # Step 4: Apply custom mappings if provided
        if custom_mappings:
            print("\n" + "="*60)
            print("STEP 4: Applying Custom Business Mappings")
            print("="*60)
            self.apply_custom_mappings_hierarchical(custom_mappings)
        else:
            print("\n" + "="*60)
            print("STEP 4: Review Statistics to Create Mappings")
            print("="*60)
            self.create_manual_mapping_interface_hierarchical()
        
        # Step 5: Create multi-level summaries
        print("\n" + "="*60)
        print("STEP 5: Creating Multi-Level Summaries")
        print("="*60)
        self.create_multilevel_summary_hierarchical()
        
        # Step 6: OFP anomaly analysis
        print("\n" + "="*60)
        print("STEP 6: OFP Anomaly Detection")
        print("="*60)
        self.analyze_ofp_anomalies_hierarchical()
        
        # Step 7: Plot all full groupings if requested
        if plot_all_groupings:
            print("\n" + "="*60)
            print("STEP 7: Creating Full Grouping Visualizations")
            if split_by_dc:
                print("*** Including DC Channel Split Visualizations ***")
            print("="*60)
            self.plot_all_groupings_with_clusters()
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.output_dir}")
        if split_by_dc:
            print("Models trained separately for DC1 and DC2 channels")
        print("="*60)
        
        return self.hierarchical_models
    
    def apply_custom_mappings_hierarchical(self, custom_mappings: Dict):
        """Apply custom mappings to hierarchical models"""
        for level in ['primary', 'test_case']:
            for key, model_info in self.hierarchical_models[level].items():
                if key in custom_mappings:
                    data = model_info['data']
                    data['business_state'] = data['cluster'].map(custom_mappings[key])
                    data['business_state'].fillna('unknown', inplace=True)
                    model_info['business_mapping'] = custom_mappings[key]
    
    def create_manual_mapping_interface_hierarchical(self):
        """Show statistics for all models to help with manual mapping"""
        print("\nModel Statistics for Manual Mapping:")
        print("="*50)
        
        all_models = []
        for level in ['primary', 'test_case']:
            for key, model_info in self.hierarchical_models[level].items():
                all_models.append((level, key, model_info))
        
        for level, key, model_info in all_models:
            print(f"\n[{level}] {key}:")
            print(f"  Points: {model_info['n_points']}, Clusters: {model_info['n_clusters']}")
            
            stats_df = model_info['cluster_mapping']['stats']
            for _, row in stats_df.iterrows():
                print(f"    Cluster {row['cluster']}: {row['percentage']:.1f}% | "
                      f"V={row.get('avg_voltage_dc1', 0):.1f} | "
                      f"Slope={row.get('avg_slope_dc1', 0):.3f} | "
                      f"Pos={row.get('avg_position', 0.5)*100:.0f}%")
    
    def create_multilevel_summary_hierarchical(self):
        """Create summaries for hierarchical models"""
        all_summaries = []
        
        for level in ['primary', 'test_case']:
            for key, model_info in self.hierarchical_models[level].items():
                data = model_info['data']
                
                if 'business_state' in data.columns:
                    summary = data.groupby('business_state').agg({
                        'cluster': 'count'
                    }).rename(columns={'cluster': 'count'})
                    summary['model'] = f"{level}_{key}"
                    all_summaries.append(summary)
        
        if all_summaries:
            combined = pd.concat(all_summaries)
            summary_path = self.stats_dir / "hierarchical_model_summary.csv"
            combined.to_csv(summary_path)
    
    def analyze_ofp_anomalies_hierarchical(self):
        """Analyze OFP anomalies using hierarchical models"""
        all_data = []
        
        for level in ['primary', 'test_case']:
            for key, model_info in self.hierarchical_models[level].items():
                data = model_info['data'].copy()
                data['model_level'] = level
                data['model_key'] = key
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates (same data point from different model levels)
            if 'ofp' in combined_data.columns:
                # Keep primary model predictions when available
                dedup_cols = ['ofp', 'test_case', 'test_run'] if all(col in combined_data.columns for col in ['ofp', 'test_case', 'test_run']) else ['ofp']
                combined_data = combined_data.sort_values('model_level').drop_duplicates(subset=dedup_cols)
                
                # Now run the anomaly analysis on deduplicated data
                return self._perform_ofp_anomaly_analysis(combined_data)
            else:
                print("No 'ofp' column found in data")
                return None
    
    def _perform_ofp_anomaly_analysis(self, combined_data: pd.DataFrame):
        """Core OFP anomaly analysis logic"""
        if 'ofp' not in combined_data.columns:
            print("No OFP column found")
            return None
            
        ofp_metrics = []
        
        for ofp in combined_data['ofp'].unique():
            ofp_data = combined_data[combined_data['ofp'] == ofp]
            
            metrics = {
                'ofp': ofp,
                'total_points': len(ofp_data),
                'models_used': ofp_data['model_key'].nunique() if 'model_key' in ofp_data.columns else 1
            }
            
            # Add test case diversity
            if 'test_case' in ofp_data.columns:
                metrics['test_cases'] = ofp_data['test_case'].nunique()
            
            # Calculate voltage metrics
            if 'voltage_28v_dc1_cal' in ofp_data.columns:
                metrics['avg_voltage'] = ofp_data['voltage_28v_dc1_cal'].mean()
                metrics['voltage_std'] = ofp_data['voltage_28v_dc1_cal'].std()
                metrics['voltage_drops'] = (ofp_data['voltage_28v_dc1_cal'] < 25).sum()
                metrics['drop_rate'] = metrics['voltage_drops'] / len(ofp_data) * 100
            
            # Cluster transition metrics
            if 'cluster' in ofp_data.columns:
                transitions = ofp_data['cluster'].diff().ne(0).sum()
                metrics['transitions'] = transitions
                metrics['transition_rate'] = transitions / len(ofp_data) * 100
            
            ofp_metrics.append(metrics)
        
        ofp_df = pd.DataFrame(ofp_metrics)
        
        # Calculate anomaly scores
        if len(ofp_df) > 1:
            for col in ['voltage_std', 'drop_rate', 'transition_rate']:
                if col in ofp_df.columns:
                    ofp_df[f'{col}_zscore'] = (ofp_df[col] - ofp_df[col].mean()) / ofp_df[col].std()
            
            zscore_cols = [col for col in ofp_df.columns if col.endswith('_zscore')]
            if zscore_cols:
                ofp_df['anomaly_score'] = ofp_df[zscore_cols].mean(axis=1)
                ofp_df = ofp_df.sort_values('anomaly_score', ascending=False)
        
        # Save results
        ofp_path = self.stats_dir / "ofp_anomaly_analysis.csv"
        ofp_df.to_csv(ofp_path, index=False)
        
        print("\nTop 5 Problematic OFPs:")
        if 'anomaly_score' in ofp_df.columns:
            print(ofp_df.head()[['ofp', 'anomaly_score', 'voltage_std', 'drop_rate', 'transition_rate']])
        else:
            print(ofp_df.head())
        
        # Create OFP comparison visualizations
        self._create_ofp_comparison_plots(ofp_df, combined_data)
        
        return ofp_df
    
    def plot_all_groupings_with_clusters(self):
        """
        Create comprehensive plots for EVERY full grouping combination showing their clusters
        This helps visualize how well the clustering is working at the most granular level
        """
        print("\n" + "="*60)
        print("Creating Full Grouping Cluster Visualizations")
        print("="*60)
        
        # Get the full granularity grouping
        full_grouping_cols = [col for col in ['test_case', 'test_run', 'save', 'unit_id', 'station', 'ofp'] 
                             if col in self.df.columns]
        
        if not full_grouping_cols:
            print("No grouping columns found")
            return
        
        # Create a subdirectory for full grouping plots
        full_plots_dir = self.plots_dir / "full_groupings"
        full_plots_dir.mkdir(exist_ok=True)
        
        # Group by full granularity
        grouped = self.df.groupby(full_grouping_cols)
        
        print(f"Creating plots for {len(grouped)} unique groupings...")
        print(f"Grouping by: {' + '.join(full_grouping_cols)}")
        
        # Create summary statistics
        grouping_summary = []
        
        for idx, (group_key, group_data) in enumerate(grouped):
            if isinstance(group_key, tuple):
                group_dict = dict(zip(full_grouping_cols, group_key))
            else:
                group_dict = {full_grouping_cols[0]: group_key}
            
            # Create a meaningful filename
            group_str = '_'.join([f"{k}={v}" for k, v in group_dict.items()])
            group_str = group_str.replace('/', '_').replace('\\', '_')[:100]  # Limit length and sanitize
            
            # Find which model was used for this grouping
            model_used = None
            cluster_data = None
            
            # Check if this specific grouping has a model
            for level in ['primary', 'test_case']:
                if level in self.hierarchical_models:
                    for model_key, model_info in self.hierarchical_models[level].items():
                        # Check if this model's data matches our grouping
                        model_data = model_info['data']
                        
                        # Create a mask for matching rows
                        mask = pd.Series([True] * len(model_data))
                        for col, val in group_dict.items():
                            if col in model_data.columns:
                                mask &= (model_data[col] == val)
                        
                        if mask.any():
                            cluster_data = model_data[mask].copy()
                            model_used = f"{level}/{model_key}"
                            break
                    if cluster_data is not None:
                        break
            
            # If no cluster data found, skip
            if cluster_data is None or 'cluster' not in cluster_data.columns:
                print(f"  Skipping {group_str[:50]}... - no cluster data")
                continue
            
            # Prepare data for plotting
            n_points = len(cluster_data)
            n_clusters = cluster_data['cluster'].nunique()
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            
            # Add title with grouping information
            title_text = f"Full Grouping Analysis\n"
            for k, v in group_dict.items():
                title_text += f"{k}: {v} | "
            title_text = title_text[:-3]  # Remove last separator
            title_text += f"\nModel: {model_used} | Points: {n_points} | Clusters: {n_clusters}"
            fig.suptitle(title_text, fontsize=12, y=0.98)
            
            # Create 6 subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Voltage time series colored by cluster
            ax1 = fig.add_subplot(gs[0, :])
            if 'voltage_28v_dc1_cal' in cluster_data.columns:
                for cluster in sorted(cluster_data['cluster'].unique()):
                    cluster_mask = cluster_data['cluster'] == cluster
                    indices = np.where(cluster_mask)[0]
                    ax1.scatter(indices, 
                              cluster_data.loc[cluster_mask, 'voltage_28v_dc1_cal'],
                              label=f'Cluster {cluster}',
                              alpha=0.6, s=2)
                ax1.set_xlabel('Time Index')
                ax1.set_ylabel('Voltage DC1 (V)')
                ax1.set_title('Voltage Time Series by Cluster')
                ax1.legend(loc='upper right', ncol=n_clusters)
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cluster sequence
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(cluster_data['cluster'].values, linewidth=1, color='darkblue')
            ax2.fill_between(range(len(cluster_data)), 
                            cluster_data['cluster'].values, 
                            alpha=0.3)
            ax2.set_xlabel('Time Index')
            ax2.set_ylabel('Cluster ID')
            ax2.set_title('Cluster Transitions Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Voltage vs Slope scatter
            ax3 = fig.add_subplot(gs[1, 1])
            if 'voltage_28v_dc1_cal' in cluster_data.columns and 'slope_3_V_dc1' in cluster_data.columns:
                scatter = ax3.scatter(cluster_data['voltage_28v_dc1_cal'], 
                                    cluster_data['slope_3_V_dc1'],
                                    c=cluster_data['cluster'],
                                    cmap='viridis',
                                    alpha=0.6, s=10)
                ax3.set_xlabel('Voltage DC1 (V)')
                ax3.set_ylabel('Slope DC1')
                ax3.set_title('Feature Space Clustering')
                plt.colorbar(scatter, ax=ax3, label='Cluster')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Cluster distribution pie chart
            ax4 = fig.add_subplot(gs[1, 2])
            cluster_counts = cluster_data['cluster'].value_counts().sort_index()
            colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
            wedges, texts, autotexts = ax4.pie(cluster_counts.values, 
                                               labels=[f'C{i}' for i in cluster_counts.index],
                                               colors=colors,
                                               autopct='%1.1f%%',
                                               startangle=90)
            ax4.set_title('Cluster Distribution')
            
            # Plot 5: DC1 vs DC2 if both exist
            ax5 = fig.add_subplot(gs[2, 0])
            if 'voltage_28v_dc1_cal' in cluster_data.columns and 'voltage_28v_dc2_cal' in cluster_data.columns:
                scatter = ax5.scatter(cluster_data['voltage_28v_dc1_cal'],
                                    cluster_data['voltage_28v_dc2_cal'],
                                    c=cluster_data['cluster'],
                                    cmap='viridis',
                                    alpha=0.6, s=10)
                ax5.set_xlabel('Voltage DC1 (V)')
                ax5.set_ylabel('Voltage DC2 (V)')
                ax5.set_title('DC1 vs DC2 Correlation')
                ax5.plot([0, 30], [0, 30], 'r--', alpha=0.3, label='y=x')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # Plot 6: Cluster statistics table
            ax6 = fig.add_subplot(gs[2, 1:])
            ax6.axis('tight')
            ax6.axis('off')
            
            # Calculate statistics for each cluster
            cluster_stats = []
            for cluster in sorted(cluster_data['cluster'].unique()):
                c_data = cluster_data[cluster_data['cluster'] == cluster]
                stats = [
                    f"C{cluster}",
                    f"{len(c_data)}",
                    f"{len(c_data)/len(cluster_data)*100:.1f}%"
                ]
                
                if 'voltage_28v_dc1_cal' in c_data.columns:
                    stats.append(f"{c_data['voltage_28v_dc1_cal'].mean():.1f}V")
                    stats.append(f"±{c_data['voltage_28v_dc1_cal'].std():.1f}")
                
                if 'slope_3_V_dc1' in c_data.columns:
                    stats.append(f"{c_data['slope_3_V_dc1'].mean():.3f}")
                
                cluster_stats.append(stats)
            
            # Create table
            columns = ['Cluster', 'Points', '%', 'Avg V', 'Std V', 'Avg Slope']
            table = ax6.table(cellText=cluster_stats,
                            colLabels=columns,
                            cellLoc='center',
                            loc='center',
                            colColours=['lightgray']*len(columns))
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            ax6.set_title('Cluster Statistics Summary', pad=20)
            
            # Save figure
            plot_path = full_plots_dir / f"full_group_{idx:04d}_{group_str[:50]}.png"
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Add to summary
            grouping_summary.append({
                'index': idx,
                'filename': plot_path.name,
                **group_dict,
                'n_points': n_points,
                'n_clusters': n_clusters,
                'model_used': model_used
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Created {idx + 1}/{len(grouped)} plots...")
        
        # Save summary CSV for reference
        summary_df = pd.DataFrame(grouping_summary)
        summary_path = full_plots_dir / "grouping_plot_index.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nCreated {len(grouping_summary)} full grouping plots")
        print(f"Plots saved in: {full_plots_dir}")
        print(f"Index saved as: {summary_path}")
        
        # Create a master summary plot showing all groupings
        self._create_grouping_overview_plot(summary_df)
        
        return summary_df
    
    def _create_grouping_overview_plot(self, summary_df: pd.DataFrame):
        """
        Create a master overview plot showing statistics across all groupings
        """
        if len(summary_df) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Overview of All Full Groupings', fontsize=16)
        
        # Plot 1: Distribution of data points per grouping
        ax = axes[0, 0]
        ax.hist(summary_df['n_points'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Data Points')
        ax.set_ylabel('Number of Groupings')
        ax.set_title('Data Points Distribution Across Groupings')
        ax.axvline(summary_df['n_points'].median(), color='red', linestyle='--', 
                  label=f'Median: {summary_df["n_points"].median():.0f}')
        ax.legend()
        
        # Plot 2: Number of clusters distribution
        ax = axes[0, 1]
        cluster_counts = summary_df['n_clusters'].value_counts().sort_index()
        ax.bar(cluster_counts.index, cluster_counts.values)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Number of Groupings')
        ax.set_title('Cluster Count Distribution')
        ax.set_xticks(cluster_counts.index)
        
        # Plot 3: Points vs Clusters scatter
        ax = axes[1, 0]
        ax.scatter(summary_df['n_points'], summary_df['n_clusters'], alpha=0.6)
        ax.set_xlabel('Number of Data Points')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Relationship: Data Points vs Clusters Found')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=6)
        
        # Plot 4: Groupings by test case
        ax = axes[1, 1]
        if 'test_case' in summary_df.columns:
            test_case_counts = summary_df['test_case'].value_counts()
            ax.barh(range(len(test_case_counts)), test_case_counts.values)
            ax.set_yticks(range(len(test_case_counts)))
            ax.set_yticklabels(test_case_counts.index)
            ax.set_xlabel('Number of Groupings')
            ax.set_title('Groupings per Test Case')
        
        plt.tight_layout()
        overview_path = self.plots_dir / "full_groupings_overview.png"
        plt.savefig(overview_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved overview plot: {overview_path}")
        """
        Create visualizations comparing OFP performance
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('OFP Performance Comparison', fontsize=16)
        
        # Plot 1: Anomaly scores by OFP
        ax = axes[0, 0]
        if 'anomaly_score' in ofp_df.columns:
            ofp_sorted = ofp_df.sort_values('anomaly_score', ascending=False).head(10)
            ax.barh(range(len(ofp_sorted)), ofp_sorted['anomaly_score'])
            ax.set_yticks(range(len(ofp_sorted)))
            ax.set_yticklabels(ofp_sorted['ofp'])
            ax.set_xlabel('Anomaly Score (Higher = More Issues)')
            ax.set_title('Top 10 OFPs by Anomaly Score')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 2: Voltage stability comparison
        ax = axes[0, 1]
        if 'voltage_std' in ofp_df.columns:
            ofp_sorted = ofp_df.sort_values('voltage_std', ascending=False).head(10)
            ax.barh(range(len(ofp_sorted)), ofp_sorted['voltage_std'])
            ax.set_yticks(range(len(ofp_sorted)))
            ax.set_yticklabels(ofp_sorted['ofp'])
            ax.set_xlabel('Voltage Stability (Std Dev)')
            ax.set_title('OFPs with Highest Voltage Instability')
        
        # Plot 3: State distribution by OFP
        ax = axes[1, 0]
        if 'business_state' in combined_data.columns:
            top_ofps = ofp_df.head(5)['ofp'].tolist() if len(ofp_df) > 5 else ofp_df['ofp'].tolist()
            state_dist = combined_data[combined_data['ofp'].isin(top_ofps)].groupby(['ofp', 'business_state']).size().unstack(fill_value=0)
            state_dist_pct = state_dist.div(state_dist.sum(axis=1), axis=0) * 100
            state_dist_pct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_xlabel('OFP')
            ax.set_ylabel('State Distribution (%)')
            ax.set_title('Power State Distribution by OFP')
            ax.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Voltage drop frequency
        ax = axes[1, 1]
        if 'drop_rate' in ofp_df.columns and 'test_cases' in ofp_df.columns:
            ax.scatter(ofp_df['test_cases'], ofp_df['drop_rate'], 
                      s=ofp_df['total_points']/100, alpha=0.6)
            ax.set_xlabel('Number of Test Cases')
            ax.set_ylabel('Voltage Drop Rate (%)')
            ax.set_title('Voltage Drop Rate vs Test Diversity\n(bubble size = data points)')
            
            # Annotate outliers
            for _, row in ofp_df.iterrows():
                if row['drop_rate'] > ofp_df['drop_rate'].mean() + 2*ofp_df['drop_rate'].std():
                    ax.annotate(row['ofp'], (row['test_cases'], row['drop_rate']),
                              fontsize=8, alpha=0.7)
        
        # Plot 5: Transition rate comparison
        ax = axes[2, 0]
        if 'transition_rate' in ofp_df.columns:
            ofp_sorted = ofp_df.sort_values('transition_rate', ascending=False).head(10)
            ax.bar(range(len(ofp_sorted)), ofp_sorted['transition_rate'])
            ax.set_xticks(range(len(ofp_sorted)))
            ax.set_xticklabels(ofp_sorted['ofp'], rotation=45, ha='right')
            ax.set_ylabel('Transition Rate (%)')
            ax.set_title('State Transition Frequency (Higher = Less Stable)')
        
        # Plot 6: Voltage profile of worst OFP
        ax = axes[2, 1]
        if 'voltage_28v_dc1_cal' in combined_data.columns and 'anomaly_score' in ofp_df.columns:
            worst_ofp = ofp_df.iloc[0]['ofp']
            worst_data = combined_data[combined_data['ofp'] == worst_ofp]
            
            if len(worst_data) > 0:
                # Sample if too many points
                if len(worst_data) > 5000:
                    worst_data = worst_data.sample(5000)
                
                ax.plot(worst_data['voltage_28v_dc1_cal'].values, alpha=0.7, linewidth=0.5)
                ax.set_xlabel('Time Index')
                ax.set_ylabel('Voltage (V)')
                ax.set_title(f'Voltage Profile: OFP {worst_ofp}\n(Most Problematic)')
                ax.axhline(y=28, color='g', linestyle='--', alpha=0.5, label='Target')
                ax.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Low Threshold')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'ofp_comparison_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved OFP comparison plots")
    
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
