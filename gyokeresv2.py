import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from typing import Optional, List, Dict, Tuple
warnings.filterwarnings('ignore')

# =============================================================================
# NEW FORMAT DATA LOADER
# =============================================================================
class NewFormatDataLoader:
    """Loader for the new parquet format with different column structure"""
    
    def __init__(self, filepath: str, model_pkl_path: str, window_size: int = 100, stride: int = 20):
        """
        Args:
            filepath: Path to new format parquet file
            model_pkl_path: Path to saved model pickle file
            window_size: Window size (should match trained model)
            stride: Stride for sliding window
        """
        self.filepath = filepath
        self.window_size = window_size
        self.stride = stride
        
        # Load the saved model package
        print(f"Loading model from {model_pkl_path}...")
        with open(model_pkl_path, 'rb') as f:
            self.model_package = pickle.load(f)
        
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.label_encoder = self.model_package['label_encoder']
        
        # New format group columns
        self.group_cols = ['Scenario', 'Dam', 'Test', 'DC', 'Zip']
        
        # Load parquet file
        self.parquet_file = pq.ParquetFile(filepath)
        self.schema = self.parquet_file.schema_arrow
        
        print(f"Loaded parquet with columns: {self.schema.names}")
        
    def load_and_reshape_data(self) -> pd.DataFrame:
        """Load and reshape the new format data to match expected structure"""
        print("\nLoading and reshaping data...")
        
        # Read the entire parquet file (or you can read by row groups if it's large)
        df = self.parquet_file.read().to_pandas()
        
        print(f"Loaded {len(df):,} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"DC values: {df['DC'].unique() if 'DC' in df.columns else 'No DC column'}")
        print(f"Measurement types: {df['Measurement'].unique() if 'Measurement' in df.columns else 'No Measurement column'}")
        
        return df
    
    def process_groups(self, df: pd.DataFrame, max_samples: Optional[int] = None, 
                      voltage_only: bool = False):
        """Process data by groups and extract features
        
        Args:
            df: DataFrame with the data
            max_samples: Maximum number of samples to process
            voltage_only: If True, use zeros for current-related features
        """
        
        # Check if we have current data
        if 'Measurement' in df.columns:
            has_current = 'Current' in df['Measurement'].unique()
            if not has_current:
                print("WARNING: No current data found in file")
                print("Using ZEROS for current and power features - accuracy may be reduced")
                voltage_only = True
        
        # Create group identifier
        if all(col in df.columns for col in self.group_cols):
            df['group_id'] = df[self.group_cols].apply(
                lambda x: '_'.join(str(v) for v in x), axis=1
            )
        else:
            # Use available columns
            available_group_cols = [col for col in self.group_cols if col in df.columns]
            if available_group_cols:
                df['group_id'] = df[available_group_cols].apply(
                    lambda x: '_'.join(str(v) for v in x), axis=1
                )
            else:
                df['group_id'] = 'all'
        
        unique_groups = df['group_id'].unique()
        print(f"\nFound {len(unique_groups)} unique groups")
        if voltage_only:
            print("Running in VOLTAGE-ONLY mode - current/power features will be zeros")
        
        all_features = []
        all_labels = []  # Will be None since we don't have labels in test data
        all_group_ids = []
        all_dc_systems = []
        all_timestamps = []
        
        samples_collected = 0
        
        # Process each group
        for group_idx, group_id in enumerate(unique_groups):
            group_data = df[df['group_id'] == group_id].copy()
            
            # Sort by Timestamp
            if 'Timestamp' in group_data.columns:
                group_data = group_data.sort_values('Timestamp')
            
            # Process DC1 and DC2 separately
            for dc_num in [1, 2]:
                dc_str = f'DC{dc_num}'
                
                # Filter for this DC
                if 'DC' in group_data.columns:
                    dc_data = group_data[group_data['DC'] == dc_str].copy()
                else:
                    # If no DC column, assume it's for both DCs
                    dc_data = group_data.copy()
                
                if len(dc_data) == 0:
                    continue
                
                # Get voltage data
                if 'Measurement' in dc_data.columns:
                    voltage_data = dc_data[dc_data['Measurement'] == 'Voltage'].copy()
                else:
                    # Assume all data is voltage if no Measurement column
                    voltage_data = dc_data.copy()
                
                if voltage_data.empty:
                    continue
                
                # Process voltage data
                if 'Timestamp' in voltage_data.columns:
                    voltage_data = voltage_data.set_index('Timestamp')['Value'].sort_index()
                    timestamps = voltage_data.index.values
                    voltage = voltage_data.values
                else:
                    voltage = voltage_data['Value'].values if 'Value' in voltage_data.columns else voltage_data.values.flatten()
                    timestamps = np.arange(len(voltage))
                
                if len(voltage) < self.window_size:
                    continue
                
                # Handle current data
                if voltage_only:
                    # Use zeros for current when not available
                    current = np.zeros_like(voltage)
                else:
                    # Try to get actual current data
                    current_data = dc_data[dc_data['Measurement'] == 'Current'].copy() if 'Measurement' in dc_data.columns else None
                    
                    if current_data is None or current_data.empty:
                        # No current data available
                        current = np.zeros_like(voltage)
                    else:
                        # Align voltage and current by timestamp
                        if 'Timestamp' in current_data.columns:
                            current_data = current_data.set_index('Timestamp')['Value'].sort_index()
                            
                            # Get common timestamps
                            common_timestamps = voltage_data.index.intersection(current_data.index)
                            
                            if len(common_timestamps) < self.window_size:
                                current = np.zeros_like(voltage)
                            else:
                                voltage = voltage_data.loc[common_timestamps].values
                                current = current_data.loc[common_timestamps].values
                                timestamps = common_timestamps.values
                        else:
                            # No timestamp, use values in order
                            current = current_data['Value'].values if 'Value' in current_data.columns else current_data.values.flatten()
                            if len(current) != len(voltage):
                                current = np.zeros_like(voltage)
                
                # Extract windows
                features, window_timestamps = self._extract_windows(voltage, current, dc_num, timestamps)
                
                if features:
                    all_features.extend(features)
                    all_group_ids.extend([group_id] * len(features))
                    all_dc_systems.extend([dc_num] * len(features))
                    all_timestamps.extend(window_timestamps)
                    samples_collected += len(features)
            
            if group_idx % 10 == 0:
                print(f"  Processed {group_idx}/{len(unique_groups)} groups, {samples_collected:,} samples")
            
            if max_samples and samples_collected >= max_samples:
                break
        
        if not all_features:
            raise ValueError("No valid data found for feature extraction")
        
        X = np.array(all_features, dtype=np.float32)
        group_ids = np.array(all_group_ids)
        dc_systems = np.array(all_dc_systems)
        timestamps = np.array(all_timestamps)
        
        print(f"\nExtracted {len(X):,} samples")
        print(f"  DC1 samples: {(dc_systems == 1).sum():,}")
        print(f"  DC2 samples: {(dc_systems == 2).sum():,}")
        
        if voltage_only:
            print("\n⚠️  IMPORTANT: Running with voltage data only!")
            print("  - Current features are set to zero")
            print("  - Power features are set to zero")
            print("  - Model accuracy may be significantly reduced")
            print("  - Consider this when interpreting results")
        
        return X, group_ids, dc_systems, timestamps
    
    def _extract_windows(self, voltage, current, dc_num, timestamps):
        """Extract feature windows from aligned voltage/current data"""
        if len(voltage) < self.window_size:
            return [], []
        
        features = []
        window_timestamps = []
        
        # Pre-compute power (will be zero if current is zero)
        power = voltage * current
        
        for i in range(0, len(voltage) - self.window_size + 1, self.stride):
            end_idx = i + self.window_size
            
            v_window = voltage[i:end_idx]
            c_window = current[i:end_idx]
            p_window = power[i:end_idx]
            
            # Skip if NaN in voltage (current can be zero)
            if np.any(np.isnan(v_window)):
                continue
            
            # Compute features (matching trained model structure)
            feature_vec = self._compute_features_with_dc(v_window, c_window, p_window, dc_num)
            features.append(feature_vec)
            
            # Store the timestamp of the last sample in window
            window_timestamps.append(timestamps[end_idx - 1])
        
        return features, window_timestamps
    
    def _compute_features_with_dc(self, voltage, current, power, dc_num):
        """Compute features matching the trained model's feature structure"""
        features = np.zeros(44, dtype=np.float32)
        idx = 0
        
        # DC system indicator (0 for DC1, 1 for DC2) - MUST match training
        features[idx] = dc_num - 1
        idx += 1
        
        for signal in [voltage, current, power]:
            # Basic stats (8 features per signal)
            features[idx:idx+8] = [
                signal.mean(),
                signal.std(),
                signal.min(),
                signal.max(),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                signal[-1] - signal[0]
            ]
            idx += 8
            
            # Differential features (6 features per signal)
            diff = np.diff(signal)
            features[idx] = diff.mean()
            features[idx+1] = diff.std()
            features[idx+2] = np.abs(diff).max()
            features[idx+3] = (signal[-1] - signal[0]) / len(signal)
            features[idx+4] = np.abs(diff).sum()
            features[idx+5] = np.sum(np.diff(np.sign(diff)) != 0)
            idx += 6
        
        # Correlation
        features[idx] = np.corrcoef(voltage, current)[0, 1] if len(voltage) > 1 else 0
        
        return features

# =============================================================================
# MODEL TESTER AND VISUALIZER
# =============================================================================
class ModelTester:
    """Test the saved model on new data and visualize results"""
    
    def __init__(self, loader: NewFormatDataLoader):
        self.loader = loader
        self.predictions = None
        self.probabilities = None
        
    def test_model(self, X, group_ids, dc_systems, timestamps):
        """Apply the loaded model to new data"""
        print("\n" + "="*60)
        print("TESTING SAVED MODEL ON NEW DATA")
        print("="*60)
        
        # Scale features using the saved scaler
        X_scaled = self.loader.scaler.transform(X)
        
        # Make predictions
        print("Making predictions...")
        self.predictions_encoded = self.loader.model.predict(X_scaled)
        self.probabilities = self.loader.model.predict_proba(X_scaled)
        
        # Decode predictions
        self.predictions = self.loader.label_encoder.inverse_transform(self.predictions_encoded)
        
        print(f"Predictions complete for {len(self.predictions):,} samples")
        
        # Get prediction statistics
        unique_preds, counts = np.unique(self.predictions, return_counts=True)
        print("\nPrediction Distribution:")
        for pred, count in zip(unique_preds, counts):
            print(f"  {pred}: {count:,} ({count/len(self.predictions)*100:.1f}%)")
        
        # Store results
        self.results = {
            'X': X,
            'X_scaled': X_scaled,
            'predictions': self.predictions,
            'predictions_encoded': self.predictions_encoded,
            'probabilities': self.probabilities,
            'group_ids': group_ids,
            'dc_systems': dc_systems,
            'timestamps': timestamps
        }
        
        return self.results
    
    def visualize_predictions(self, num_groups_to_plot: int = 5, use_plotly: bool = False):
        """Create visualizations of predictions for each group"""
        print("\nGenerating prediction visualizations...")
        
        import os
        os.makedirs('prediction_results', exist_ok=True)
        
        unique_groups = np.unique(self.results['group_ids'])
        print(f"Total unique groups found: {len(unique_groups)}")
        
        # Determine how many groups to actually plot
        groups_to_plot_count = min(num_groups_to_plot, len(unique_groups))
        groups_to_plot = unique_groups[:groups_to_plot_count]
        
        print(f"Creating visualizations for {groups_to_plot_count} groups...")
        
        # Plot each group
        for i, group_id in enumerate(groups_to_plot):
            print(f"  Processing group {i+1}/{groups_to_plot_count}: {group_id}")
            self._plot_group_predictions(group_id)
            if use_plotly:
                self._create_plotly_timeline(group_id)
        
        # Generate simplified overall summary
        self._plot_simplified_summary()
        
        print(f"\nVisualization complete! Created {groups_to_plot_count} group plots.")
    
    def _plot_group_predictions(self, group_id):
        """Plot predictions for a specific group with color-coded states"""
        group_mask = self.results['group_ids'] == group_id
        
        if not group_mask.any():
            return
        
        group_preds = self.results['predictions'][group_mask]
        group_dc = self.results['dc_systems'][group_mask]
        group_timestamps = self.results['timestamps'][group_mask]
        
        # Define color mapping for states
        def get_color(pred_label):
            """Map prediction labels to colors based on state"""
            pred_lower = pred_label.lower()
            if 'de-energized' in pred_lower or 'deenergized' in pred_lower or 'off' in pred_lower:
                return 'blue'
            elif 'stabilizing' in pred_lower or 'stabilize' in pred_lower or 'transition' in pred_lower:
                return 'orange'
            elif 'steady' in pred_lower or 'stable' in pred_lower or 'on' in pred_lower or 'normal' in pred_lower:
                return 'green'
            else:
                # Default color for unknown states
                return 'gray'
        
        # Create simple figure with 2 subplots (DC1 and DC2)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Model Predictions for Group: {group_id}', fontsize=14, fontweight='bold')
        
        # Process DC1
        dc1_mask = group_dc == 1
        if dc1_mask.any():
            ax = axes[0]
            dc1_preds = group_preds[dc1_mask]
            dc1_timestamps = np.arange(len(dc1_preds))  # Use indices if timestamps not available
            
            # Plot colored segments based on state
            for i in range(len(dc1_preds)):
                color = get_color(dc1_preds[i])
                if i == 0:
                    ax.scatter(dc1_timestamps[i], 0, c=color, s=50, alpha=0.8, 
                             label=dc1_preds[i].replace('DC1_', ''))
                else:
                    # Only add label if it's a new state
                    if dc1_preds[i] != dc1_preds[i-1]:
                        ax.scatter(dc1_timestamps[i], 0, c=color, s=50, alpha=0.8,
                                 label=dc1_preds[i].replace('DC1_', ''))
                    else:
                        ax.scatter(dc1_timestamps[i], 0, c=color, s=50, alpha=0.8)
            
            # Create continuous line with color segments
            for i in range(len(dc1_preds)):
                color = get_color(dc1_preds[i])
                if i < len(dc1_preds) - 1:
                    ax.plot([dc1_timestamps[i], dc1_timestamps[i+1]], [0, 0], 
                           color=color, linewidth=8, alpha=0.6)
            
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlim(-1, len(dc1_preds))
            ax.set_xlabel('Time Step')
            ax.set_title('DC1 System State', fontsize=12)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add legend with unique states
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', 
                     fontsize=9, ncol=min(3, len(by_label)))
            
            # Add state summary text
            unique_states = np.unique(dc1_preds)
            state_counts = {state: np.sum(dc1_preds == state) for state in unique_states}
            summary_text = "States: " + ", ".join([f"{s.replace('DC1_', '')}: {c}" 
                                                  for s, c in state_counts.items()])
            ax.text(0.02, 0.85, summary_text, transform=ax.transAxes, 
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[0].text(0.5, 0.5, 'No DC1 Data', ha='center', va='center', 
                        transform=axes[0].transAxes, fontsize=12)
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
        
        # Process DC2
        dc2_mask = group_dc == 2
        if dc2_mask.any():
            ax = axes[1]
            dc2_preds = group_preds[dc2_mask]
            dc2_timestamps = np.arange(len(dc2_preds))
            
            # Plot colored segments based on state
            for i in range(len(dc2_preds)):
                color = get_color(dc2_preds[i])
                if i == 0:
                    ax.scatter(dc2_timestamps[i], 0, c=color, s=50, alpha=0.8,
                             label=dc2_preds[i].replace('DC2_', ''))
                else:
                    # Only add label if it's a new state
                    if dc2_preds[i] != dc2_preds[i-1]:
                        ax.scatter(dc2_timestamps[i], 0, c=color, s=50, alpha=0.8,
                                 label=dc2_preds[i].replace('DC2_', ''))
                    else:
                        ax.scatter(dc2_timestamps[i], 0, c=color, s=50, alpha=0.8)
            
            # Create continuous line with color segments
            for i in range(len(dc2_preds)):
                color = get_color(dc2_preds[i])
                if i < len(dc2_preds) - 1:
                    ax.plot([dc2_timestamps[i], dc2_timestamps[i+1]], [0, 0], 
                           color=color, linewidth=8, alpha=0.6)
            
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlim(-1, len(dc2_preds))
            ax.set_xlabel('Time Step')
            ax.set_title('DC2 System State', fontsize=12)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add legend with unique states
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right',
                     fontsize=9, ncol=min(3, len(by_label)))
            
            # Add state summary text
            unique_states = np.unique(dc2_preds)
            state_counts = {state: np.sum(dc2_preds == state) for state in unique_states}
            summary_text = "States: " + ", ".join([f"{s.replace('DC2_', '')}: {c}" 
                                                  for s, c in state_counts.items()])
            ax.text(0.02, 0.85, summary_text, transform=ax.transAxes,
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1].text(0.5, 0.5, 'No DC2 Data', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save figure
        safe_group_id = str(group_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(f'prediction_results/predictions_{safe_group_id}.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print(f"  Plotted group: {group_id}")
        
    def _create_plotly_timeline(self, group_id):
        """Create an interactive Plotly timeline visualization"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        group_mask = self.results['group_ids'] == group_id
        if not group_mask.any():
            return
        
        group_preds = self.results['predictions'][group_mask]
        group_dc = self.results['dc_systems'][group_mask]
        
        # Define color mapping
        def get_color_plotly(pred_label):
            pred_lower = pred_label.lower()
            if 'de-energized' in pred_lower or 'deenergized' in pred_lower or 'off' in pred_lower:
                return 'blue'
            elif 'stabilizing' in pred_lower or 'stabilize' in pred_lower or 'transition' in pred_lower:
                return 'orange'
            elif 'steady' in pred_lower or 'stable' in pred_lower or 'on' in pred_lower or 'normal' in pred_lower:
                return 'green'
            else:
                return 'gray'
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('DC1 System State', 'DC2 System State'),
            vertical_spacing=0.15
        )
        
        # Process DC1
        dc1_mask = group_dc == 1
        if dc1_mask.any():
            dc1_preds = group_preds[dc1_mask]
            dc1_x = list(range(len(dc1_preds)))
            dc1_colors = [get_color_plotly(pred) for pred in dc1_preds]
            dc1_labels = [pred.replace('DC1_', '') for pred in dc1_preds]
            
            # Create scatter plot with colored markers
            fig.add_trace(
                go.Scatter(
                    x=dc1_x,
                    y=[1] * len(dc1_x),
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=dc1_colors,
                        line=dict(width=1, color='darkgray')
                    ),
                    line=dict(color='lightgray', width=2),
                    text=dc1_labels,
                    hovertemplate='Step: %{x}<br>State: %{text}<extra></extra>',
                    name='DC1',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add state change annotations
            for i in range(1, len(dc1_preds)):
                if dc1_preds[i] != dc1_preds[i-1]:
                    fig.add_annotation(
                        x=i, y=1,
                        text=dc1_labels[i],
                        showarrow=True,
                        arrowhead=2,
                        ax=0, ay=-30,
                        row=1, col=1,
                        font=dict(size=9)
                    )
        
        # Process DC2
        dc2_mask = group_dc == 2
        if dc2_mask.any():
            dc2_preds = group_preds[dc2_mask]
            dc2_x = list(range(len(dc2_preds)))
            dc2_colors = [get_color_plotly(pred) for pred in dc2_preds]
            dc2_labels = [pred.replace('DC2_', '') for pred in dc2_preds]
            
            fig.add_trace(
                go.Scatter(
                    x=dc2_x,
                    y=[1] * len(dc2_x),
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=dc2_colors,
                        line=dict(width=1, color='darkgray')
                    ),
                    line=dict(color='lightgray', width=2),
                    text=dc2_labels,
                    hovertemplate='Step: %{x}<br>State: %{text}<extra></extra>',
                    name='DC2',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add state change annotations
            for i in range(1, len(dc2_preds)):
                if dc2_preds[i] != dc2_preds[i-1]:
                    fig.add_annotation(
                        x=i, y=1,
                        text=dc2_labels[i],
                        showarrow=True,
                        arrowhead=2,
                        ax=0, ay=-30,
                        row=2, col=1,
                        font=dict(size=9)
                    )
        
        # Update layout
        fig.update_layout(
            title=f'Model Predictions Timeline - Group: {group_id}',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time Step", row=1, col=1, showgrid=True)
        fig.update_xaxes(title_text="Time Step", row=2, col=1, showgrid=True)
        fig.update_yaxes(showticklabels=False, row=1, col=1, range=[0.5, 1.5])
        fig.update_yaxes(showticklabels=False, row=2, col=1, range=[0.5, 1.5])
        
        # Save as HTML
        safe_group_id = str(group_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        fig.write_html(f'prediction_results/timeline_{safe_group_id}.html')
        fig.show()
        
        print(f"  Created Plotly timeline for group: {group_id}")
    
    def _plot_simplified_summary(self):
        """Plot simplified overall summary focusing on model predictions only"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Prediction Summary', fontsize=16, fontweight='bold')
        
        # Define color mapping for pie chart
        def get_state_category(pred_label):
            pred_lower = pred_label.lower()
            if 'de-energized' in pred_lower or 'deenergized' in pred_lower or 'off' in pred_lower:
                return 'De-energized'
            elif 'stabilizing' in pred_lower or 'stabilize' in pred_lower or 'transition' in pred_lower:
                return 'Stabilizing'
            elif 'steady' in pred_lower or 'stable' in pred_lower or 'on' in pred_lower or 'normal' in pred_lower:
                return 'Steady State'
            else:
                return 'Other'
        
        # 1. State distribution pie chart
        ax = axes[0]
        state_categories = [get_state_category(pred) for pred in self.results['predictions']]
        unique_states, state_counts = np.unique(state_categories, return_counts=True)
        
        colors_map = {
            'De-energized': 'blue',
            'Stabilizing': 'orange',
            'Steady State': 'green',
            'Other': 'gray'
        }
        colors = [colors_map.get(state, 'gray') for state in unique_states]
        
        wedges, texts, autotexts = ax.pie(state_counts, labels=unique_states, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax.set_title('Overall State Distribution')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 2. DC System distribution
        ax = axes[1]
        dc1_count = (self.results['dc_systems'] == 1).sum()
        dc2_count = (self.results['dc_systems'] == 2).sum()
        
        bars = ax.bar(['DC1', 'DC2'], [dc1_count, dc2_count], 
                      color=['#4287f5', '#f54242'], alpha=0.7)
        ax.set_ylabel('Number of Predictions')
        ax.set_title('Predictions by DC System')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar, count in zip(bars, [dc1_count, dc2_count]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence levels
        ax = axes[2]
        max_probs = self.results['probabilities'].max(axis=1)
        
        # Define confidence bins
        high_conf = (max_probs >= 0.8).sum()
        med_conf = ((max_probs >= 0.5) & (max_probs < 0.8)).sum()
        low_conf = (max_probs < 0.5).sum()
        
        conf_levels = ['High\n(≥80%)', 'Medium\n(50-80%)', 'Low\n(<50%)']
        conf_counts = [high_conf, med_conf, low_conf]
        conf_colors = ['green', 'orange', 'red']
        
        bars = ax.bar(conf_levels, conf_counts, color=conf_colors, alpha=0.7)
        ax.set_ylabel('Number of Predictions')
        ax.set_title('Model Confidence Levels')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar, count in zip(bars, conf_counts):
            height = bar.get_height()
            percentage = count / len(max_probs) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('prediction_results/summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PREDICTION SUMMARY STATISTICS")
        print("="*60)
        print(f"Total predictions: {len(self.results['predictions']):,}")
        print(f"Mean confidence: {max_probs.mean():.3f}")
        print(f"\nState Distribution:")
        for state, count in zip(unique_states, state_counts):
            print(f"  {state}: {count:,} ({count/len(state_categories)*100:.1f}%)")
        print(f"\nConfidence Distribution:")
        print(f"  High (≥80%): {high_conf:,} ({high_conf/len(max_probs)*100:.1f}%)")
        print(f"  Medium (50-80%): {med_conf:,} ({med_conf/len(max_probs)*100:.1f}%)")
        print(f"  Low (<50%): {low_conf:,} ({low_conf/len(max_probs)*100:.1f}%)")
    
    def save_predictions(self, output_file: str = 'predictions_output.csv'):
        """Save predictions to CSV file"""
        print(f"\nSaving predictions to {output_file}...")
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'group_id': self.results['group_ids'],
            'dc_system': self.results['dc_systems'],
            'timestamp': self.results['timestamps'],
            'predicted_class': self.results['predictions'],
            'confidence': self.results['probabilities'].max(axis=1)
        })
        
        # Add top 3 predictions with probabilities
        top_k = min(3, self.results['probabilities'].shape[1])
        top_k_indices = np.argsort(self.results['probabilities'], axis=1)[:, -top_k:][:, ::-1]
        
        for i in range(top_k):
            class_idx = top_k_indices[:, i]
            class_names = self.loader.label_encoder.inverse_transform(class_idx)
            class_probs = self.results['probabilities'][np.arange(len(class_idx)), class_idx]
            
            results_df[f'top_{i+1}_class'] = class_names
            results_df[f'top_{i+1}_prob'] = class_probs
        
        results_df.to_csv(output_file, index=False)
        print(f"  Saved {len(results_df):,} predictions to {output_file}")
        
        return results_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def test_saved_model_on_new_data(
    new_parquet_path: str,
    model_pkl_path: str,
    window_size: int = 100,
    stride: int = 20,
    max_samples: Optional[int] = None,
    num_groups_to_visualize: int = 5,
    use_plotly: bool = False,
    voltage_only: bool = False):
    """
    Test a saved model on new parquet data with different format
    
    Args:
        new_parquet_path: Path to new format parquet file
        model_pkl_path: Path to saved model pickle file
        window_size: Window size (must match trained model)
        stride: Stride for sliding window
        max_samples: Maximum samples to process (None for all)
        num_groups_to_visualize: Number of groups to create plots for
        use_plotly: If True, also create interactive Plotly visualizations
        voltage_only: If True, synthesize current from voltage (when no current data available)
    """
    
    print("="*80)
    print("TESTING SAVED MODEL ON NEW DATA FORMAT")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Initialize loader with new format
        loader = NewFormatDataLoader(
            filepath=new_parquet_path,
            model_pkl_path=model_pkl_path,
            window_size=window_size,
            stride=stride
        )
        
        # Load and reshape data
        df = loader.load_and_reshape_data()
        
        # Process groups and extract features
        X, group_ids, dc_systems, timestamps = loader.process_groups(df, max_samples, voltage_only)
        
        # Initialize tester
        tester = ModelTester(loader)
        
        # Test model
        results = tester.test_model(X, group_ids, dc_systems, timestamps)
        
        # Generate visualizations
        tester.visualize_predictions(num_groups_to_visualize, use_plotly)
        
        # Save predictions to CSV
        predictions_df = tester.save_predictions('predictions_output.csv')
        
        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("="*80)
        print(f"Total execution time: {time.time()-start_time:.1f} seconds")
        print("\nOutput files:")
        print("  - predictions_output.csv (all predictions)")
        print("  - prediction_results/predictions_*.png (visualizations)")
        print("  - prediction_results/summary.png (simplified summary)")
        if use_plotly:
            print("  - prediction_results/timeline_*.html (interactive Plotly timelines)")
        
        return results, predictions_df
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Test the saved model on new data with VOLTAGE ONLY
    results, predictions_df = test_saved_model_on_new_data(
        new_parquet_path='new_format_data.parquet',  # Your new parquet file
        model_pkl_path='unified_dc_model.pkl',       # Your saved model
        window_size=100,                             # Must match trained model
        stride=20,
        max_samples=None,                            # Process all data
        num_groups_to_visualize=5,                   # Plot first 5 groups
        use_plotly=True,                             # Create interactive plots too
        voltage_only=True                            # Set to True when you only have voltage data
    )
    
    if results is not None:
        # You can access the results for further analysis
        print("\n" + "="*60)
        print("RESULTS AVAILABLE FOR FURTHER ANALYSIS")
        print("="*60)
        print(f"Results dict keys: {results.keys()}")
        print(f"Predictions DataFrame shape: {predictions_df.shape}")
