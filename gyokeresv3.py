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
# OLD FORMAT PREDICTOR
# =============================================================================
class OldFormatPredictor:
    """Predict dc1_status and dc2_status for old format parquet files"""
    
    def __init__(self, model_pkl_path: str, window_size: int = 100, stride: int = 20):
        """
        Args:
            model_pkl_path: Path to saved unified model pickle file
            window_size: Window size (should match trained model)
            stride: Stride for sliding window
        """
        self.window_size = window_size
        self.stride = stride
        
        # Old format group columns
        self.group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
        
        # Load the saved model
        print(f"Loading model from {model_pkl_path}...")
        with open(model_pkl_path, 'rb') as f:
            self.model_package = pickle.load(f)
        
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.label_encoder = self.model_package['label_encoder']
        
        print(f"Model loaded successfully")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def load_and_predict(self, parquet_path: str, compare_with_actual: bool = True,
                        max_samples: Optional[int] = None):
        """
        Load old format parquet and predict status labels
        
        Args:
            parquet_path: Path to old format parquet file
            compare_with_actual: If True, compare predictions with actual labels
            max_samples: Maximum number of samples to process (None for all)
        
        Returns:
            results_df: DataFrame with predictions and comparisons
        """
        print("\n" + "="*80)
        print("PREDICTING STATUS LABELS FOR OLD FORMAT DATA")
        print("="*80)
        
        # Load parquet file
        print(f"\nLoading parquet file: {parquet_path}")
        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema_arrow
        
        print(f"Columns found: {schema.names}")
        
        # Build group mapping
        print("\nBuilding group mapping...")
        group_mapping = self._build_group_mapping(parquet_file)
        print(f"Found {len(group_mapping)} unique groups")
        
        # Process data and make predictions
        all_results = []
        samples_processed = 0
        
        for group_idx, (group_id, row_groups) in enumerate(group_mapping.items()):
            if group_idx % 10 == 0:
                print(f"Processing group {group_idx}/{len(group_mapping)}: {group_id}")
            
            # Process DC1
            dc1_results = self._process_dc_system(
                parquet_file, row_groups, 1, group_id, compare_with_actual
            )
            if dc1_results:
                all_results.extend(dc1_results)
                samples_processed += len(dc1_results)
            
            # Process DC2
            dc2_results = self._process_dc_system(
                parquet_file, row_groups, 2, group_id, compare_with_actual
            )
            if dc2_results:
                all_results.extend(dc2_results)
                samples_processed += len(dc2_results)
            
            if max_samples and samples_processed >= max_samples:
                print(f"Reached max samples limit ({max_samples})")
                break
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total samples processed: {len(results_df):,}")
        
        # Calculate and display accuracy if comparing with actual
        if compare_with_actual and 'actual_status' in results_df.columns:
            self._display_accuracy_metrics(results_df)
        
        return results_df
    
    def _build_group_mapping(self, parquet_file) -> Dict:
        """Build mapping of groups to row group indices"""
        num_row_groups = parquet_file.num_row_groups
        schema = parquet_file.schema_arrow
        
        # Find which group columns exist
        existing_group_cols = [col for col in self.group_cols if col in schema.names]
        
        if not existing_group_cols:
            print("No group columns found, treating as single group")
            return {'all': set(range(num_row_groups))}
        
        # Read group columns and create mapping
        group_data = []
        for i in range(num_row_groups):
            table = parquet_file.read_row_group(i, columns=existing_group_cols)
            group_df = table.to_pandas()
            group_df['row_group'] = i
            group_df['group_id'] = group_df[existing_group_cols].apply(
                lambda x: '_'.join(str(v) for v in x), axis=1
            )
            group_data.append(group_df[['group_id', 'row_group']])
        
        all_groups = pd.concat(group_data, ignore_index=True)
        group_to_rowgroups = all_groups.groupby('group_id')['row_group'].apply(set).to_dict()
        
        return group_to_rowgroups
    
    def _process_dc_system(self, parquet_file, row_groups: set, dc_num: int, 
                          group_id: str, compare_with_actual: bool) -> List[Dict]:
        """Process a single DC system for a group"""
        voltage_col = f'dc{dc_num}_voltage'
        current_col = f'dc{dc_num}_current'
        status_col = f'dc{dc_num}_status'
        
        schema = parquet_file.schema_arrow
        
        # Check if columns exist
        if voltage_col not in schema.names:
            return []
        
        # Read data for this group
        group_data = []
        for rg_idx in row_groups:
            cols_to_read = [voltage_col, current_col]
            if compare_with_actual and status_col in schema.names:
                cols_to_read.append(status_col)
            if 'timestamp' in schema.names:
                cols_to_read.append('timestamp')
            
            try:
                table = parquet_file.read_row_group(rg_idx, columns=cols_to_read)
                df = table.to_pandas()
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                group_data.append(df)
            except:
                continue
        
        if not group_data:
            return []
        
        # Concatenate all data for this group
        data_df = pd.concat(group_data, ignore_index=True)
        
        if len(data_df) < self.window_size:
            return []
        
        # Extract features and make predictions
        voltage = data_df[voltage_col].values
        current = data_df[current_col].values if current_col in data_df.columns else np.zeros_like(voltage)
        
        # Extract windows
        features, window_indices = self._extract_windows(voltage, current, dc_num)
        
        if not features:
            return []
        
        # Scale features
        X = np.array(features, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Create results
        results = []
        for i, (pred, prob, window_end_idx) in enumerate(zip(predictions, probabilities, window_indices)):
            result = {
                'group_id': group_id,
                'dc_system': dc_num,
                'window_end_index': window_end_idx,
                'predicted_status': pred,
                'confidence': prob.max(),
                'predicted_class_encoded': predictions_encoded[i]
            }
            
            # Remove DC prefix from prediction for cleaner output
            result['predicted_status_clean'] = pred.replace(f'DC{dc_num}_', '')
            
            # Add actual status if available
            if compare_with_actual and status_col in data_df.columns:
                actual_status = data_df[status_col].iloc[window_end_idx]
                result['actual_status'] = actual_status
                result['correct'] = (result['predicted_status_clean'] == actual_status)
            
            # Add timestamp if available
            if 'timestamp' in data_df.columns:
                result['timestamp'] = data_df['timestamp'].iloc[window_end_idx]
            
            results.append(result)
        
        return results
    
    def _extract_windows(self, voltage, current, dc_num):
        """Extract feature windows from voltage and current data"""
        features = []
        window_indices = []
        
        # Pre-compute power
        power = voltage * current
        
        for i in range(0, len(voltage) - self.window_size + 1, self.stride):
            end_idx = i + self.window_size - 1
            
            v_window = voltage[i:i + self.window_size]
            c_window = current[i:i + self.window_size]
            p_window = power[i:i + self.window_size]
            
            # Skip if NaN in voltage
            if np.any(np.isnan(v_window)):
                continue
            
            # Compute features
            feature_vec = self._compute_features_with_dc(v_window, c_window, p_window, dc_num)
            features.append(feature_vec)
            window_indices.append(end_idx)
        
        return features, window_indices
    
    def _compute_features_with_dc(self, voltage, current, power, dc_num):
        """Compute features matching the trained model structure"""
        features = np.zeros(44, dtype=np.float32)
        idx = 0
        
        # DC system indicator (0 for DC1, 1 for DC2)
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
            features[idx] = diff.mean() if len(diff) > 0 else 0
            features[idx+1] = diff.std() if len(diff) > 0 else 0
            features[idx+2] = np.abs(diff).max() if len(diff) > 0 else 0
            features[idx+3] = (signal[-1] - signal[0]) / len(signal)
            features[idx+4] = np.abs(diff).sum() if len(diff) > 0 else 0
            features[idx+5] = np.sum(np.diff(np.sign(diff)) != 0) if len(diff) > 1 else 0
            idx += 6
        
        # Correlation
        features[idx] = np.corrcoef(voltage, current)[0, 1] if len(voltage) > 1 else 0
        
        return features
    
    def _display_accuracy_metrics(self, results_df):
        """Display accuracy metrics comparing predictions with actual"""
        print("\n" + "="*60)
        print("ACCURACY METRICS")
        print("="*60)
        
        # Overall accuracy
        overall_acc = results_df['correct'].mean()
        print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        
        # Per DC system accuracy
        for dc_num in [1, 2]:
            dc_mask = results_df['dc_system'] == dc_num
            if dc_mask.any():
                dc_acc = results_df[dc_mask]['correct'].mean()
                dc_count = dc_mask.sum()
                print(f"DC{dc_num} Accuracy: {dc_acc:.4f} ({dc_count:,} samples)")
        
        # Per class accuracy
        print("\nPer-Class Performance:")
        print("-" * 50)
        
        for dc_num in [1, 2]:
            dc_mask = results_df['dc_system'] == dc_num
            if not dc_mask.any():
                continue
            
            dc_data = results_df[dc_mask]
            
            print(f"\nDC{dc_num} Classes:")
            unique_actual = dc_data['actual_status'].unique()
            
            for status in unique_actual:
                status_mask = dc_data['actual_status'] == status
                status_acc = dc_data[status_mask]['correct'].mean()
                status_count = status_mask.sum()
                print(f"  {status}: {status_acc:.4f} accuracy ({status_count:,} samples)")
        
        # Confusion matrix
        self._plot_confusion_matrices(results_df)
    
    def _plot_confusion_matrices(self, results_df):
        """Plot confusion matrices for each DC system"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, dc_num in enumerate([1, 2]):
            dc_mask = results_df['dc_system'] == dc_num
            if not dc_mask.any():
                continue
            
            dc_data = results_df[dc_mask]
            
            # Create confusion matrix
            cm = confusion_matrix(
                dc_data['actual_status'],
                dc_data['predicted_status_clean']
            )
            
            # Get labels
            labels = sorted(set(dc_data['actual_status'].unique()) | 
                          set(dc_data['predicted_status_clean'].unique()))
            
            # Plot
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f'DC{dc_num} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('old_format_confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_predictions(self, results_df: pd.DataFrame, output_path: str = 'predictions_old_format.csv'):
        """Save predictions to CSV file"""
        print(f"\nSaving predictions to {output_path}...")
        
        # Reorganize columns for better readability
        column_order = ['group_id', 'dc_system', 'timestamp', 'predicted_status_clean', 
                       'confidence', 'actual_status', 'correct']
        
        # Only include columns that exist
        columns_to_save = [col for col in column_order if col in results_df.columns]
        
        # Add any remaining columns
        remaining_cols = [col for col in results_df.columns if col not in columns_to_save]
        columns_to_save.extend(remaining_cols)
        
        # Save
        results_df[columns_to_save].to_csv(output_path, index=False)
        print(f"Saved {len(results_df):,} predictions to {output_path}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def visualize_predictions_timeline(results_df: pd.DataFrame, num_groups: int = 5):
    """Create timeline visualizations for predictions"""
    print("\nGenerating timeline visualizations...")
    
    unique_groups = results_df['group_id'].unique()
    groups_to_plot = unique_groups[:min(num_groups, len(unique_groups))]
    
    for group_id in groups_to_plot:
        group_data = results_df[results_df['group_id'] == group_id]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Predictions for Group: {group_id}', fontsize=14, fontweight='bold')
        
        # Define color mapping
        def get_color(status):
            status_lower = status.lower()
            if 'de-energized' in status_lower or 'deenergized' in status_lower or 'off' in status_lower:
                return 'blue'
            elif 'stabilizing' in status_lower or 'stabilize' in status_lower:
                return 'orange'
            elif 'steady' in status_lower or 'stable' in status_lower or 'on' in status_lower:
                return 'green'
            else:
                return 'gray'
        
        # Plot DC1 and DC2
        for idx, dc_num in enumerate([1, 2]):
            ax = axes[idx]
            dc_data = group_data[group_data['dc_system'] == dc_num]
            
            if len(dc_data) == 0:
                ax.text(0.5, 0.5, f'No DC{dc_num} Data', ha='center', va='center')
                continue
            
            # Sort by index or timestamp
            if 'timestamp' in dc_data.columns:
                dc_data = dc_data.sort_values('timestamp')
            else:
                dc_data = dc_data.sort_values('window_end_index')
            
            x = range(len(dc_data))
            
            # Plot predictions
            for i, row in enumerate(dc_data.itertuples()):
                pred_color = get_color(row.predicted_status_clean)
                ax.scatter(i, 1, c=pred_color, s=50, alpha=0.8)
                
                # Plot actual if available
                if hasattr(row, 'actual_status'):
                    actual_color = get_color(row.actual_status)
                    ax.scatter(i, 0, c=actual_color, s=50, alpha=0.8, marker='s')
            
            ax.set_ylim(-0.5, 1.5)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Actual', 'Predicted'])
            ax.set_xlabel('Sample Index')
            ax.set_title(f'DC{dc_num} Status')
            ax.grid(True, alpha=0.3)
            
            # Add accuracy if available
            if 'correct' in dc_data.columns:
                acc = dc_data['correct'].mean()
                ax.text(0.02, 0.95, f'Accuracy: {acc:.2%}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def predict_old_format_parquet(
    parquet_path: str,
    model_pkl_path: str,
    compare_with_actual: bool = True,
    max_samples: Optional[int] = None,
    visualize: bool = True,
    num_groups_to_visualize: int = 5):
    """
    Main function to predict status labels for old format parquet
    
    Args:
        parquet_path: Path to old format parquet file
        model_pkl_path: Path to saved unified model
        compare_with_actual: If True, compare predictions with actual labels
        max_samples: Maximum samples to process (None for all)
        visualize: If True, create visualizations
        num_groups_to_visualize: Number of groups to visualize
    
    Returns:
        results_df: DataFrame with all predictions
    """
    print("="*80)
    print("OLD FORMAT PARQUET STATUS PREDICTION")
    print("="*80)
    
    start_time = time.time()
    
    # Initialize predictor
    predictor = OldFormatPredictor(model_pkl_path)
    
    # Load data and make predictions
    results_df = predictor.load_and_predict(
        parquet_path,
        compare_with_actual=compare_with_actual,
        max_samples=max_samples
    )
    
    # Save predictions
    predictor.save_predictions(results_df)
    
    # Create visualizations if requested
    if visualize and len(results_df) > 0:
        visualize_predictions_timeline(results_df, num_groups_to_visualize)
    
    print(f"\nTotal execution time: {time.time()-start_time:.1f} seconds")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total predictions: {len(results_df):,}")
    print(f"Unique groups: {results_df['group_id'].nunique()}")
    print(f"Mean confidence: {results_df['confidence'].mean():.3f}")
    
    # Distribution of predicted classes
    print("\nPredicted Class Distribution:")
    for dc_num in [1, 2]:
        dc_data = results_df[results_df['dc_system'] == dc_num]
        if len(dc_data) > 0:
            print(f"\nDC{dc_num}:")
            class_counts = dc_data['predicted_status_clean'].value_counts()
            for status, count in class_counts.items():
                print(f"  {status}: {count:,} ({count/len(dc_data)*100:.1f}%)")
    
    return results_df

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Predict status for old format parquet
    results = predict_old_format_parquet(
        parquet_path='old_format_data.parquet',  # Your old format parquet
        model_pkl_path='unified_dc_model.pkl',   # Your trained model
        compare_with_actual=True,                # Compare with actual labels
        max_samples=None,                        # Process all data
        visualize=True,                          # Create visualizations
        num_groups_to_visualize=5                # Number of groups to plot
    )
