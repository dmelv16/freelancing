import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from typing import Tuple, Optional, List, Dict
import gc
import warnings
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# =============================================================================
# UNIFIED DATA LOADER FOR BOTH DC SYSTEMS
# =============================================================================
class UnifiedDataLoader:
    """Data loader that combines DC1 and DC2 data into a single dataset"""
    
    def __init__(self, filepath: str, window_size: int = 100, stride: int = 20):
        self.filepath = filepath
        self.window_size = window_size
        self.stride = stride
        
        # Group columns for maintaining data integrity
        self.group_cols = ['save', 'unit_id', 'ofp', 'station', 'test_case', 'test_run', 'Aircraft']
        
        # Get parquet file info
        self.parquet_file = pq.ParquetFile(filepath)
        self.num_row_groups = self.parquet_file.num_row_groups
        self.schema = self.parquet_file.schema_arrow
        
        # Single unified label encoder for both DC systems
        self.label_encoder = LabelEncoder()
        
        # Build group mapping
        print("Building optimized group mapping...")
        self.group_mapping = self._build_fast_group_mapping()
        
        # Fit encoder with combined statuses
        self._fit_unified_encoder()
        
    def _build_fast_group_mapping(self) -> Dict:
        """Build a fast mapping of groups to row group indices"""
        start_time = time.time()
        
        existing_group_cols = [col for col in self.group_cols if col in self.schema.names]
        
        if not existing_group_cols:
            print("No group columns found, treating as single group")
            return {'all': list(range(self.num_row_groups))}
        
        print(f"Reading group columns: {existing_group_cols}")
        
        group_data = []
        for i in range(self.num_row_groups):
            table = self.parquet_file.read_row_group(i, columns=existing_group_cols)
            group_df = table.to_pandas()
            group_df['row_group'] = i
            group_df['group_id'] = group_df[existing_group_cols].apply(
                lambda x: '_'.join(str(v) for v in x), axis=1
            )
            group_data.append(group_df[['group_id', 'row_group']])
        
        all_groups = pd.concat(group_data, ignore_index=True)
        group_to_rowgroups = all_groups.groupby('group_id')['row_group'].apply(set).to_dict()
        
        group_sizes = all_groups.groupby('group_id').size()
        print(f"Found {len(group_to_rowgroups)} unique groups in {time.time()-start_time:.1f} seconds")
        print(f"Group size stats: mean={group_sizes.mean():.0f}, median={group_sizes.median():.0f}, max={group_sizes.max()}")
        
        return group_to_rowgroups
    
    def _fit_unified_encoder(self):
        """Fit label encoder with combined DC1 and DC2 statuses"""
        print("Fitting unified label encoder...")
        
        all_statuses = set()
        
        for i in range(min(5, self.num_row_groups)):
            # Collect DC1 statuses
            if 'dc1_status' in self.schema.names:
                table = self.parquet_file.read_row_group(i, columns=['dc1_status'])
                statuses = table.column('dc1_status').to_pylist()
                # Add DC1 prefix to distinguish
                all_statuses.update([f"DC1_{s}" for s in statuses])
            
            # Collect DC2 statuses
            if 'dc2_status' in self.schema.names:
                table = self.parquet_file.read_row_group(i, columns=['dc2_status'])
                statuses = table.column('dc2_status').to_pylist()
                # Add DC2 prefix to distinguish
                all_statuses.update([f"DC2_{s}" for s in statuses])
        
        if all_statuses:
            self.label_encoder.fit(list(all_statuses))
            print(f"Unified classes ({len(self.label_encoder.classes_)}): {self.label_encoder.classes_}")
    
    def load_unified_data(self, test_groups: Optional[List[str]] = None,
                          is_training: bool = True, max_samples: Optional[int] = None):
        """Load data from both DC systems combined"""
        start_time = time.time()
        
        # Determine which groups to use
        if test_groups is not None:
            if is_training:
                groups_to_use = [g for g in self.group_mapping.keys() if g not in test_groups]
            else:
                groups_to_use = test_groups
        else:
            groups_to_use = list(self.group_mapping.keys())
        
        print(f"Processing {len(groups_to_use)} groups for {'training' if is_training else 'testing'}...")
        
        all_features = []
        all_labels = []
        all_group_ids = []  # Track which group each sample comes from
        all_dc_systems = []  # Track which DC system (1 or 2)
        samples_collected = 0
        
        for group_idx, group_id in enumerate(groups_to_use):
            if group_id not in self.group_mapping:
                continue
            
            row_groups_for_group = self.group_mapping[group_id]
            
            # Process DC1 data
            dc1_data = self._load_dc_data_for_group(row_groups_for_group, 1)
            if dc1_data:
                features_dc1, labels_dc1 = self._extract_windows_from_data(dc1_data, 1)
                if features_dc1:
                    all_features.extend(features_dc1)
                    all_labels.extend(labels_dc1)
                    all_group_ids.extend([group_id] * len(features_dc1))
                    all_dc_systems.extend([1] * len(features_dc1))
                    samples_collected += len(features_dc1)
            
            # Process DC2 data
            dc2_data = self._load_dc_data_for_group(row_groups_for_group, 2)
            if dc2_data:
                features_dc2, labels_dc2 = self._extract_windows_from_data(dc2_data, 2)
                if features_dc2:
                    all_features.extend(features_dc2)
                    all_labels.extend(labels_dc2)
                    all_group_ids.extend([group_id] * len(features_dc2))
                    all_dc_systems.extend([2] * len(features_dc2))
                    samples_collected += len(features_dc2)
            
            if group_idx % 10 == 0:
                print(f"  Processed {group_idx}/{len(groups_to_use)} groups, "
                      f"{samples_collected:,} samples collected ({time.time()-start_time:.1f}s)")
            
            if max_samples and samples_collected >= max_samples:
                print(f"Reached max samples limit ({max_samples})")
                break
        
        if not all_features:
            raise ValueError("No data found for specified criteria")
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        group_ids = np.array(all_group_ids)
        dc_systems = np.array(all_dc_systems)
        
        print(f"Loaded {len(X):,} samples in {time.time()-start_time:.1f} seconds")
        print(f"  DC1 samples: {(dc_systems == 1).sum():,}")
        print(f"  DC2 samples: {(dc_systems == 2).sum():,}")
        
        return X, y, group_ids, dc_systems
    
    def _load_dc_data_for_group(self, row_groups: set, dc_num: int) -> pd.DataFrame:
        """Load data for a specific DC system from row groups"""
        voltage_col = f'dc{dc_num}_voltage'
        current_col = f'dc{dc_num}_current'
        status_col = f'dc{dc_num}_status'
        
        # Check if columns exist
        if voltage_col not in self.schema.names:
            return None
        
        group_data = []
        for rg_idx in row_groups:
            cols_to_read = [voltage_col, current_col, status_col]
            if 'timestamp' in self.schema.names:
                cols_to_read.append('timestamp')
            
            try:
                table = self.parquet_file.read_row_group(rg_idx, columns=cols_to_read)
                df = table.to_pandas()
                df['dc_num'] = dc_num  # Add DC system identifier
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                group_data.append(df)
            except:
                continue
        
        if not group_data:
            return None
        
        return pd.concat(group_data, ignore_index=True)
    
    def _extract_windows_from_data(self, data: pd.DataFrame, dc_num: int):
        """Extract windows from data"""
        if data is None or len(data) < self.window_size:
            return [], []
        
        voltage_col = f'dc{dc_num}_voltage'
        current_col = f'dc{dc_num}_current'
        status_col = f'dc{dc_num}_status'
        
        voltage = data[voltage_col].values
        current = data[current_col].values
        status = data[status_col].values
        
        features = []
        labels = []
        
        # Pre-compute power
        power = voltage * current
        
        for i in range(0, len(voltage) - self.window_size + 1, self.stride):
            end_idx = i + self.window_size
            
            v_window = voltage[i:end_idx]
            c_window = current[i:end_idx]
            p_window = power[i:end_idx]
            
            if np.any(np.isnan(v_window)) or np.any(np.isnan(c_window)):
                continue
            
            # Compute features with DC indicator
            feature_vec = self._compute_features_with_dc(v_window, c_window, p_window, dc_num)
            features.append(feature_vec)
            
            # Get label with DC prefix
            label = f"DC{dc_num}_{status[end_idx - 1]}"
            label_encoded = self.label_encoder.transform([label])[0]
            labels.append(label_encoded)
        
        return features, labels
    
    def _compute_features_with_dc(self, voltage, current, power, dc_num):
        """Compute features including DC system indicator"""
        # Original 43 features
        features = np.zeros(44, dtype=np.float32)  # Added 1 for DC indicator
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
    
    def get_train_test_split(self, test_ratio: float = 0.3):
        """Get train/test split based on groups"""
        all_groups = list(self.group_mapping.keys())
        n_test = max(1, int(len(all_groups) * test_ratio))
        
        np.random.seed(42)
        np.random.shuffle(all_groups)
        test_groups = all_groups[:n_test]
        
        print(f"Split: {len(all_groups)-n_test} training groups, {n_test} test groups")
        return test_groups

# =============================================================================
# TRAINER WITH VISUALIZATION AND MODEL SAVING
# =============================================================================
class UnifiedTrainer:
    """Trainer for unified XGBoost model with visualization"""
    
    def __init__(self, data_loader: UnifiedDataLoader):
        self.loader = data_loader
        self.model = None
        self.scaler = None
        self.results = {}
        
    def train_unified_model(self, test_groups: List[str], 
                           max_train_samples: Optional[int] = None,
                           max_test_samples: Optional[int] = None,
                           save_model_path: str = 'unified_dc_model.pkl'):
        """Train unified XGBoost model and save it"""
        print("\n" + "="*80)
        print("TRAINING UNIFIED XGBOOST MODEL FOR DC1 & DC2")
        print("="*80)
        
        # Load training data
        print("\nLoading training data...")
        X_train, y_train, train_groups, train_dc_systems = self.loader.load_unified_data(
            test_groups, is_training=True, max_samples=max_train_samples
        )
        
        # Load test data
        print("\nLoading test data...")
        X_test, y_test, test_group_ids, test_dc_systems = self.loader.load_unified_data(
            test_groups, is_training=False, max_samples=max_test_samples
        )
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            tree_method='hist'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\nModel Performance:")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Generalization Gap: {train_accuracy - test_accuracy:.4f}")
        
        # Save model
        self._save_model(save_model_path)
        
        # Store results for visualization
        self.results = {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'test_groups': test_group_ids,
            'test_dc_systems': test_dc_systems,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }
        
        # Generate visualizations
        self._generate_group_visualizations(test_groups)
        self._generate_overall_metrics()
        
        return self.results
    
    def _save_model(self, filepath: str):
        """Save the trained model, scaler, and encoder"""
        print(f"\nSaving model to {filepath}...")
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.loader.label_encoder,
            'window_size': self.loader.window_size,
            'stride': self.loader.stride,
            'feature_names': self._get_feature_names()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Model saved successfully!")
    
    def _get_feature_names(self):
        """Generate feature names for interpretability"""
        feature_names = ['dc_system']
        
        for signal in ['voltage', 'current', 'power']:
            # Basic stats
            feature_names.extend([
                f'{signal}_mean', f'{signal}_std', f'{signal}_min', f'{signal}_max',
                f'{signal}_median', f'{signal}_q25', f'{signal}_q75', f'{signal}_range'
            ])
            # Differential features
            feature_names.extend([
                f'{signal}_diff_mean', f'{signal}_diff_std', f'{signal}_diff_max',
                f'{signal}_slope', f'{signal}_total_variation', f'{signal}_zero_crossings'
            ])
        
        feature_names.append('voltage_current_correlation')
        return feature_names
    
    def _generate_group_visualizations(self, test_groups: List[str]):
        """Generate visualization for each test group"""
        print("\nGenerating per-group visualizations...")
        
        y_test = self.results['y_test']
        y_pred = self.results['y_pred_test']
        group_ids = self.results['test_groups']
        dc_systems = self.results['test_dc_systems']
        
        # Create output directory
        import os
        os.makedirs('group_visualizations', exist_ok=True)
        
        # Process each test group
        for group_id in test_groups[:10]:  # Limit to first 10 groups for display
            # Get samples for this group
            group_mask = group_ids == group_id
            if not group_mask.any():
                continue
            
            group_y_test = y_test[group_mask]
            group_y_pred = y_pred[group_mask]
            group_dc = dc_systems[group_mask]
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Group: {group_id}\nSamples: {len(group_y_test)}', fontsize=14)
            
            # 1. Prediction timeline for DC1
            dc1_mask = group_dc == 1
            if dc1_mask.any():
                ax = axes[0, 0]
                dc1_y_test = group_y_test[dc1_mask]
                dc1_y_pred = group_y_pred[dc1_mask]
                
                # Decode labels for better visualization
                dc1_test_decoded = self.loader.label_encoder.inverse_transform(dc1_y_test)
                dc1_pred_decoded = self.loader.label_encoder.inverse_transform(dc1_y_pred)
                
                x_range = np.arange(len(dc1_y_test))
                ax.plot(x_range, dc1_y_test, 'g-', label='Actual', alpha=0.7, linewidth=2)
                ax.plot(x_range, dc1_y_pred, 'b--', label='Predicted', alpha=0.7, linewidth=1.5)
                
                # Mark errors
                errors = dc1_y_test != dc1_y_pred
                if errors.any():
                    ax.scatter(x_range[errors], dc1_y_test[errors], 
                             color='red', s=50, label='Errors', zorder=5)
                
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Class')
                ax.set_title(f'DC1 Predictions (Accuracy: {accuracy_score(dc1_y_test, dc1_y_pred):.2%})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 2. Prediction timeline for DC2
            dc2_mask = group_dc == 2
            if dc2_mask.any():
                ax = axes[0, 1]
                dc2_y_test = group_y_test[dc2_mask]
                dc2_y_pred = group_y_pred[dc2_mask]
                
                x_range = np.arange(len(dc2_y_test))
                ax.plot(x_range, dc2_y_test, 'g-', label='Actual', alpha=0.7, linewidth=2)
                ax.plot(x_range, dc2_y_pred, 'b--', label='Predicted', alpha=0.7, linewidth=1.5)
                
                errors = dc2_y_test != dc2_y_pred
                if errors.any():
                    ax.scatter(x_range[errors], dc2_y_test[errors], 
                             color='red', s=50, label='Errors', zorder=5)
                
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Class')
                ax.set_title(f'DC2 Predictions (Accuracy: {accuracy_score(dc2_y_test, dc2_y_pred):.2%})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 3. Confusion matrix for this group
            ax = axes[1, 0]
            cm = confusion_matrix(group_y_test, group_y_pred)
            
            # Get unique classes in this group
            unique_classes = np.unique(np.concatenate([group_y_test, group_y_pred]))
            class_names = self.loader.label_encoder.inverse_transform(unique_classes)
            
            # Create smaller confusion matrix with only present classes
            cm_reduced = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
            for i, true_class in enumerate(unique_classes):
                for j, pred_class in enumerate(unique_classes):
                    true_mask = group_y_test == true_class
                    pred_mask = group_y_pred == pred_class
                    cm_reduced[i, j] = np.sum(true_mask & pred_mask)
            
            sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            # 4. Class distribution
            ax = axes[1, 1]
            
            # Count occurrences
            actual_counts = pd.Series(group_y_test).value_counts().sort_index()
            pred_counts = pd.Series(group_y_pred).value_counts().sort_index()
            
            # Ensure both have same indices
            all_classes = sorted(set(actual_counts.index) | set(pred_counts.index))
            actual_counts = actual_counts.reindex(all_classes, fill_value=0)
            pred_counts = pred_counts.reindex(all_classes, fill_value=0)
            
            x = np.arange(len(all_classes))
            width = 0.35
            
            ax.bar(x - width/2, actual_counts.values, width, label='Actual', alpha=0.8)
            ax.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.8)
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels([self.loader.label_encoder.inverse_transform([c])[0] 
                               for c in all_classes], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            safe_group_id = str(group_id).replace('/', '_').replace('\\', '_')
            plt.savefig(f'group_visualizations/group_{safe_group_id}.png', dpi=100, bbox_inches='tight')
            plt.show()
            
            # Print summary for this group
            group_acc = accuracy_score(group_y_test, group_y_pred)
            print(f"  Group {group_id}: {len(group_y_test)} samples, Accuracy: {group_acc:.2%}")
    
    def _generate_overall_metrics(self):
        """Generate overall performance metrics visualization"""
        print("\nGenerating overall metrics visualization...")
        
        y_test = self.results['y_test']
        y_pred = self.results['y_pred_test']
        dc_systems = self.results['test_dc_systems']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Overall Model Performance', fontsize=16)
        
        # 1. Overall confusion matrix
        ax = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix (Acc: {self.results["test_accuracy"]:.2%})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # 2. Performance by DC system
        ax = axes[0, 1]
        dc1_mask = dc_systems == 1
        dc2_mask = dc_systems == 2
        
        dc1_acc = accuracy_score(y_test[dc1_mask], y_pred[dc1_mask]) if dc1_mask.any() else 0
        dc2_acc = accuracy_score(y_test[dc2_mask], y_pred[dc2_mask]) if dc2_mask.any() else 0
        
        systems = ['DC1', 'DC2', 'Overall']
        accuracies = [dc1_acc, dc2_acc, self.results['test_accuracy']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(systems, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by DC System')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Class-wise performance
        ax = axes[0, 2]
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Get class names
        unique_classes = np.unique(y_test)
        class_names = self.loader.label_encoder.inverse_transform(unique_classes)
        
        # Plot F1 scores
        ax.barh(range(len(f1)), f1, alpha=0.7)
        ax.set_yticks(range(len(f1)))
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_title('F1 Score by Class')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Error distribution over time
        ax = axes[1, 0]
        errors = y_test != y_pred
        window = 100
        error_rate = pd.Series(errors).rolling(window=window, min_periods=1).mean()
        
        ax.plot(error_rate, alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Error Rate')
        ax.set_title(f'Error Rate (Rolling {window}-sample window)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1-self.results['test_accuracy'], color='r', linestyle='--', 
                  label=f'Mean: {1-self.results["test_accuracy"]:.2%}')
        ax.legend()
        
        # 5. Feature importance (top 15)
        ax = axes[1, 1]
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self._get_feature_names()
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            ax.barh(range(len(indices)), importances[indices], alpha=0.7)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title('Top 15 Feature Importances')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 6. Train vs Test performance
        ax = axes[1, 2]
        metrics = ['Training', 'Testing']
        accuracies = [self.results['train_accuracy'], self.results['test_accuracy']]
        colors = ['#2ca02c', '#d62728']
        
        bars = ax.bar(metrics, accuracies, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Train vs Test Performance')
        ax.set_ylim([0, 1])
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}', ha='center', va='bottom')
        
        gap = self.results['train_accuracy'] - self.results['test_accuracy']
        ax.text(0.5, 0.5, f'Gap: {gap:.2%}', transform=ax.transAxes,
               ha='center', va='center', fontsize=12, 
               color='red' if gap > 0.1 else 'green')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('overall_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================
def main(filepath: str = 'power_data.parquet',
         window_size: int = 100,
         stride: int = 20,
         max_train_samples: Optional[int] = None,
         max_test_samples: Optional[int] = None,
         test_ratio: float = 0.3,
         model_save_path: str = 'unified_dc_model.pkl'):
    """
    Main function to train unified model and generate visualizations
    
    Args:
        filepath: Path to parquet file
        window_size: Window size for feature extraction
        stride: Stride for sliding window
        max_train_samples: Maximum training samples (None for all)
        max_test_samples: Maximum test samples (None for all)
        test_ratio: Ratio of data to use for testing
        model_save_path: Path to save the trained model
    """
    print("="*80)
    print("UNIFIED DC1/DC2 POWER CLASSIFICATION SYSTEM")
    print("="*80)
    
    total_start = time.time()
    
    # Initialize loader
    loader = UnifiedDataLoader(filepath, window_size, stride)
    
    # Get train/test split
    test_groups = loader.get_train_test_split(test_ratio)
    print(f"\nTest groups: {test_groups[:5]}..." if len(test_groups) > 5 else f"\nTest groups: {test_groups}")
    
    # Initialize trainer
    trainer = UnifiedTrainer(loader)
    
    # Train model and generate visualizations
    results = trainer.train_unified_model(
        test_groups,
        max_train_samples,
        max_test_samples,
        model_save_path
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: {model_save_path}")
    print(f"Overall Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Total execution time: {time.time()-total_start:.1f} seconds")
    print("\nVisualization files saved:")
    print("  - overall_metrics.png")
    print("  - group_visualizations/*.png")
    
    return results

# =============================================================================
# INFERENCE FUNCTION FOR SAVED MODEL
# =============================================================================
def load_and_predict(model_path: str, data: np.ndarray):
    """
    Load saved model and make predictions
    
    Args:
        model_path: Path to saved model pickle file
        data: Input data array (shape: [n_samples, 44])
    
    Returns:
        predictions: Decoded predictions
        probabilities: Prediction probabilities
    """
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoder = model_package['label_encoder']
    
    # Scale data
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions_encoded = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)
    
    # Decode predictions
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    return predictions, probabilities

if __name__ == "__main__":
    # Run the main training pipeline
    results = main(
        filepath='your_power_data.parquet',  # Update with your file path
        window_size=100,
        stride=20,
        max_train_samples=None,  # Use all available data
        max_test_samples=None,
        test_ratio=0.3,
        model_save_path='unified_dc_model.pkl'
    )
    
    # Example of how to use the saved model
    print("\n" + "="*80)
    print("EXAMPLE: Loading and using saved model")
    print("="*80)
    
    # Create dummy data for demonstration
    dummy_data = np.random.randn(10, 44).astype(np.float32)
    dummy_data[:, 0] = np.random.choice([0, 1], size=10)  # DC system indicator
    
    predictions, probabilities = load_and_predict('unified_dc_model.pkl', dummy_data)
    print(f"Predictions: {predictions[:5]}")
    print(f"Prediction probabilities shape: {probabilities.shape}")
