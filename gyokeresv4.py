import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score, log_loss,
    jaccard_score, hamming_loss, zero_one_loss, 
    precision_score, recall_score, f1_score,
    top_k_accuracy_score, average_precision_score,
    roc_curve, auc, multilabel_confusion_matrix
)
import xgboost as xgb
from typing import Tuple, Optional, List, Dict
import gc
import warnings
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
# COMPREHENSIVE METRICS CALCULATOR
# =============================================================================
class MetricsCalculator:
    """Calculate comprehensive metrics for model evaluation"""
    
    def __init__(self, model, scaler, label_encoder):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.metrics_dict = {}
        
    def calculate_all_metrics(self, X, y_true, y_pred, y_proba, 
                            group_ids=None, dc_systems=None, 
                            dataset_name='test'):
        """Calculate all possible metrics"""
        
        print(f"\nCalculating comprehensive metrics for {dataset_name} set...")
        
        # Get unique classes
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        class_names = self.label_encoder.inverse_transform(unique_classes)
        
        # Overall Metrics
        overall_metrics = self._calculate_overall_metrics(y_true, y_pred, y_proba, n_classes)
        
        # Per-Class Metrics
        class_metrics = self._calculate_per_class_metrics(y_true, y_pred, y_proba, unique_classes, class_names)
        
        # Per-DC System Metrics (if applicable)
        dc_metrics = None
        if dc_systems is not None:
            dc_metrics = self._calculate_dc_system_metrics(y_true, y_pred, y_proba, dc_systems)
        
        # Per-Group Metrics (if applicable)
        group_metrics = None
        if group_ids is not None:
            group_metrics = self._calculate_group_metrics(y_true, y_pred, y_proba, group_ids)
        
        # Confusion Matrix Details
        cm_metrics = self._calculate_confusion_matrix_metrics(y_true, y_pred, unique_classes, class_names)
        
        # Feature Importance Metrics
        feature_metrics = self._calculate_feature_importance_metrics()
        
        return {
            'overall': overall_metrics,
            'per_class': class_metrics,
            'per_dc': dc_metrics,
            'per_group': group_metrics,
            'confusion_matrix': cm_metrics,
            'feature_importance': feature_metrics
        }
    
    def _calculate_overall_metrics(self, y_true, y_pred, y_proba, n_classes):
        """Calculate overall model metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Averaging methods for multi-class
        for avg in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[fillf'jaccard_{avg}'] = jaccard_score(y_true, y_pred, average=avg, zero_division=0)
        
        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['zero_one_loss'] = zero_one_loss(y_true, y_pred)
        
        # Probabilistic metrics
        if y_proba is not None:
            metrics['log_loss'] = log_loss(y_true, y_proba)
            
            # Top-k accuracy
            for k in [2, 3, 5]:
                if k <= n_classes:
                    metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=k)
            
            # Multi-class ROC AUC
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')
            except:
                pass
        
        # Error analysis
        errors = y_true != y_pred
        metrics['error_rate'] = errors.mean()
        metrics['total_errors'] = errors.sum()
        metrics['total_samples'] = len(y_true)
        
        # Confidence metrics (if probabilities available)
        if y_proba is not None:
            max_probs = y_proba.max(axis=1)
            metrics['mean_confidence'] = max_probs.mean()
            metrics['std_confidence'] = max_probs.std()
            metrics['min_confidence'] = max_probs.min()
            metrics['max_confidence'] = max_probs.max()
            
            # Confidence on correct vs incorrect predictions
            correct_mask = y_true == y_pred
            metrics['mean_confidence_correct'] = max_probs[correct_mask].mean() if correct_mask.any() else 0
            metrics['mean_confidence_incorrect'] = max_probs[~correct_mask].mean() if (~correct_mask).any() else 0
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true, y_pred, y_proba, unique_classes, class_names):
        """Calculate metrics for each class"""
        
        class_metrics_list = []
        
        for i, (class_idx, class_name) in enumerate(zip(unique_classes, class_names)):
            metrics = {'class_name': class_name, 'class_index': int(class_idx)}
            
            # Binary classification metrics for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            # Basic metrics
            metrics['support'] = int((y_true == class_idx).sum())
            metrics['predicted_count'] = int((y_pred == class_idx).sum())
            metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
            
            # True/False Positives/Negatives
            tp = ((y_true == class_idx) & (y_pred == class_idx)).sum()
            fp = ((y_true != class_idx) & (y_pred == class_idx)).sum()
            tn = ((y_true != class_idx) & (y_pred != class_idx)).sum()
            fn = ((y_true == class_idx) & (y_pred != class_idx)).sum()
            
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            
            # Additional binary metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            metrics['prevalence'] = (tp + fn) / len(y_true) if len(y_true) > 0 else 0
            
            # Matthews correlation coefficient for this class
            metrics['mcc'] = matthews_corrcoef(y_true_binary, y_pred_binary)
            
            # Probabilistic metrics
            if y_proba is not None and class_idx < y_proba.shape[1]:
                y_proba_binary = y_proba[:, class_idx]
                
                # ROC AUC
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true_binary, y_proba_binary)
                except:
                    metrics['roc_auc'] = 0
                
                # Average Precision
                try:
                    metrics['average_precision'] = average_precision_score(y_true_binary, y_proba_binary)
                except:
                    metrics['average_precision'] = 0
                
                # Mean probability for this class
                metrics['mean_prob_when_true'] = y_proba_binary[y_true == class_idx].mean() if (y_true == class_idx).any() else 0
                metrics['mean_prob_when_false'] = y_proba_binary[y_true != class_idx].mean() if (y_true != class_idx).any() else 0
                
                # Brier score (for binary case)
                brier_score = np.mean((y_proba_binary - y_true_binary) ** 2)
                metrics['brier_score'] = brier_score
            
            class_metrics_list.append(metrics)
        
        return pd.DataFrame(class_metrics_list)
    
    def _calculate_dc_system_metrics(self, y_true, y_pred, y_proba, dc_systems):
        """Calculate metrics for each DC system"""
        
        dc_metrics_list = []
        
        for dc_num in [1, 2]:
            mask = dc_systems == dc_num
            if not mask.any():
                continue
            
            dc_y_true = y_true[mask]
            dc_y_pred = y_pred[mask]
            dc_y_proba = y_proba[mask] if y_proba is not None else None
            
            metrics = {'dc_system': dc_num}
            metrics['n_samples'] = len(dc_y_true)
            metrics['accuracy'] = accuracy_score(dc_y_true, dc_y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(dc_y_true, dc_y_pred)
            
            # Averaging methods
            for avg in ['micro', 'macro', 'weighted']:
                metrics[f'precision_{avg}'] = precision_score(dc_y_true, dc_y_pred, average=avg, zero_division=0)
                metrics[f'recall_{avg}'] = recall_score(dc_y_true, dc_y_pred, average=avg, zero_division=0)
                metrics[f'f1_{avg}'] = f1_score(dc_y_true, dc_y_pred, average=avg, zero_division=0)
            
            metrics['cohen_kappa'] = cohen_kappa_score(dc_y_true, dc_y_pred)
            metrics['matthews_corrcoef'] = matthews_corrcoef(dc_y_true, dc_y_pred)
            
            if dc_y_proba is not None:
                metrics['log_loss'] = log_loss(dc_y_true, dc_y_proba)
                max_probs = dc_y_proba.max(axis=1)
                metrics['mean_confidence'] = max_probs.mean()
            
            dc_metrics_list.append(metrics)
        
        return pd.DataFrame(dc_metrics_list)
    
    def _calculate_group_metrics(self, y_true, y_pred, y_proba, group_ids):
        """Calculate metrics for each group"""
        
        unique_groups = np.unique(group_ids)
        group_metrics_list = []
        
        for group_id in unique_groups:
            mask = group_ids == group_id
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            if len(group_y_true) == 0:
                continue
            
            metrics = {
                'group_id': str(group_id),
                'n_samples': len(group_y_true),
                'accuracy': accuracy_score(group_y_true, group_y_pred),
                'n_classes': len(np.unique(group_y_true))
            }
            
            # Only calculate if sufficient samples
            if len(group_y_true) > 1:
                metrics['precision_weighted'] = precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
                metrics['recall_weighted'] = recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
                metrics['f1_weighted'] = f1_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
                
                if len(np.unique(group_y_true)) > 1:
                    metrics['cohen_kappa'] = cohen_kappa_score(group_y_true, group_y_pred)
            
            group_metrics_list.append(metrics)
        
        return pd.DataFrame(group_metrics_list)
    
    def _calculate_confusion_matrix_metrics(self, y_true, y_pred, unique_classes, class_names):
        """Calculate detailed confusion matrix metrics"""
        
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        metrics = {
            'confusion_matrix': cm,
            'normalized_cm': cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        }
        
        # Per-class confusion metrics
        cm_details = []
        for i, class_name in enumerate(class_names):
            for j, pred_class_name in enumerate(class_names):
                cm_details.append({
                    'true_class': class_name,
                    'predicted_class': pred_class_name,
                    'count': int(cm[i, j]),
                    'percentage_of_true_class': float(cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0)
                })
        
        metrics['confusion_details'] = pd.DataFrame(cm_details)
        
        return metrics
    
    def _calculate_feature_importance_metrics(self):
        """Calculate feature importance metrics"""
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        feature_names = self._get_feature_names()
        importances = self.model.feature_importances_
        
        # Create DataFrame with feature importance details
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_rank': pd.Series(importances).rank(ascending=False, method='min').astype(int)
        }).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        feature_df['cumulative_importance'] = feature_df['importance'].cumsum()
        feature_df['importance_percentage'] = feature_df['importance'] / feature_df['importance'].sum() * 100
        
        return feature_df
    
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
    
    def save_metrics_to_csv(self, all_metrics, output_prefix='model_metrics'):
        """Save all metrics to a single comprehensive CSV file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        master_metrics = []
        
        # 1. Add overall metrics
        if 'overall' in all_metrics and all_metrics['overall']:
            for metric_name, metric_value in all_metrics['overall'].items():
                master_metrics.append({
                    'metric_category': f'{output_prefix}_overall',
                    'metric_subcategory': 'aggregate',
                    'metric_name': metric_name,
                    'class_or_group': 'all',
                    'value': metric_value,
                    'metric_type': 'overall'
                })
        
        # 2. Add per-class metrics
        if 'per_class' in all_metrics and all_metrics['per_class'] is not None:
            class_df = all_metrics['per_class']
            for idx, row in class_df.iterrows():
                class_name = row['class_name']
                for col in class_df.columns:
                    if col != 'class_name':
                        master_metrics.append({
                            'metric_category': f'{output_prefix}_per_class',
                            'metric_subcategory': col,
                            'metric_name': f"class_{class_name}_{col}",
                            'class_or_group': class_name,
                            'value': row[col],
                            'metric_type': 'per_class'
                        })
            
            # Add class summary statistics
            numeric_cols = class_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                master_metrics.extend([
                    {
                        'metric_category': f'{output_prefix}_class_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_mean_across_classes",
                        'class_or_group': 'all_classes',
                        'value': class_df[col].mean(),
                        'metric_type': 'class_summary'
                    },
                    {
                        'metric_category': f'{output_prefix}_class_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_std_across_classes",
                        'class_or_group': 'all_classes',
                        'value': class_df[col].std(),
                        'metric_type': 'class_summary'
                    },
                    {
                        'metric_category': f'{output_prefix}_class_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_min_across_classes",
                        'class_or_group': 'all_classes',
                        'value': class_df[col].min(),
                        'metric_type': 'class_summary'
                    },
                    {
                        'metric_category': f'{output_prefix}_class_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_max_across_classes",
                        'class_or_group': 'all_classes',
                        'value': class_df[col].max(),
                        'metric_type': 'class_summary'
                    }
                ])
        
        # 3. Add per-DC system metrics
        if 'per_dc' in all_metrics and all_metrics['per_dc'] is not None:
            dc_df = all_metrics['per_dc']
            for idx, row in dc_df.iterrows():
                dc_system = f"DC{int(row['dc_system'])}"
                for col in dc_df.columns:
                    if col != 'dc_system':
                        master_metrics.append({
                            'metric_category': f'{output_prefix}_per_dc',
                            'metric_subcategory': col,
                            'metric_name': f"{dc_system}_{col}",
                            'class_or_group': dc_system,
                            'value': row[col],
                            'metric_type': 'per_dc_system'
                        })
        
        # 4. Add per-group metrics
        if 'per_group' in all_metrics and all_metrics['per_group'] is not None:
            group_df = all_metrics['per_group']
            
            # Add individual group metrics
            for idx, row in group_df.iterrows():
                group_id = row['group_id']
                for col in group_df.columns:
                    if col != 'group_id':
                        master_metrics.append({
                            'metric_category': f'{output_prefix}_per_group',
                            'metric_subcategory': col,
                            'metric_name': f"group_{group_id}_{col}",
                            'class_or_group': group_id,
                            'value': row[col],
                            'metric_type': 'per_group'
                        })
            
            # Add group summary statistics
            numeric_cols = group_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                master_metrics.extend([
                    {
                        'metric_category': f'{output_prefix}_group_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_mean_across_groups",
                        'class_or_group': 'all_groups',
                        'value': group_df[col].mean(),
                        'metric_type': 'group_summary'
                    },
                    {
                        'metric_category': f'{output_prefix}_group_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_std_across_groups",
                        'class_or_group': 'all_groups',
                        'value': group_df[col].std(),
                        'metric_type': 'group_summary'
                    },
                    {
                        'metric_category': f'{output_prefix}_group_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_min_across_groups",
                        'class_or_group': 'all_groups',
                        'value': group_df[col].min(),
                        'metric_type': 'group_summary'
                    },
                    {
                        'metric_category': f'{output_prefix}_group_summary',
                        'metric_subcategory': col,
                        'metric_name': f"{col}_max_across_groups",
                        'class_or_group': 'all_groups',
                        'value': group_df[col].max(),
                        'metric_type': 'group_summary'
                    }
                ])
        
        # 5. Add confusion matrix metrics
        if 'confusion_matrix' in all_metrics and all_metrics['confusion_matrix']:
            if 'confusion_details' in all_metrics['confusion_matrix']:
                cm_df = all_metrics['confusion_matrix']['confusion_details']
                for idx, row in cm_df.iterrows():
                    master_metrics.extend([
                        {
                            'metric_category': f'{output_prefix}_confusion_matrix',
                            'metric_subcategory': 'count',
                            'metric_name': f"cm_{row['true_class']}_predicted_as_{row['predicted_class']}",
                            'class_or_group': f"{row['true_class']}->{row['predicted_class']}",
                            'value': row['count'],
                            'metric_type': 'confusion_matrix'
                        },
                        {
                            'metric_category': f'{output_prefix}_confusion_matrix',
                            'metric_subcategory': 'percentage',
                            'metric_name': f"cm_pct_{row['true_class']}_predicted_as_{row['predicted_class']}",
                            'class_or_group': f"{row['true_class']}->{row['predicted_class']}",
                            'value': row['percentage_of_true_class'],
                            'metric_type': 'confusion_matrix'
                        }
                    ])
        
        # 6. Add feature importance metrics
        if 'feature_importance' in all_metrics and all_metrics['feature_importance'] is not None:
            feat_df = all_metrics['feature_importance']
            for idx, row in feat_df.iterrows():
                master_metrics.extend([
                    {
                        'metric_category': f'{output_prefix}_feature_importance',
                        'metric_subcategory': 'importance',
                        'metric_name': f"feature_importance_{row['feature']}",
                        'class_or_group': row['feature'],
                        'value': row['importance'],
                        'metric_type': 'feature_importance'
                    },
                    {
                        'metric_category': f'{output_prefix}_feature_importance',
                        'metric_subcategory': 'rank',
                        'metric_name': f"feature_rank_{row['feature']}",
                        'class_or_group': row['feature'],
                        'value': row['importance_rank'],
                        'metric_type': 'feature_importance'
                    },
                    {
                        'metric_category': f'{output_prefix}_feature_importance',
                        'metric_subcategory': 'percentage',
                        'metric_name': f"feature_pct_{row['feature']}",
                        'class_or_group': row['feature'],
                        'value': row['importance_percentage'],
                        'metric_type': 'feature_importance'
                    }
                ])
        
        # Convert to DataFrame
        master_df = pd.DataFrame(master_metrics)
        
        # Sort for better organization
        master_df = master_df.sort_values(['metric_type', 'metric_category', 'metric_subcategory', 'metric_name'])
        
        # Save to CSV
        filename = f'MASTER_METRICS_{timestamp}.csv'
        master_df.to_csv(filename, index=False)
        print(f"\n✅ Saved ALL metrics to single file: {filename}")
        print(f"   Total metrics recorded: {len(master_df)} rows")
        
        return timestamp, master_df

# =============================================================================
# ENHANCED TRAINER WITH COMPREHENSIVE METRICS
# =============================================================================
class UnifiedTrainerWithMetrics(UnifiedTrainer):
    """Extended trainer that includes comprehensive metrics calculation"""
    
    def train_unified_model(self, test_groups: List[str], 
                           max_train_samples: Optional[int] = None,
                           max_test_samples: Optional[int] = None,
                           save_model_path: str = 'unified_dc_model.pkl',
                           save_metrics: bool = True):
        """Train unified XGBoost model with comprehensive metrics"""
        
        print("\n" + "="*80)
        print("TRAINING UNIFIED XGBOOST MODEL WITH COMPREHENSIVE METRICS")
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
        
        # Get prediction probabilities
        y_proba_train = self.model.predict_proba(X_train_scaled)
        y_proba_test = self.model.predict_proba(X_test_scaled)
        
        # Initialize metrics calculator
        metrics_calc = MetricsCalculator(self.model, self.scaler, self.loader.label_encoder)
        
        # Calculate comprehensive metrics for training set
        train_metrics = metrics_calc.calculate_all_metrics(
            X_train_scaled, y_train, y_pred_train, y_proba_train,
            train_groups, train_dc_systems, 'train'
        )
        
        # Calculate comprehensive metrics for test set
        test_metrics = metrics_calc.calculate_all_metrics(
            X_test_scaled, y_test, y_pred_test, y_proba_test,
            test_group_ids, test_dc_systems, 'test'
        )
        
        # Save metrics to CSV if requested
        if save_metrics:
            print("\nSaving metrics to master CSV file...")
            
            # Combine train and test metrics into one master file
            all_combined_metrics = []
            
            # Save train metrics
            train_timestamp, train_df = metrics_calc.save_metrics_to_csv(train_metrics, 'train')
            
            # Save test metrics
            test_timestamp, test_df = metrics_calc.save_metrics_to_csv(test_metrics, 'test')
            
            # Create combined master file with both train and test
            master_combined = pd.concat([train_df, test_df], ignore_index=True)
            
            # Add comparison metrics directly to master file
            self._add_comparison_metrics_to_master(train_metrics, test_metrics, master_combined, test_timestamp)
        
        # Display key metrics
        self._display_key_metrics(train_metrics, test_metrics)
        
        # Save model
        self._save_model(save_model_path)
        
        # Store results for visualization
        self.results = {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test,
            'test_groups': test_group_ids,
            'test_dc_systems': test_dc_systems,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        # Generate visualizations
        self._generate_group_visualizations(test_groups)
        self._generate_overall_metrics()
        
        return self.results
    
    def _add_comparison_metrics_to_master(self, train_metrics, test_metrics, master_df, timestamp):
        """Add comparison metrics directly to the master dataframe"""
        
        comparison_rows = []
        
        # Compare overall metrics
        for metric_name in train_metrics['overall'].keys():
            if metric_name in test_metrics['overall']:
                train_val = train_metrics['overall'][metric_name]
                test_val = test_metrics['overall'][metric_name]
                
                if isinstance(train_val, (int, float)) and isinstance(test_val, (int, float)):
                    # Add difference metric
                    comparison_rows.append({
                        'metric_category': 'comparison',
                        'metric_subcategory': 'difference',
                        'metric_name': f"diff_{metric_name}",
                        'class_or_group': 'train_vs_test',
                        'value': train_val - test_val,
                        'metric_type': 'comparison'
                    })
                    
                    # Add ratio metric
                    if test_val != 0:
                        comparison_rows.append({
                            'metric_category': 'comparison',
                            'metric_subcategory': 'ratio',
                            'metric_name': f"ratio_{metric_name}",
                            'class_or_group': 'train_vs_test',
                            'value': train_val / test_val,
                            'metric_type': 'comparison'
                        })
                    
                    # Add percentage change
                    if test_val != 0:
                        comparison_rows.append({
                            'metric_category': 'comparison',
                            'metric_subcategory': 'percent_change',
                            'metric_name': f"pct_change_{metric_name}",
                            'class_or_group': 'train_vs_test',
                            'value': ((train_val - test_val) / abs(test_val)) * 100,
                            'metric_type': 'comparison'
                        })
        
        # Add comparison rows to master dataframe
        comparison_df = pd.DataFrame(comparison_rows)
        master_combined = pd.concat([master_df, comparison_df], ignore_index=True)
        
        # Save the complete master file
        master_filename = f'COMPLETE_MODEL_METRICS_{timestamp}.csv'
        master_combined.to_csv(master_filename, index=False)
        
        print(f"\n" + "="*80)
        print(f"✅ MASTER METRICS FILE CREATED: {master_filename}")
        print(f"   Total rows: {len(master_combined)}")
        print(f"   Unique metrics: {master_combined['metric_name'].nunique()}")
        print(f"   Categories included: {master_combined['metric_category'].unique().tolist()}")
        print("="*80)
    
    def _display_key_metrics(self, train_metrics, test_metrics):
        """Display key metrics in console"""
        
        print("\n" + "="*80)
        print("KEY PERFORMANCE METRICS")
        print("="*80)
        
        print("\nOVERALL METRICS:")
        print("-" * 40)
        print(f"{'Metric':<25} {'Train':>12} {'Test':>12} {'Difference':>12}")
        print("-" * 40)
        
        key_metrics = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'cohen_kappa', 
                      'matthews_corrcoef', 'log_loss']
        
        for metric in key_metrics:
            if metric in train_metrics['overall'] and metric in test_metrics['overall']:
                train_val = train_metrics['overall'][metric]
                test_val = test_metrics['overall'][metric]
                diff = train_val - test_val
                
                print(f"{metric:<25} {train_val:>12.4f} {test_val:>12.4f} {diff:>12.4f}")
        
        print("\nPER-CLASS F1 SCORES (Test Set):")
        print("-" * 40)
        if test_metrics['per_class'] is not None:
            class_df = test_metrics['per_class'].sort_values('f1', ascending=False)
            for _, row in class_df.head(10).iterrows():
                print(f"{row['class_name']:<30} F1: {row['f1']:.4f} (Support: {row['support']})")
        
        print("\nTOP FEATURE IMPORTANCES:")
        print("-" * 40)
        if test_metrics['feature_importance'] is not None:
            for _, row in test_metrics['feature_importance'].head(10).iterrows():
                print(f"{row['feature']:<30} Importance: {row['importance']:.4f} ({row['importance_percentage']:.1f}%)")

# Update the UnifiedTrainer class name
class UnifiedTrainer(UnifiedTrainerWithMetrics):
    pass

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
    Main function to train unified model with comprehensive metrics
    
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
    print("UNIFIED DC1/DC2 POWER CLASSIFICATION WITH COMPREHENSIVE METRICS")
    print("="*80)
    
    total_start = time.time()
    
    # Initialize loader
    loader = UnifiedDataLoader(filepath, window_size, stride)
    
    # Get train/test split
    test_groups = loader.get_train_test_split(test_ratio)
    print(f"\nTest groups: {test_groups[:5]}..." if len(test_groups) > 5 else f"\nTest groups: {test_groups}")
    
    # Initialize trainer
    trainer = UnifiedTrainer(loader)
    
    # Train model with comprehensive metrics
    results = trainer.train_unified_model(
        test_groups,
        max_train_samples,
        max_test_samples,
        model_save_path,
        save_metrics=True  # Enable metrics saving
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE WITH COMPREHENSIVE METRICS")
    print("="*80)
    print(f"Model saved to: {model_save_path}")
    print(f"Total execution time: {time.time()-total_start:.1f} seconds")
    print("\n GENERATED FILES:")
    print(f"   Master Metrics File: COMPLETE_MODEL_METRICS_*.csv")
    print(f"     - Contains ALL metrics in a single file")
    print(f"     - Includes train, test, and comparison metrics")
    print(f"     - Organized by category and type")
    print(f"   Model: {model_save_path}")
    print(f"   Visualizations: overall_metrics.png & group_visualizations/*.png")
    
    return results

if __name__ == "__main__":
    # Run the main training pipeline with metrics
    results = main(
        filepath='your_power_data.parquet',  # Update with your file path
        window_size=100,
        stride=20,
        max_train_samples=None,  # Use all available data
        max_test_samples=None,
        test_ratio=0.3,
        model_save_path='unified_dc_model.pkl'
    )
