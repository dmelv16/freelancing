    def is_deenergized(self, voltages, labels, timestamps, slope_threshold=0.1, mean_threshold=5):
        """
        Identify de-energized segments based on slope and mean voltage.
        Returns a boolean mask array.
        """
        try:
            from scipy.stats import linregress
            import numpy as np
            
            # Ensure inputs are numpy arrays
            voltages = np.asarray(voltages)
            labels = np.asarray(labels)
            timestamps = np.asarray(timestamps)
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Initialize mask - all False initially
            deenergized_mask = np.zeros(len(voltages), dtype=bool)
            
            # Check each cluster
            for lab in unique_labels:
                # Get points in this cluster
                cluster_mask = (labels == lab)
                cluster_voltages = voltages[cluster_mask]
                cluster_timestamps = timestamps[cluster_mask]
                
                if len(cluster_voltages) < 2:
                    # Not enough points for regression, use mean only
                    if np.mean(cluster_voltages) < mean_threshold:
                        deenergized_mask[cluster_mask] = True
                else:
                    # Calculate slope and mean
                    try:
                        slope = abs(linregress(cluster_timestamps, cluster_voltages).slope)
                        mean_voltage = np.mean(cluster_voltages)
                        
                        # Check if cluster is de-energized
                        if slope < slope_threshold and mean_voltage < mean_threshold:
                            deenergized_mask[cluster_mask] = True
                    except:
                        # If regression fails, use mean only
                        if np.mean(cluster_voltages) < mean_threshold:
                            deenergized_mask[cluster_mask] = True
            
            return deenergized_mask
            
        except Exception as e:
            print(f"Error in is_deenergized: {e}")
            # Return all False if error
            return np.zeros(len(voltages), dtype=bool)
    
    def is_stabilizing(self, voltages, labels, timestamps, slope_cutoff=1):
        """
        Identify stabilizing segments based on slope.
        Returns a boolean mask array.
        """
        try:
            from scipy.stats import linregress
            import numpy as np
            
            # Ensure inputs are numpy arrays
            voltages = np.asarray(voltages)
            labels = np.asarray(labels)
            timestamps = np.asarray(timestamps)
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Initialize mask - all False initially
            stabilizing_mask = np.zeros(len(voltages), dtype=bool)
            
            # Check each cluster
            for lab in unique_labels:
                # Get points in this cluster
                cluster_mask = (labels == lab)
                cluster_voltages = voltages[cluster_mask]
                cluster_timestamps = timestamps[cluster_mask]
                
                if len(cluster_voltages) < 2:
                    # Not enough points for regression, skip
                    continue
                else:
                    # Calculate slope
                    try:
                        slope = abs(linregress(cluster_timestamps, cluster_voltages).slope)
                        
                        # Check if cluster is stabilizing
                        if slope > slope_cutoff:
                            stabilizing_mask[cluster_mask] = True
                    except:
                        # If regression fails, skip
                        continue
            
            return stabilizing_mask
            
        except Exception as e:
            print(f"Error in is_stabilizing: {e}")
            # Return all False if error
            return np.zeros(len(voltages), dtype=bool)
    
    def is_steadystate(self, voltages, labels, timestamps, slope_threshold=0.1, mean_threshold=20):
        """
        Identify steady state segments based on slope and mean voltage.
        Returns a boolean mask array.
        """
        try:
            from scipy.stats import linregress
            import numpy as np
            
            # Ensure inputs are numpy arrays
            voltages = np.asarray(voltages)
            labels = np.asarray(labels)
            timestamps = np.asarray(timestamps)
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Initialize mask - all False initially
            steadystate_mask = np.zeros(len(voltages), dtype=bool)
            
            # Check each cluster
            for lab in unique_labels:
                # Get points in this cluster
                cluster_mask = (labels == lab)
                cluster_voltages = voltages[cluster_mask]
                cluster_timestamps = timestamps[cluster_mask]
                
                if len(cluster_voltages) < 2:
                    # Not enough points for regression, use mean only
                    if np.mean(cluster_voltages) > mean_threshold:
                        steadystate_mask[cluster_mask] = True
                else:
                    # Calculate slope and mean
                    try:
                        slope = abs(linregress(cluster_timestamps, cluster_voltages).slope)
                        mean_voltage = np.mean(cluster_voltages)
                        
                        # Check if cluster is steady state
                        if slope < slope_threshold and mean_voltage > mean_threshold:
                            steadystate_mask[cluster_mask] = True
                    except:
                        # If regression fails, use mean only
                        if np.mean(cluster_voltages) > mean_threshold:
                            steadystate_mask[cluster_mask] = True
            
            return steadystate_mask
            
        except Exception as e:
            print(f"Error in is_steadystate: {e}")
            # Return all False if error
            return np.zeros(len(voltages), dtype=bool)
    
    def apply_status_labels(self, df):
        """
        Apply status labels using mask functions.
        """
        try:
            # Debug: Check what we're working with
            print(f"  Applying labels to dataframe with shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Get arrays from dataframe
            voltage_array = df['voltage'].to_numpy()
            segment_array = df['segment'].to_numpy()
            timestamp_array = df['timestamp'].to_numpy()
            
            print(f"  Arrays created - voltage: {voltage_array.shape}, segment: {segment_array.shape}, timestamp: {timestamp_array.shape}")
            
            # Get masks from your functions
            deenergized_mask = self.is_deenergized(
                voltage_array,
                segment_array,
                timestamp_array
            )
            
            print(f"  Deenergized mask created: {type(deenergized_mask)}, shape: {deenergized_mask.shape if hasattr(deenergized_mask, 'shape') else 'no shape'}")
            
            stabilizing_mask = self.is_stabilizing(
                voltage_array,
                segment_array,
                timestamp_array
            )
            
            print(f"  Stabilizing mask created: {type(stabilizing_mask)}, shape: {stabilizing_mask.shape if hasattr(stabilizing_mask, 'shape') else 'no shape'}")
            
            steadystate_mask = self.is_steadystate(
                voltage_array,
                segment_array,
                timestamp_array
            )
            
            print(f"  Steady state mask created: {type(steadystate_mask)}, shape: {steadystate_mask.shape if hasattr(steadystate_mask, 'shape') else 'no shape'}")
            
            # Apply labels based on masks
            df['predicted_status'] = "unidentified"
            df.loc[deenergized_mask, 'predicted_status'] = "de-energized"
            df.loc[stabilizing_mask, 'predicted_status'] = "stabilizing"
            df.loc[steadystate_mask, 'predicted_status'] = "steady_state"
            
            print(f"  Labels applied. Value counts: {df['predicted_status'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"ERROR in apply_status_labels: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
    
