def is_deenergized(self, voltages: np.ndarray, labels: np.ndarray, timestamps: np.ndarray, 
                   slope_threshold: float = 0.1, mean_threshold: float = 5) -> np.ndarray:
    from scipy.stats import linregress
    
    unique_labels = np.unique(labels)
    deenergized_clusters = np.zeros_like(unique_labels, dtype=bool)
    cluster_mean_voltages = []
    
    for ix, lab in enumerate(unique_labels):
        cluster_mask = labels == lab
        if np.sum(cluster_mask) > 1:  # Need at least 2 points for regression
            cluster_mean_voltage = np.mean(voltages[cluster_mask])
            cluster_abs_slope = np.abs(
                linregress(timestamps[cluster_mask], voltages[cluster_mask]).slope)
            
            if (cluster_abs_slope < slope_threshold and cluster_mean_voltage < mean_threshold):
                deenergized_clusters[ix] = True
            cluster_mean_voltages.append(cluster_mean_voltage)
        else:
            cluster_mean_voltages.append(np.mean(voltages[cluster_mask]))
    
    if len(deenergized_clusters) > 2:
        for i in range(1, len(deenergized_clusters) - 1):
            if (deenergized_clusters[i-1] and deenergized_clusters[i+1] and 
                cluster_mean_voltages[i] < mean_threshold):
                deenergized_clusters[i] = True
    
    deenergized_mask = np.zeros_like(voltages, dtype=bool)
    for ix, lab in enumerate(unique_labels):
        if deenergized_clusters[ix]:  # FIX: Only assign True when cluster is deenergized
            deenergized_mask[labels == lab] = True
    
    return deenergized_mask

def is_steadystate(self, voltages: np.ndarray, labels: np.ndarray, timestamps: np.ndarray,
                   slope_threshold: float = 0.1, mean_threshold: float = 20) -> np.ndarray:
    from scipy.stats import linregress
    
    unique_labels = np.unique(labels)
    steadystate_clusters = np.zeros_like(unique_labels, dtype=bool)
    cluster_mean_voltages = []
    
    for ix, lab in enumerate(unique_labels):
        cluster_mask = labels == lab
        if np.sum(cluster_mask) > 1:  # Need at least 2 points for regression
            cluster_mean_voltage = np.mean(voltages[cluster_mask])
            cluster_abs_slope = np.abs(
                linregress(timestamps[cluster_mask], voltages[cluster_mask]).slope
            )
            if (cluster_abs_slope < slope_threshold and cluster_mean_voltage > mean_threshold):
                steadystate_clusters[ix] = True
            cluster_mean_voltages.append(cluster_mean_voltage)
        else:
            cluster_mean_voltages.append(np.mean(voltages[cluster_mask]))
    
    if len(steadystate_clusters) > 2:
        for i in range(1, len(steadystate_clusters) - 1):
            if (steadystate_clusters[i-1] and steadystate_clusters[i+1] and  # FIX: Fixed typo here
                cluster_mean_voltages[i] > mean_threshold):
                steadystate_clusters[i] = True
    
    steadystate_mask = np.zeros_like(voltages, dtype=bool)
    for ix, lab in enumerate(unique_labels):
        if steadystate_clusters[ix]:  # FIX: Only assign True when cluster is steady state
            steadystate_mask[labels == lab] = True
    
    return steadystate_mask

