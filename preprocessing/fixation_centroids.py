import pandas as pd
from sklearn.cluster import DBSCAN

def calculate_fixation_centroids(parquet_path, eps=0.1, min_samples=2, duration_threshold=1000):
    """
    Extract weighted centroids from fixation data using DBSCAN clustering.
    
    Parameters:
    -----------
    parquet_path : str
        Path to the parquet file containing fixation data
    eps : float
        DBSCAN epsilon parameter (maximum distance between neighbors)
    min_samples : int
        DBSCAN minimum samples parameter
    duration_threshold : float
        Minimum total duration in ms to include a cluster/point (default: 1000)
    
    Returns:
    --------
    list
        List of [x, y] coordinates for centroids with duration >= threshold
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    # Return empty list if no data
    if df.empty or len(df) == 0:
        return []
    
    # Apply DBSCAN clustering
    coordinates = df[['x', 'y']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(coordinates)
    
    # Collect centroids that meet the duration threshold
    centroids = []
    
    for cluster_id in df['cluster'].unique():
        cluster_points = df[df['cluster'] == cluster_id]
        total_duration = cluster_points['duration_ms'].sum()
        
        # Only include if duration meets threshold
        if total_duration >= duration_threshold:
            # Calculate weighted centroid
            weighted_x = (cluster_points['x'] * cluster_points['duration_ms']).sum() / total_duration
            weighted_y = (cluster_points['y'] * cluster_points['duration_ms']).sum() / total_duration
            centroids.append([float(weighted_x), float(weighted_y)])
    
    return centroids