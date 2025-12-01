import string
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple


def visualize_fixations_and_centroids(
    path: string,
    centroids: List[Tuple[float, float]],
    title: str = 'Fixation Points'
):
    fig, ax = plt.subplots(figsize=(6, 6))
    fixations= pd.read_parquet(path)
    ax.scatter(fixations['x'], fixations['y'], s=100, c='red', alpha=0.6, 
               edgecolors='black', linewidth=1.5)
    
    for i, (x, y) in enumerate(zip(fixations['x'], fixations['y'])):
        ax.annotate(str(i), (x, y), fontsize=10, ha='center', va='center', 
                    color='white', weight='bold')
    
    for centroid in centroids:
        ax.scatter(centroid[0], centroid[1], s=200, c='blue', alpha=0.8, 
                   edgecolors='black', linewidth=2, marker='X')
        ax.annotate('C', (centroid[0], centroid[1]), fontsize=12, ha='center', 
                    va='center', color='white', weight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

