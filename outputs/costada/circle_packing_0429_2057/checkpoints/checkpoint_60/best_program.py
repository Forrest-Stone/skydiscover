# EVOLVE-BLOCK-START
"""Voronoi-guided iterative circle packing for n=26 circles in a unit square."""
import numpy as np
from scipy.spatial import Voronoi

def construct_packing():
    """
    Voronoi-guided iterative optimization for circle packing.
    Uses Lloyd's algorithm with boundary constraints to find optimal positions.
    Hexagonal 5-4-5-4-5-3 row pattern initialization (26 circles).
    """
    n = 26
    centers = init_hexagonal(n)
    
    # Voronoi-guided iteration (Lloyd's algorithm variant)
    for iteration in range(50):
        new_centers = voronoi_relax(centers)
        # Clamp to unit square with margin
        new_centers = np.clip(new_centers, 0.05, 0.95)
        
        # Check convergence
        if np.max(np.abs(new_centers - centers)) < 1e-6:
            break
        centers = new_centers
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def init_hexagonal(n):
    """Initialize with hexagonal packing pattern: 5-4-5-4-5-3 rows."""
    centers = np.zeros((n, 2))
    idx = 0
    # Hexagonal spacing: row offset = sqrt(3)/2 * col_spacing
    col_spacing = 0.18
    row_spacing = col_spacing * np.sqrt(3) / 2
    
    rows = [5, 4, 5, 4, 5, 3]  # 26 circles total
    y_start = 0.08
    for row_idx, count in enumerate(rows):
        y = y_start + row_idx * row_spacing
        # Offset alternate rows for hexagonal packing
        x_offset = 0.12 if row_idx % 2 == 1 else 0.08
        for col in range(count):
            x = x_offset + col * col_spacing
            centers[idx] = [x, y]
            idx += 1
    return centers

def voronoi_relax(centers):
    """Move each center toward centroid of its Voronoi cell."""
    n = len(centers)
    # Add mirror points for bounded Voronoi near edges
    mirrors = []
    for c in centers:
        mirrors.extend([[c[0], -c[1]], [c[0], 2-c[1]], 
                       [-c[0], c[1]], [2-c[0], c[1]]])
    all_pts = np.vstack([centers, mirrors])
    
    vor = Voronoi(all_pts)
    new_centers = np.zeros_like(centers)
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            new_centers[i] = centers[i]
            continue
        
        # Get vertices of Voronoi cell
        verts = vor.vertices[region]
        # Clip to unit square
        verts = np.clip(verts, 0, 1)
        
        # Move toward centroid
        centroid = np.mean(verts, axis=0)
        new_centers[i] = 0.7 * centers[i] + 0.3 * centroid
    
    return new_centers

def compute_max_radii(centers):
    """Compute maximum valid radii respecting boundaries and non-overlap."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                s = d / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii