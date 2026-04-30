# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles in a unit square."""
import numpy as np

def construct_packing():
    """
    Optimized packing: 4 corner + 16 edge + 6 interior circles.
    Strategy: corner circles at optimal radius, maximize edge coverage,
    fill interior gaps. Target sum: 2.635 (AlphaEvolve benchmark).
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corner circles - optimal radius touching two walls
    rc = 0.1464  # Optimal corner radius for unit square
    centers[idx] = [rc, rc]; idx += 1
    centers[idx] = [1-rc, rc]; idx += 1
    centers[idx] = [rc, 1-rc]; idx += 1
    centers[idx] = [1-rc, 1-rc]; idx += 1
    
    # 16 edge circles (4 per edge) - positioned to maximize radii
    re = 0.06  # Edge offset from wall
    for i in range(4):
        t = 0.26 + i * 0.16
        centers[idx] = [t, re]; idx += 1      # bottom
        centers[idx] = [t, 1-re]; idx += 1    # top
        centers[idx] = [re, t]; idx += 1      # left
        centers[idx] = [1-re, t]; idx += 1    # right
    
    # 6 interior circles in 2 staggered rows
    for i in range(3):
        centers[idx] = [0.30 + i * 0.20, 0.38]; idx += 1
    for i in range(3):
        centers[idx] = [0.30 + i * 0.20, 0.62]; idx += 1
    
    centers = np.clip(centers, 0.02, 0.98)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

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