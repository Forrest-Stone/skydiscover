# EVOLVE-BLOCK-START
"""L-BFGS-B optimization with multi-start for n=26 circle packing."""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Multi-start L-BFGS-B optimization from hexagonal initializations."""
    n = 26
    best_centers, best_sum = None, 0
    
    # Try multiple initial configurations
    configs = [
        ([5,4,5,4,5,3], 0.19, 0.165, 0.08, 0.08),
        ([5,4,5,4,5,3], 0.185, 0.16, 0.09, 0.08),
        ([4,5,4,5,4,4], 0.19, 0.165, 0.08, 0.08),
        ([6,4,5,4,5,2], 0.18, 0.155, 0.07, 0.07),
    ]
    
    for rows, cs, rs, xs, ys in configs:
        centers = init_pattern(n, rows, cs, rs, xs, ys)
        centers = optimize(centers)
        radii = compute_radii(centers)
        s = np.sum(radii)
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    # Refine best solution
    best_centers = optimize(best_centers, maxiter=1000)
    radii = compute_radii(best_centers)
    return best_centers, radii, np.sum(radii)

def init_pattern(n, rows, cs, rs, x_start, y_start):
    """Initialize with configurable hexagonal pattern."""
    centers = np.zeros((n, 2))
    idx = 0
    for ri, count in enumerate(rows):
        y = y_start + ri * rs
        xo = x_start + (cs/2 if ri % 2 else 0)
        for ci in range(count):
            centers[idx] = [xo + ci * cs, y]
            idx += 1
    return centers

def optimize(centers, maxiter=500):
    """L-BFGS-B optimization for circle positions."""
    n = len(centers)
    bounds = [(0.01, 0.99)] * (2 * n)
    result = minimize(objective, centers.flatten(), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': maxiter})
    return result.x.reshape((n, 2))

def objective(x_flat):
    """Negative sum of radii for minimization."""
    return -np.sum(compute_radii(x_flat.reshape((26, 2))))

def compute_radii(centers):
    """Compute maximum valid radii respecting boundaries and non-overlap."""
    n = centers.shape[0]
    radii = np.array([min(max(x,0.001), max(y,0.001), max(1-x,0.001), max(1-y,0.001))
                      for x, y in centers])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d > 0 and radii[i] + radii[j] > d:
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