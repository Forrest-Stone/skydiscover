# EVOLVE-BLOCK-START
"""Basinhopping global optimization for n=26 circle packing in a unit square."""
import numpy as np
from scipy.optimize import basinhopping

def construct_packing():
    """
    Stochastic global optimization using basinhopping algorithm.
    Escapes local optima through random perturbations followed by
    local minimization. L-BFGS-B handles bounds [0.01,0.99] on positions.
    """
    n = 26
    x0 = init_hexagonal(n).flatten()
    
    def objective(x_flat):
        centers = x_flat.reshape((n, 2))
        radii = compute_max_radii(centers)
        return -np.sum(radii)
    
    bounds = [(0.01, 0.99)] * (2 * n)
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds,
                        "options": {"maxiter": 100}}
    
    result = basinhopping(objective, x0, niter=300, T=0.3, stepsize=0.1,
                          minimizer_kwargs=minimizer_kwargs, niter_success=50)
    
    centers = result.x.reshape((n, 2))
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def init_hexagonal(n):
    """Initialize with hexagonal packing pattern: 5-4-5-4-5-3 rows."""
    centers = np.zeros((n, 2))
    idx, col_spacing = 0, 0.18
    row_spacing = col_spacing * np.sqrt(3) / 2
    rows = [5, 4, 5, 4, 5, 3]
    for row_idx, count in enumerate(rows):
        y = 0.08 + row_idx * row_spacing
        x_offset = 0.12 if row_idx % 2 else 0.08
        for col in range(count):
            centers[idx] = [x_offset + col * col_spacing, y]
            idx += 1
    return centers

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