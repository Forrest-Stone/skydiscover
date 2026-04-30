# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using diverse multi-start with enhanced optimization"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import pdist, squareform


def construct_packing():
    """
    Diverse multi-start with 8 configurations exploring different patterns,
    aggressive basinhopping with adaptive steps, and dual-stage refinement.
    """
    best_centers, best_sum = None, 0
    
    # Expanded configurations: (corner_d, edge_e, spacing, y_start, edge_pos, pattern_type)
    configs = [
        # Best previous patterns
        (0.10, 0.07, 0.160, 0.24, 0.28, '545'),
        (0.11, 0.085, 0.163, 0.255, 0.29, '545'),
        (0.12, 0.08, 0.170, 0.26, 0.30, '545'),
        (0.095, 0.078, 0.158, 0.248, 0.31, '545'),
        # Larger corners, tighter interior
        (0.13, 0.065, 0.145, 0.22, 0.26, '444'),
        (0.14, 0.07, 0.150, 0.23, 0.27, '444'),
        # Smaller corners, more edge space
        (0.08, 0.08, 0.155, 0.25, 0.32, '545'),
        # Alternative pattern
        (0.105, 0.075, 0.158, 0.245, 0.285, '545'),
    ]
    
    for d, e, sp, y1, ep, pattern in configs:
        centers = create_initial_config(d, e, sp, y1, ep, pattern)
        centers = refine_positions_lbfgs(centers, max_iter=300)
        radii = compute_max_radii_vectorized(centers)
        s = np.sum(radii)
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    # Basin-hopping with adaptive steps
    best_centers = basin_hopping_adaptive(best_centers)
    
    # Final polish
    best_centers = refine_positions_lbfgs(best_centers, max_iter=500)
    radii = compute_max_radii_vectorized(best_centers)
    return best_centers, radii, np.sum(radii)


def create_initial_config(d, e, sp, y1, ep, pattern='545'):
    """Create initial configuration with configurable interior pattern."""
    centers = np.zeros((26, 2))
    
    # 4 corner circles - position for wall-constrained growth
    centers[0] = [d, d]
    centers[1] = [1-d, d]
    centers[2] = [d, 1-d]
    centers[3] = [1-d, 1-d]
    
    # 8 edge circles - configurable positions
    centers[4] = [ep, e]
    centers[5] = [1-ep, e]
    centers[6] = [ep, 1-e]
    centers[7] = [1-ep, 1-e]
    centers[8] = [e, ep]
    centers[9] = [e, 1-ep]
    centers[10] = [1-e, ep]
    centers[11] = [1-e, 1-ep]
    
    # 14 interior circles
    dy = sp * np.sqrt(3) / 2
    idx = 12
    
    if pattern == '545':
        # Standard 5-4-5 hexagonal pattern
        x1_start = 0.5 - 2 * sp
        for i in range(5):
            centers[idx] = [x1_start + i * sp, y1]
            idx += 1
        y2 = y1 + dy
        x2_start = 0.5 - 1.5 * sp
        for i in range(4):
            centers[idx] = [x2_start + i * sp, y2]
            idx += 1
        y3 = y2 + dy
        for i in range(5):
            centers[idx] = [x1_start + i * sp, y3]
            idx += 1
    elif pattern == '444':
        # Alternative 4-4-4-2 pattern
        x_start = 0.5 - 1.5 * sp
        for row in range(3):
            y = y1 + row * dy
            for i in range(4):
                centers[idx] = [x_start + i * sp, y]
                idx += 1
        # Remaining 2 circles
        centers[idx] = [0.5 - sp/2, y1 + 3*dy]
        centers[idx+1] = [0.5 + sp/2, y1 + 3*dy]
    
    return np.clip(centers, 0.02, 0.98)


def compute_max_radii_vectorized(centers, max_iter=300):
    """Vectorized computation of max radii constrained by borders and neighbors."""
    n = centers.shape[0]
    # Border constraints
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])
    
    # Pairwise distances
    dists = squareform(pdist(centers))
    
    for _ in range(max_iter):
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < 1e-10:
                    continue
                viol = radii[i] + radii[j] - d
                if viol > 1e-12:
                    max_viol = max(max_viol, viol)
                    total = radii[i] + radii[j]
                    radii[i] = d * radii[i] / total
                    radii[j] = d * radii[j] / total
        if max_viol < 1e-10:
            break
    return radii


def basin_hopping_adaptive(centers, niter=80):
    """Adaptive basinhopping with varying step sizes for better exploration."""
    n = len(centers)
    
    def objective(x):
        c = x.reshape(n, 2)
        r = compute_max_radii_vectorized(c, max_iter=200)
        return -np.sum(r)
    
    class AdaptiveStep:
        def __init__(self):
            self.stepsize = 0.04
            self.count = 0
        def __call__(self, x):
            # Alternate between larger exploration and smaller refinement steps
            step = self.stepsize * (1.5 if self.count % 5 < 2 else 0.5)
            self.count += 1
            x_new = x + np.random.uniform(-step, step, x.shape)
            return np.clip(x_new, 0.02, 0.98)
    
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': [(0.02, 0.98)] * (2 * n),
        'options': {'maxiter': 150, 'ftol': 1e-13}
    }
    
    result = basinhopping(objective, centers.flatten(), niter=niter,
                         take_step=AdaptiveStep(), minimizer_kwargs=minimizer_kwargs,
                         seed=42, disp=False)
    return result.x.reshape(n, 2)


def refine_positions_lbfgs(centers, max_iter=300):
    """Refine positions using L-BFGS-B with tight convergence tolerances."""
    n = len(centers)
    
    def objective(x):
        c = x.reshape(n, 2)
        r = compute_max_radii_vectorized(c)
        return -np.sum(r)
    
    bounds = [(0.02, 0.98)] * (2 * n)
    result = minimize(objective, centers.flatten(), method='L-BFGS-B',
                     bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
    return result.x.reshape(n, 2)





# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
