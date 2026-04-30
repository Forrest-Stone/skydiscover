# EVOLVE-BLOCK-START
"""Fast circle packing using SLSQP with multiple restarts for global optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct_packing():
    """
    Maximize sum of radii for 26 circles in unit square using SLSQP.
    Uses multiple random restarts to escape local optima.
    Analytical radius computation ensures fast convergence.
    """
    n = 26
    best_sum, best_centers = 0, None
    
    # Multiple restarts with different perturbations
    for trial in range(12):
        centers = initialize_hexagonal(n, trial)
        centers = optimize_slsqp(centers)
        radii = compute_radii(centers)
        s = np.sum(radii)
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    return best_centers, compute_radii(best_centers), best_sum


def initialize_hexagonal(n, trial):
    """Initialize with hexagonal pattern plus trial-specific perturbation."""
    centers = np.zeros((n, 2))
    # Pattern: 5,4,5,4,5,3 = 26 circles for efficient hexagonal packing
    dx, dy = 0.19, 0.165
    y = 0.09
    idx = 0
    for count in [5, 4, 5, 4, 5, 3]:
        x = 0.1 if count == 5 else 0.19
        for i in range(count):
            centers[idx] = [x + i * dx, y]
            idx += 1
        y += dy
    
    # Add perturbation for exploration
    np.random.seed(trial * 31 + 7)
    centers += np.random.randn(n, 2) * (0.01 + 0.005 * (trial % 4))
    return np.clip(centers, 0.02, 0.98)


def compute_radii(centers):
    """Compute maximum radius analytically from boundary and neighbor constraints."""
    # Boundary constraints: distance to each wall
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 
                               1 - centers[:, 0], 1 - centers[:, 1]])
    # Neighbor constraints: half distance to nearest circle
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    radii = np.minimum(radii, dists.min(axis=1) / 2)
    return np.maximum(radii, 1e-8)


def optimize_slsqp(centers):
    """Fast SLSQP optimization with explicit bounds."""
    n = len(centers)
    bounds = [(0.02, 0.98)] * (2 * n)
    
    def objective(x):
        return -np.sum(compute_radii(x.reshape((n, 2))))
    
    res = minimize(objective, centers.flatten(), method='SLSQP',
                   bounds=bounds, options={'maxiter': 200, 'ftol': 1e-9})
    return res.x.reshape((n, 2))


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
