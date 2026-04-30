# EVOLVE-BLOCK-START
"""Global optimization approach for n=26 circle packing using differential evolution."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct_packing():
    """
    Uses differential evolution to optimize circle positions for maximum radii sum.
    This approach simultaneously optimizes all positions rather than using fixed patterns.
    Exploits corner geometry where circles can be larger due to fewer boundary constraints.
    """
    n = 26
    np.random.seed(42)
    
    # Objective: minimize negative sum of radii (maximize sum)
    def objective(positions):
        centers = positions.reshape(n, 2)
        radii = compute_radii_fast(centers)
        return -np.sum(radii)
    
    # Bounds: all coordinates in valid range
    bounds = [(0.03, 0.97)] * (2 * n)
    
    # Global optimization with differential evolution
    result = differential_evolution(objective, bounds, maxiter=300, 
                                   seed=42, workers=1, polish=False,
                                   mutation=(0.5, 1.0), recombination=0.7,
                                   popsize=15, tol=1e-6)
    
    centers = result.x.reshape(n, 2)
    
    # Local refinement with L-BFGS-B
    result2 = minimize(objective, centers.flatten(), method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 200})
    centers = result2.x.reshape(n, 2)
    
    radii = compute_radii_fast(centers)
    return centers, radii, np.sum(radii)


def compute_radii_fast(centers):
    """Vectorized radius computation using distance matrix."""
    n = len(centers)
    # Distance to boundaries
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 
                               1 - centers[:, 0], 1 - centers[:, 1]])
    
    # Pairwise distances
    if n > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        radii = np.minimum(radii, distances.min(axis=1) / 2)
    
    return radii


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
