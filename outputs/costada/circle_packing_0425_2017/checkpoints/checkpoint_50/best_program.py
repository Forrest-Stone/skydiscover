# EVOLVE-BLOCK-START
"""Global optimization for n=26 circle packing using differential evolution"""
import numpy as np
from scipy.optimize import differential_evolution


def construct_packing():
    """
    Use differential evolution to find optimal positions for 26 circles
    in a unit square, maximizing sum of radii.
    
    Uses 52 variables (26 circles × 2 coordinates) with bounds [0.05, 0.95].
    Constraints handled via iterative radius computation that satisfies
    both boundary and non-overlap constraints.
    """
    n = 26
    
    def objective(params):
        """Objective: minimize negative sum of radii to maximize packing."""
        centers = params.reshape(n, 2)
        # Clip to ensure valid positions
        centers = np.clip(centers, 0.01, 0.99)
        radii = compute_max_radii(centers)
        return -np.sum(radii)
    
    # Bounds for each coordinate: [0.05, 0.95] to stay inside unit square
    bounds = [(0.05, 0.95)] * (2 * n)
    
    # Run differential evolution with conservative settings for reliability
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        popsize=15,
        maxiter=200,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1
    )
    
    centers = result.x.reshape(n, 2)
    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Iteratively compute maximum radii satisfying boundary and non-overlap.
    Uses iterative constraint propagation until convergence.
    """
    n = centers.shape[0]
    # Initialize with distance to nearest boundary
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Iteratively resolve overlaps
    for _ in range(500):
        adjusted = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist > 0:
                    total = radii[i] + radii[j]
                    scale = dist / total
                    radii[i] *= scale
                    radii[j] *= scale
                    adjusted = True
        if not adjusted:
            break
    
    # Re-apply boundary constraints after all adjustments
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], centers[i, 1], 
                       1 - centers[i, 0], 1 - centers[i, 1])
    
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
