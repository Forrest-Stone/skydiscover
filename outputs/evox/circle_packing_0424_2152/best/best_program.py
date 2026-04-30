# EVOLVE-BLOCK-START
"""Circle packing using SLSQP+L-BFGS-B with optimized hexagonal initializations."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct_packing():
    """Two-phase optimization: diverse hexagonal patterns + local refinement."""
    n = 26
    best_sum, best_centers = 0, None
    candidates = []
    
    # Phase 1: Diverse initializations
    for trial in range(45):
        np.random.seed(trial * 31 + 7)
        centers = init_hex(n, trial)
        centers = optimize(centers)
        s = np.sum(compute_radii(centers))
        candidates.append((s, centers.copy()))
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    # Phase 2: Refine top candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    for _, base in candidates[:8]:
        for pt in range(8):
            np.random.seed(pt * 17 + 999)
            scale = 0.003 + 0.008 * np.random.rand()
            perturbed = base + np.random.randn(n, 2) * scale
            perturbed = np.clip(perturbed, 0.02, 0.98)
            perturbed = optimize(perturbed)
            s = np.sum(compute_radii(perturbed))
            if s > best_sum:
                best_sum, best_centers = s, perturbed.copy()
    
    return best_centers, compute_radii(best_centers), best_sum


def init_hex(n, trial):
    """Hexagonal pattern with varied configurations for 26 circles."""
    np.random.seed(trial * 17 + 3)
    configs = [[5,4,5,4,5,3], [4,5,4,5,4,4], [3,4,5,4,5,3,2], 
               [5,5,4,5,4,3], [4,4,5,4,5,4], [6,5,4,5,4,2],
               [4,5,5,4,5,3], [5,4,4,5,4,4], [3,5,4,5,4,5],
               [5,4,5,4,4,4], [4,5,4,4,5,4], [6,4,5,4,5,2]]
    config = configs[trial % len(configs)]
    centers = []
    dx = 0.158 + 0.022 * np.random.rand()
    dy = dx * np.sqrt(3) / 2
    y = 0.065 + 0.035 * np.random.rand()
    for count in config:
        x = (1 - (count - 1) * dx) / 2 + 0.008 * np.random.randn()
        for i in range(count):
            centers.append([x + i * dx, y])
        y += dy
    centers = np.array(centers[:n])
    centers += np.random.randn(len(centers), 2) * 0.006
    return np.clip(centers, 0.02, 0.98)


def compute_radii(centers):
    """Maximum radius from boundary and neighbor constraints."""
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 
                               1 - centers[:, 0], 1 - centers[:, 1]])
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    return np.maximum(np.minimum(radii, dists.min(axis=1) / 2), 1e-8)


def optimize(centers):
    """SLSQP global search + L-BFGS-B refinement."""
    n = len(centers)
    bounds = [(0.02, 0.98)] * (2 * n)
    def obj(x):
        return -np.sum(compute_radii(x.reshape((n, 2))))
    res = minimize(obj, centers.flatten(), method='SLSQP',
                   bounds=bounds, options={'maxiter': 700, 'ftol': 1e-12})
    res2 = minimize(obj, res.x, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 350, 'ftol': 1e-13})
    return res2.x.reshape((n, 2))


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
