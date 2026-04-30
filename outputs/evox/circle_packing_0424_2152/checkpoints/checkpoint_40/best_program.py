# EVOLVE-BLOCK-START
"""Fast circle packing using SLSQP with multiple restarts for global optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct_packing():
    """
    Maximize sum of radii for 26 circles using diverse initializations.
    Combines hexagonal, corner-weighted, and random patterns with SLSQP.
    """
    n = 26
    best_sum, best_centers = 0, None
    
    for trial in range(25):
        if trial < 10:
            centers = init_hex(n, trial)
        elif trial < 18:
            centers = init_corner(n, trial)
        else:
            centers = init_random(n, trial)
        centers = optimize(centers)
        s = np.sum(compute_radii(centers))
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    return best_centers, compute_radii(best_centers), best_sum


def init_hex(n, trial):
    """Hexagonal pattern with various row configurations."""
    np.random.seed(trial * 17 + 3)
    configs = [[5,4,5,4,5,3], [4,5,4,5,4,4], [3,4,5,4,5,3,2], 
               [5,5,4,5,4,3], [4,4,5,4,5,4]]
    config = configs[trial % len(configs)]
    
    centers = []
    dx = 0.18 + 0.02 * np.random.rand()
    dy = 0.16 + 0.02 * np.random.rand()
    y = 0.08 + 0.03 * np.random.rand()
    
    for count in config:
        x = (1 - (count - 1) * dx) / 2 + 0.01 * np.random.randn()
        for i in range(count):
            centers.append([x + i * dx, y])
        y += dy
    
    centers = np.array(centers[:n])
    centers += np.random.randn(n, 2) * 0.01
    return np.clip(centers, 0.02, 0.98)


def init_corner(n, trial):
    """Corner-weighted initialization for larger corner circles."""
    np.random.seed(trial * 23 + 5)
    centers = np.zeros((n, 2))
    
    # 4 corners with larger offset for bigger radii
    offset = 0.10 + 0.02 * np.random.rand()
    centers[0] = [offset, offset]
    centers[1] = [1-offset, offset]
    centers[2] = [offset, 1-offset]
    centers[3] = [1-offset, 1-offset]
    
    # 8 edge circles
    for i in range(8):
        t = 0.28 + 0.44 * ((i % 2) + 0.1 * np.random.rand())
        side = i // 2
        if side == 0: centers[4+i] = [t, 0.06 + 0.02*np.random.rand()]
        elif side == 1: centers[4+i] = [t, 0.94 - 0.02*np.random.rand()]
        elif side == 2: centers[4+i] = [0.06 + 0.02*np.random.rand(), t]
        else: centers[4+i] = [0.94 - 0.02*np.random.rand(), t]
    
    # 14 interior in staggered rows
    idx = 12
    for row in range(5):
        count = 4 if row % 2 == 0 else 3
        y = 0.22 + row * 0.14
        for col in range(count):
            x = 0.18 + col * 0.22 + 0.11 * (row % 2)
            centers[idx] = [x + 0.02*np.random.randn(), y + 0.02*np.random.randn()]
            idx += 1
            if idx >= n: break
        if idx >= n: break
    
    return np.clip(centers, 0.02, 0.98)


def init_random(n, trial):
    """Random initialization with good spread."""
    np.random.seed(trial * 37 + 11)
    centers = np.random.rand(n, 2) * 0.76 + 0.12
    return np.clip(centers, 0.02, 0.98)


def compute_radii(centers):
    """Maximum radius from boundary and neighbor constraints."""
    radii = np.minimum.reduce([centers[:, 0], centers[:, 1], 
                               1 - centers[:, 0], 1 - centers[:, 1]])
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    return np.maximum(np.minimum(radii, dists.min(axis=1) / 2), 1e-8)


def optimize(centers):
    """SLSQP optimization with more iterations for better convergence."""
    n = len(centers)
    bounds = [(0.02, 0.98)] * (2 * n)
    
    def obj(x):
        return -np.sum(compute_radii(x.reshape((n, 2))))
    
    res = minimize(obj, centers.flatten(), method='SLSQP',
                   bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
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
