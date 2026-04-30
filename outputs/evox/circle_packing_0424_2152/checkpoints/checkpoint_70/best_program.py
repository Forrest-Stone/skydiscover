# EVOLVE-BLOCK-START
"""Fast circle packing using SLSQP with multiple restarts for global optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct_packing():
    """
    Two-phase optimization: Phase 1 explores diverse initializations,
    Phase 2 refines top candidates with perturbation and local search.
    """
    n = 26
    best_sum, best_centers = 0, None
    candidates = []
    
    # Phase 1: Global exploration with diverse patterns
    for trial in range(40):
        np.random.seed(trial * 31 + 7)
        r = trial % 4
        if r == 0:
            centers = init_hex(n, trial)
        elif r == 1:
            centers = init_corner(n, trial)
        elif r == 2:
            centers = init_shell(n, trial)
        else:
            centers = init_random(n, trial)
        
        centers = optimize(centers)
        s = np.sum(compute_radii(centers))
        candidates.append((s, centers.copy()))
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    # Phase 2: Local refinement around top candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    for _, base_centers in candidates[:6]:
        for pert_trial in range(6):
            np.random.seed(pert_trial * 17 + 999)
            scale = 0.003 + 0.007 * np.random.rand()
            perturbed = base_centers + np.random.randn(n, 2) * scale
            perturbed = np.clip(perturbed, 0.02, 0.98)
            perturbed = optimize(perturbed)
            s = np.sum(compute_radii(perturbed))
            if s > best_sum:
                best_sum, best_centers = s, perturbed.copy()
    
    return best_centers, compute_radii(best_centers), best_sum


def init_hex(n, trial):
    """Hexagonal pattern with optimized spacing for dense interior packing."""
    np.random.seed(trial * 17 + 3)
    configs = [[5,4,5,4,5,3], [4,5,4,5,4,4], [3,4,5,4,5,3,2], 
               [5,5,4,5,4,3], [4,4,5,4,5,4], [6,5,4,5,4,2],
               [4,5,5,4,5,3], [5,4,4,5,4,4], [3,5,4,5,4,5]]
    config = configs[trial % len(configs)]
    
    centers = []
    dx = 0.160 + 0.020 * np.random.rand()
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


def init_corner(n, trial):
    """Corner-weighted: tight corners for max radius, strategic edge placement."""
    np.random.seed(trial * 23 + 5)
    centers = np.zeros((n, 2))
    
    # 4 corners - tight placement for maximum corner radius
    offset = 0.065 + 0.025 * np.random.rand()
    centers[0] = [offset, offset]
    centers[1] = [1-offset, offset]
    centers[2] = [offset, 1-offset]
    centers[3] = [1-offset, 1-offset]
    
    # 8 edge circles - optimized positions
    for i in range(8):
        t = 0.15 + 0.70 * ((i % 4) / 3) + 0.02 * np.random.rand()
        edge_offset = 0.045 + 0.02 * np.random.rand()
        side = i // 2
        if side == 0: centers[4+i] = [t, edge_offset]
        elif side == 1: centers[4+i] = [t, 1-edge_offset]
        elif side == 2: centers[4+i] = [edge_offset, t]
        else: centers[4+i] = [1-edge_offset, t]
    
    # 14 interior in hexagonal arrangement
    idx = 12
    spacing = 0.165 + 0.018 * np.random.rand()
    for row in range(5):
        count = 4 if row % 2 == 0 else 3
        y = 0.19 + row * spacing * np.sqrt(3) / 2
        for col in range(count):
            x = 0.17 + col * spacing * 1.05 + 0.082 * (row % 2)
            centers[idx] = [x + 0.008*np.random.randn(), y + 0.008*np.random.randn()]
            idx += 1
            if idx >= n: break
        if idx >= n: break
    
    return np.clip(centers, 0.02, 0.98)


def init_shell(n, trial):
    """Shell-based: corners, edges, then interior in concentric layers."""
    np.random.seed(trial * 41 + 13)
    centers = np.zeros((n, 2))
    
    offset = 0.06 + 0.02 * np.random.rand()
    centers[:4] = [[offset, offset], [1-offset, offset], 
                   [offset, 1-offset], [1-offset, 1-offset]]
    
    for i in range(8):
        t = 0.10 + 0.80 * ((i % 4) / 3) + 0.02 * np.random.rand()
        edge_offset = 0.042 + 0.018 * np.random.rand()
        side = i // 2
        if side == 0: centers[4+i] = [t, edge_offset]
        elif side == 1: centers[4+i] = [t, 1-edge_offset]
        elif side == 2: centers[4+i] = [edge_offset, t]
        else: centers[4+i] = [1-edge_offset, t]
    
    idx = 12
    dx = 0.162 + 0.018 * np.random.rand()
    dy = dx * np.sqrt(3) / 2
    y_start = 0.17 + 0.025 * np.random.rand()
    for row in range(5):
        count = 4 if row % 2 == 0 else 3
        y = y_start + row * dy
        for col in range(count):
            x = 0.18 + col * dx + 0.081 * (row % 2)
            centers[idx] = [x, y]
            idx += 1
            if idx >= n: break
        if idx >= n: break
    
    centers += np.random.randn(n, 2) * 0.006
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
    """Multi-stage optimization: SLSQP for global, L-BFGS-B for refinement."""
    n = len(centers)
    bounds = [(0.02, 0.98)] * (2 * n)
    
    def obj(x):
        return -np.sum(compute_radii(x.reshape((n, 2))))
    
    # Stage 1: SLSQP for global exploration
    res = minimize(obj, centers.flatten(), method='SLSQP',
                   bounds=bounds, options={'maxiter': 600, 'ftol': 1e-12})
    
    # Stage 2: L-BFGS-B for local refinement
    res2 = minimize(obj, res.x, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 300, 'ftol': 1e-13})
    
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
