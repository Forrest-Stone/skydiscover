# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using position refinement via scipy.optimize"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Hybrid packing: 4 corner + 8 edge + 14 interior circles.
    Larger initial offsets allow better wall-constrained growth.
    Positions refined via L-BFGS-B with tighter convergence.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corner circles - larger offset for better wall-constrained growth
    d = 0.125
    centers[0] = [d, d]
    centers[1] = [1-d, d]
    centers[2] = [d, 1-d]
    centers[3] = [1-d, 1-d]
    
    # 8 edge circles - positioned for larger wall-constrained radii
    e = 0.09
    centers[4] = [0.32, e]
    centers[5] = [0.68, e]
    centers[6] = [0.32, 1-e]
    centers[7] = [0.68, 1-e]
    centers[8] = [e, 0.32]
    centers[9] = [e, 0.68]
    centers[10] = [1-e, 0.32]
    centers[11] = [1-e, 0.68]
    
    # 14 interior circles - centered hexagonal pattern with optimal spacing
    spacing = 0.172
    dy = spacing * np.sqrt(3) / 2
    idx = 12
    
    # Row 1: 5 circles (better centered)
    y1 = 0.265
    x1_start = 0.5 - 2 * spacing
    for i in range(5):
        centers[idx] = [x1_start + i * spacing, y1]
        idx += 1
    
    # Row 2: 4 circles (offset row)
    y2 = y1 + dy
    x2_start = 0.5 - 1.5 * spacing
    for i in range(4):
        centers[idx] = [x2_start + i * spacing, y2]
        idx += 1
    
    # Row 3: 5 circles
    y3 = y2 + dy
    for i in range(5):
        centers[idx] = [x1_start + i * spacing, y3]
        idx += 1
    
    centers = np.clip(centers, 0.02, 0.98)
    centers = refine_positions(centers)
    radii = compute_max_radii_iterative(centers)
    return centers, radii, np.sum(radii)


def refine_positions(centers, max_iter=300):
    """Refine circle positions using L-BFGS-B with tighter convergence."""
    n = len(centers)
    
    def objective(x):
        centers = x.reshape(n, 2)
        radii = compute_max_radii_iterative(centers)
        return -np.sum(radii)
    
    bounds = [(0.02, 0.98)] * (2 * n)
    
    result = minimize(objective, centers.flatten(), method='L-BFGS-B',
                     bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-11})
    
    return result.x.reshape(n, 2)


def compute_max_radii_iterative(centers, max_iter=400):
    """Compute maximum radii respecting borders and neighbor constraints."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist + 1e-10:
                    total = radii[i] + radii[j]
                    radii[i] = dist * radii[i] / total
                    radii[j] = dist * radii[j] / total
                    changed = True
        if not changed:
            break
    
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
