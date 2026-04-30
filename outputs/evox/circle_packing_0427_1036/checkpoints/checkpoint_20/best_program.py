# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized hexagonal packing for 26 circles in unit square.
    4 corners + 8 edges + 14 interior. Corners at (r,r) for max radius r.
    Edge circles positioned for larger boundary distance.
    Interior uses consistent hexagonal spacing (sqrt(3)/2 ratio).
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corners - positioned for larger radii while maintaining spacing
    for cx, cy in [(0.11, 0.11), (0.89, 0.11), (0.11, 0.89), (0.89, 0.89)]:
        centers[idx] = [cx, cy]
        idx += 1
    
    # 8 edge circles - further from edges for larger potential radii
    for x in [0.34, 0.5, 0.66]:
        centers[idx] = [x, 0.08]
        idx += 1
    for x in [0.34, 0.5, 0.66]:
        centers[idx] = [x, 0.92]
        idx += 1
    centers[idx] = [0.08, 0.5]
    idx += 1
    centers[idx] = [0.92, 0.5]
    idx += 1
    
    # 14 interior - hexagonal staggered with consistent spacing
    for x in [0.22, 0.40, 0.60, 0.78]:
        centers[idx] = [x, 0.24]
        idx += 1
    for x in [0.31, 0.50, 0.69]:
        centers[idx] = [x, 0.42]
        idx += 1
    for x in [0.22, 0.40, 0.60, 0.78]:
        centers[idx] = [x, 0.58]
        idx += 1
    for x in [0.31, 0.50, 0.69]:
        centers[idx] = [x, 0.76]
        idx += 1
    
    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum valid radii using iterative reduction then expansion.
    Phase 1: Reduce overlapping radii proportionally.
    Phase 2: Expand each circle to its maximum feasible radius.
    Guarantees all constraints satisfied with maximal sum.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to nearest boundary
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Phase 1: Iteratively reduce overlapping radii proportionally
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    total = radii[i] + radii[j]
                    scale = dist / total
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    
    radii = np.maximum(radii, 1e-8)
    
    # Phase 2: Expand each circle to maximum feasible radius
    for _ in range(200):
        improved = False
        # Sort by current radius (smallest first) for fair expansion
        order = np.argsort(radii)
        for i in order:
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        if not improved:
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
