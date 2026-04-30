# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized packing for 26 circles with improved positioning:
    - 4 corner circles positioned for optimal wall contact
    - 12 edge circles evenly distributed along edges
    - 10 interior circles in optimized hexagonal-like pattern
    Uses iterative radius optimization for better results.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # CORNER CIRCLES: Position for maximum wall contact (radius = position)
    cr = 0.095
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    # EDGE CIRCLES: 12 circles, 3 per edge, positioned close to walls
    e = 0.065
    # Bottom edge (y=0)
    centers[4] = [0.26, e]
    centers[5] = [0.50, e]
    centers[6] = [0.74, e]
    # Top edge (y=1)
    centers[7] = [0.26, 1-e]
    centers[8] = [0.50, 1-e]
    centers[9] = [0.74, 1-e]
    # Left edge (x=0)
    centers[10] = [e, 0.26]
    centers[11] = [e, 0.50]
    centers[12] = [e, 0.74]
    # Right edge (x=1)
    centers[13] = [1-e, 0.26]
    centers[14] = [1-e, 0.50]
    centers[15] = [1-e, 0.74]
    
    # INTERIOR CIRCLES: Optimized hexagonal pattern with 4-3-3 arrangement
    # Row 1: 4 circles
    centers[16] = [0.20, 0.22]
    centers[17] = [0.40, 0.22]
    centers[18] = [0.60, 0.22]
    centers[19] = [0.80, 0.22]
    # Row 2: 3 circles (offset for hexagonal packing)
    centers[20] = [0.30, 0.40]
    centers[21] = [0.50, 0.40]
    centers[22] = [0.70, 0.40]
    # Row 3: 3 circles
    centers[23] = [0.20, 0.58]
    centers[24] = [0.50, 0.58]
    centers[25] = [0.80, 0.58]
    
    radii = compute_max_radii_iterative(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii_iterative(centers, max_iter=500):
    """Compute max valid radii with iterative optimization."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to nearest boundary
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)
    
    # Iteratively resolve overlaps
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                total = radii[i] + radii[j]
                if total > dist and total > 1e-12:
                    scale = dist / total
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    
    # Final pass: ensure boundary constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1-x, 1-y)
    
    return radii


# EVOLVE-BLOCK-END


def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")