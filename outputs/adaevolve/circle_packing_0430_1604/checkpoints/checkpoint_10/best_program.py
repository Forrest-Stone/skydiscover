# EVOLVE-BLOCK-START
"""Hexagonal-inspired circle packing for n=26 circles in a unit square."""
import numpy as np


def construct_packing():
    """
    Optimized hexagonal packing for 26 circles in a unit square.
    Uses staggered rows (5-4-5-4-5-3 pattern) with maximized spacing.
    Key insight: r constrained by vertical fit: r <= 1/(2+5*sqrt(3)) ≈ 0.094
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Optimized base radius - near maximum that fits the pattern
    r = 0.094
    
    dx = 2 * r  # Horizontal spacing
    dy = r * np.sqrt(3)  # Vertical spacing for hexagonal packing
    
    # Pattern: (count, x_start) - staggered for hexagonal arrangement
    # Rows 0,2,4 have 5 circles; rows 1,3 have 4 (offset); row 5 has 3
    rows = [(5, r), (4, 2*r), (5, r), (4, 2*r), (5, r), (3, 2*r)]
    
    idx = 0
    y = r
    for count, x_start in rows:
        for i in range(count):
            x = x_start + i * dx
            centers[idx] = [x, y]
            idx += 1
        y += dy
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum valid radii for given center positions.
    Uses iterative proportional scaling - optimal for sum maximization.
    Pre-computes pairwise distances for efficiency.
    """
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Pre-compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            dists[i, j] = dists[j, i] = d
    
    # Iteratively resolve overlaps
    for _ in range(50):
        converged = True
        for i in range(n):
            for j in range(i + 1, n):
                if radii[i] + radii[j] > dists[i, j]:
                    scale = dists[i, j] / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    converged = False
        if converged:
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
