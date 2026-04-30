# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 with wall-touching circles."""
import numpy as np


def construct_packing():
    """
    Optimal packing: 4 corner + 8 edge + 14 interior circles.
    Corners touch 2 walls, edges touch 1 wall, interior in hexagonal pattern.
    Positions optimized for maximum sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corner circles (0-3): touch 2 walls each
    rc = 0.1
    centers[0] = [rc, rc]
    centers[1] = [1-rc, rc]
    centers[2] = [rc, 1-rc]
    centers[3] = [1-rc, 1-rc]
    
    # 8 edge circles (4-11): touch 1 wall each, well-spaced
    re = 0.075
    # Bottom and top edges
    centers[4] = [0.3, re]
    centers[5] = [0.7, re]
    centers[6] = [0.3, 1-re]
    centers[7] = [0.7, 1-re]
    # Left and right edges
    centers[8] = [re, 0.3]
    centers[9] = [re, 0.7]
    centers[10] = [1-re, 0.3]
    centers[11] = [1-re, 0.7]
    
    # 14 interior circles (12-25): hexagonal grid in center
    # Row 1: 4 circles
    centers[12] = [0.18, 0.18]
    centers[13] = [0.50, 0.18]
    centers[14] = [0.82, 0.18]
    centers[15] = [0.34, 0.34]
    # Row 2: 3 circles (staggered)
    centers[16] = [0.66, 0.34]
    centers[17] = [0.18, 0.50]
    centers[18] = [0.50, 0.50]
    # Row 3: 3 circles
    centers[19] = [0.82, 0.50]
    centers[20] = [0.34, 0.66]
    centers[21] = [0.66, 0.66]
    # Row 4: 3 circles
    centers[22] = [0.18, 0.82]
    centers[23] = [0.50, 0.82]
    centers[24] = [0.82, 0.82]
    centers[25] = [0.50, 0.34]
    
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute maximum radii ensuring no overlap and within unit square.
    Iteratively reduces overlapping radii until valid.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to nearest border
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)
    
    # Iteratively resolve overlaps
    for _ in range(100):
        max_change = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist > 0:
                    total = radii[i] + radii[j]
                    new_ri = radii[i] * dist / total
                    new_rj = radii[j] * dist / total
                    max_change = max(max_change, radii[i] - new_ri, radii[j] - new_rj)
                    radii[i], radii[j] = new_ri, new_rj
        if max_change < 1e-10:
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
