# EVOLVE-BLOCK-START
"""Hexagonal circle packing for n=26 circles in a unit square."""
import numpy as np


def construct_packing():
    """
    Hexagonal packing pattern for 26 circles in a unit square.
    Uses 5 rows with alternating 5-6-5-6-4 circles for optimal density.
    Each circle gets maximum radius based on nearest neighbor constraint.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal spacing parameters
    s = 0.19  # horizontal spacing between adjacent circles
    dy = s * np.sqrt(3) / 2  # vertical spacing for hexagonal packing
    
    idx = 0
    base_y = 0.12
    
    # Row 0: 5 circles
    y = base_y
    for i in range(5):
        centers[idx] = [0.12 + i * s, y]
        idx += 1
    
    # Row 1: 6 circles (offset by s/2 for hexagonal pattern)
    y = base_y + dy
    for i in range(6):
        centers[idx] = [0.12 - s/2 + i * s, y]
        idx += 1
    
    # Row 2: 5 circles
    y = base_y + 2 * dy
    for i in range(5):
        centers[idx] = [0.12 + i * s, y]
        idx += 1
    
    # Row 3: 6 circles (offset)
    y = base_y + 3 * dy
    for i in range(6):
        centers[idx] = [0.12 - s/2 + i * s, y]
        idx += 1
    
    # Row 4: 4 circles
    y = base_y + 4 * dy
    for i in range(4):
        centers[idx] = [0.12 + s/2 + i * s, y]
        idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute optimal radii using iterative expansion method.
    Each circle gets half the distance to its nearest constraint
    (neighbor or boundary), then we iteratively expand where possible.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initial: each circle gets half distance to nearest neighbor/boundary
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1 - x, 1 - y)  # boundary constraint
        
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, dist / 2)
        radii[i] = max_r
    
    # Iterative expansion: try to grow each circle
    for _ in range(100):
        improved = False
        for i in range(n):
            x, y = centers[i]
            # Maximum radius from boundary
            max_r = min(x, y, 1 - x, 1 - y)
            
            # Maximum radius from other circles
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, dist - radii[j])
            
            if max_r > radii[i] + 1e-9:
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
