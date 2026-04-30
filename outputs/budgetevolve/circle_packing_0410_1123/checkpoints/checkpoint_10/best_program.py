# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct optimized packing with corner circles, edge circles,
    and proper hexagonal interior pattern for maximum density.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corner circles (indices 0-3) - larger radius potential
    r_corner = 0.146
    centers[0] = [r_corner, r_corner]
    centers[1] = [1 - r_corner, r_corner]
    centers[2] = [r_corner, 1 - r_corner]
    centers[3] = [1 - r_corner, 1 - r_corner]
    
    # 3 circles per edge (indices 4-15) - 12 total
    edge_r = 0.128
    # Bottom edge (4-6)
    for i, x in enumerate([0.28, 0.50, 0.72]):
        centers[4 + i] = [x, edge_r]
    # Top edge (7-9)
    for i, x in enumerate([0.28, 0.50, 0.72]):
        centers[7 + i] = [x, 1 - edge_r]
    # Left edge (10-12)
    for i, y in enumerate([0.28, 0.50, 0.72]):
        centers[10 + i] = [edge_r, y]
    # Right edge (13-15)
    for i, y in enumerate([0.28, 0.50, 0.72]):
        centers[13 + i] = [1 - edge_r, y]
    
    # 10 interior circles (indices 16-25) - proper hexagonal 3-2-3-2 pattern
    h = 0.23  # horizontal spacing
    v = h * np.sqrt(3) / 2  # vertical spacing for hexagonal
    cx, cy = 0.50, 0.50  # center of interior region
    
    # Row 1: 3 circles (top)
    centers[16] = [cx - h, cy + v]
    centers[17] = [cx, cy + v]
    centers[18] = [cx + h, cy + v]
    # Row 2: 2 circles (offset)
    centers[19] = [cx - h/2, cy]
    centers[20] = [cx + h/2, cy]
    # Row 3: 3 circles
    centers[21] = [cx - h, cy - v]
    centers[22] = [cx, cy - v]
    centers[23] = [cx + h, cy - v]
    # Row 4: 2 circles (bottom offset)
    centers[24] = [cx - h/2, cy - 2*v]
    centers[25] = [cx + h/2, cy - 2*v]
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum radii ensuring no overlaps and staying within bounds.
    Uses iterative proportional scaling with multiple passes for convergence.
    
    Args:
        centers: np.array of shape (n, 2) with circle centers
        
    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to nearest boundary
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iteratively reduce overlapping pairs with multiple passes
    for _ in range(30):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist > 0:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
    
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
