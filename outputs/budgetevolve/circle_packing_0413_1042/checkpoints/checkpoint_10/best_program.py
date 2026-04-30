# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles using corner-edge-interior strategy"""
import numpy as np

def construct_packing():
    """
    Construct an optimized arrangement of 26 circles maximizing sum of radii.
    Strategy: 4 large corner circles, edge circles, and hexagonal interior.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corner circles - can touch two walls, allowing large radii
    r_corner = 0.1464  # Optimized corner radius
    centers[idx] = [r_corner, r_corner]; idx += 1
    centers[idx] = [1 - r_corner, r_corner]; idx += 1
    centers[idx] = [r_corner, 1 - r_corner]; idx += 1
    centers[idx] = [1 - r_corner, 1 - r_corner]; idx += 1
    
    # 4 edge circles per edge (16 total) - placed to maximize radii
    # Bottom edge
    for i in range(4):
        centers[idx] = [0.22 + 0.19 * i, 0.09]; idx += 1
    # Top edge  
    for i in range(4):
        centers[idx] = [0.22 + 0.19 * i, 0.91]; idx += 1
    # Left edge
    for i in range(4):
        centers[idx] = [0.09, 0.22 + 0.19 * i]; idx += 1
    # Right edge
    for i in range(4):
        centers[idx] = [0.91, 0.22 + 0.19 * i]; idx += 1
    
    # 6 interior circles in hexagonal pattern
    # Row 1: 3 circles
    for i in range(3):
        centers[idx] = [0.28 + 0.22 * i, 0.38]; idx += 1
    # Row 2: 3 circles (offset for hexagonal packing)
    for i in range(3):
        centers[idx] = [0.39 + 0.22 * i, 0.62]; idx += 1
    
    # Compute optimal radii using iterative constraint satisfaction
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum radii satisfying wall and pairwise constraints.
    Uses iterative propagation to properly handle shared constraints.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with wall constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iteratively enforce pairwise constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    # Circle i's radius limited by distance minus j's radius
                    max_r_i = dist - radii[j]
                    if max_r_i < radii[i] and max_r_i > 0:
                        radii[i] = max_r_i
                        changed = True
        if not changed:
            break
    
    return np.maximum(radii, 1e-8)


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
