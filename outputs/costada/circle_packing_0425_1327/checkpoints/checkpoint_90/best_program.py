# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using corner-edge-interior strategy with iterative expansion."""
import numpy as np


def construct_packing():
    """
    Strategic placement: 4 corners (large), 12 edges (medium), 10 interior (varied).
    Uses iterative expansion to maximize each circle's radius.
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corners - positioned for maximum expansion potential
    r_c = 0.158
    centers[idx] = [r_c, r_c]; idx += 1
    centers[idx] = [1-r_c, r_c]; idx += 1
    centers[idx] = [r_c, 1-r_c]; idx += 1
    centers[idx] = [1-r_c, 1-r_c]; idx += 1
    
    # 12 edge circles (3 per side) - close to boundaries
    for x in [0.30, 0.50, 0.70]: centers[idx] = [x, 0.055]; idx += 1  # bottom
    for x in [0.30, 0.50, 0.70]: centers[idx] = [x, 0.945]; idx += 1  # top
    for y in [0.30, 0.50, 0.70]: centers[idx] = [0.055, y]; idx += 1  # left
    for y in [0.30, 0.50, 0.70]: centers[idx] = [0.945, y]; idx += 1  # right
    
    # 10 interior circles in optimized hexagonal-like pattern
    for i in range(4): centers[idx] = [0.22 + i*0.19, 0.22]; idx += 1
    for i in range(3): centers[idx] = [0.315 + i*0.19, 0.385]; idx += 1
    for i in range(3): centers[idx] = [0.22 + i*0.19, 0.55]; idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Iteratively expand each circle to maximum size respecting all constraints."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    for _ in range(300):
        for i in range(n):
            # Maximum radius from boundaries
            r_bound = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            # Maximum radius from other circles
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    r_bound = min(r_bound, dist - radii[j])
            radii[i] = max(0, r_bound)
    
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
