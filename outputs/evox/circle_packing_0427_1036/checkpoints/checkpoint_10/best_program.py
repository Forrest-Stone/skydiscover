# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized placement of 26 circles using hexagonal-inspired packing.
    4 corners + 8 edges + 14 interior in staggered rows.
    Corners positioned for maximum radius, interior uses hexagonal pattern.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corner circles - positioned for large radii
    for cx, cy in [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]:
        centers[idx] = [cx, cy]
        idx += 1
    
    # 8 edge circles - spaced to maximize radii
    for x in [0.32, 0.5, 0.68]:
        centers[idx] = [x, 0.065]
        idx += 1
    for x in [0.32, 0.5, 0.68]:
        centers[idx] = [x, 0.935]
        idx += 1
    centers[idx] = [0.065, 0.5]
    idx += 1
    centers[idx] = [0.935, 0.5]
    idx += 1
    
    # 14 interior circles in hexagonal staggered pattern
    # Row 1: 4 circles
    for x in [0.22, 0.40, 0.58, 0.78]:
        centers[idx] = [x, 0.22]
        idx += 1
    # Row 2: 3 circles (offset)
    for x in [0.31, 0.49, 0.67]:
        centers[idx] = [x, 0.39]
        idx += 1
    # Row 3: 4 circles
    for x in [0.22, 0.40, 0.58, 0.78]:
        centers[idx] = [x, 0.56]
        idx += 1
    # Row 4: 3 circles (offset)
    for x in [0.31, 0.49, 0.67]:
        centers[idx] = [x, 0.73]
        idx += 1
    
    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum radii using iterative constraint satisfaction.
    Each circle limited by boundaries and neighbors. Uses iterative
    reduction to find valid radii, then expands to maximize sum.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to nearest boundary
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iteratively resolve overlaps
    for iteration in range(200):
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    # Reduce proportionally to current radii
                    total = radii[i] + radii[j]
                    radii[i] -= overlap * radii[i] / total * 1.01
                    radii[j] -= overlap * radii[j] / total * 1.01
                    max_overlap = max(max_overlap, overlap)
        
        if max_overlap < 1e-10:
            break
    
    # Ensure positive radii
    radii = np.maximum(radii, 1e-6)
    
    # Expand phase: try to grow each circle
    for _ in range(50):
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, dist - radii[j])
            radii[i] = max(radii[i], min(max_r, radii[i] * 1.02))
    
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
