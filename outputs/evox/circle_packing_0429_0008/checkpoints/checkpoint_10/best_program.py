# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Construct hexagonal-inspired packing for 26 circles in a unit square.
    Uses 5 rows with pattern 5-5-6-5-5 = 26 circles.
    Circles positioned near walls maximize their radii.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    
    idx = 0
    
    # Row 1 (bottom): 5 circles near bottom wall
    for x in [0.10, 0.30, 0.50, 0.70, 0.90]:
        centers[idx] = [x, 0.10]
        idx += 1
    
    # Row 2: 5 circles (offset for hexagonal packing)
    for x in [0.20, 0.40, 0.60, 0.80, 1.00]:
        centers[idx] = [x - 0.10, 0.30]
        idx += 1
    
    # Row 3 (center): 6 circles
    for x in [0.08, 0.24, 0.40, 0.56, 0.72, 0.88]:
        centers[idx] = [x, 0.50]
        idx += 1
    
    # Row 4: 5 circles (offset)
    for x in [0.20, 0.40, 0.60, 0.80, 1.00]:
        centers[idx] = [x - 0.10, 0.70]
        idx += 1
    
    # Row 5 (top): 5 circles near top wall
    for x in [0.10, 0.30, 0.50, 0.70, 0.90]:
        centers[idx] = [x, 0.90]
        idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum valid radii using iterative constraint resolution.
    Each circle's radius is limited by walls and other circles.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with wall constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iteratively resolve circle-circle constraints
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    total = radii[i] + radii[j]
                    if total > 1e-10:
                        radii[i] = radii[i] * dist / total
                        radii[j] = radii[j] * dist / total
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
