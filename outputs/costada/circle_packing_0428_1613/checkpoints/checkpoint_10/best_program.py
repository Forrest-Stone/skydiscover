# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using hexagonal grid pattern."""
import numpy as np


def construct_packing():
    """
    Construct a hexagonal-based packing of 26 circles in a unit square.
    Uses staggered rows to maximize packing density, with circles of equal size
    arranged in a hexagonal lattice pattern.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal packing with equal radius circles
    # For n=26, use 6 rows with alternating 4-5 pattern (4+5+4+5+4+4=26)
    r = 0.1  # Target radius for tight hexagonal packing
    dy = r * np.sqrt(3)  # Vertical spacing for hex pattern
    dx = 2 * r  # Horizontal spacing
    
    idx = 0
    y_offset = r  # Start from bottom
    
    # Row pattern: 5,4,5,4,5,3 (bottom to top) = 26 circles
    row_counts = [5, 4, 5, 4, 5, 3]
    
    for row_idx, count in enumerate(row_counts):
        y = y_offset + row_idx * dy
        # Stagger odd rows
        x_start = r if row_idx % 2 == 0 else 2 * r
        for col in range(count):
            x = x_start + col * dx
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
    
    # Adjust to fit better in unit square
    centers = np.clip(centers, 0.08, 0.92)
    
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """Compute maximum radii ensuring no overlap and within unit square."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                total = radii[i] + radii[j]
                radii[i] = radii[i] * dist / total
                radii[j] = radii[j] * dist / total
    
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
