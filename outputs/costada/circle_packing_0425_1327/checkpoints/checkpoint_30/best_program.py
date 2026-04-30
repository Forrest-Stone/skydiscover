# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using hexagonal grid."""
import numpy as np


def construct_packing():
    """
    Construct a hexagonal grid arrangement of 26 circles in a unit square.
    Uses pattern [5,4,5,4,5,3]=26 circles with proper hexagonal spacing.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal grid: rows with alternating counts for efficient packing
    rows = [5, 4, 5, 4, 5, 3]
    
    # Optimal radius estimate for hexagonal packing
    # Horizontal spacing = 2r, Vertical spacing = r*sqrt(3)
    r = 0.095
    h = 2 * r
    v = r * np.sqrt(3)
    
    idx = 0
    y = r
    for row_idx, count in enumerate(rows):
        # Center each row horizontally
        row_width = h * (count - 1) if count > 1 else 0
        x_start = (1 - row_width) / 2
        # Offset alternating rows for hexagonal pattern
        if row_idx % 2 == 1:
            x_start += r
        for col in range(count):
            x = x_start + col * h
            centers[idx] = [x, y]
            idx += 1
        y += v
    
    centers = np.clip(centers, 0.02, 0.98)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum radii respecting boundaries and non-overlap constraints."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Iteratively satisfy pairwise constraints
    for _ in range(100):
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    total = radii[i] + radii[j]
                    radii[i] = dist * radii[i] / total
                    radii[j] = dist * radii[j] / total
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
