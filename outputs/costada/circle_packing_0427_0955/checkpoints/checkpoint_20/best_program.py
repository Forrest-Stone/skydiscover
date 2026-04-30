# EVOLVE-BLOCK-START
"""Hexagonal-inspired circle packing for n=26 circles in a unit square."""
import numpy as np

def construct_packing():
    """Hexagonal grid pattern with 5 rows: 5-6-5-6-4 circles."""
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    # Hexagonal spacing optimized for unit square
    dx, dy = 0.18, 0.156  # horizontal and vertical spacing
    y_positions = [0.11, 0.266, 0.422, 0.578, 0.734, 0.89]
    row_counts = [5, 6, 5, 6, 4]
    for row, (y, count) in enumerate(zip(y_positions[:5], row_counts)):
        offset = dx/2 if row % 2 == 1 else 0
        x_start = 0.1 + offset if count == 6 else 0.1
        for i in range(count):
            centers[idx] = [x_start + i * dx, y]
            idx += 1
    centers = np.clip(centers, 0.02, 0.98)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Compute max radii respecting borders and non-overlap."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    for _ in range(20):  # Iterate to equilibrate
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    s = d / (radii[i] + radii[j])
                    radii[i], radii[j] = radii[i]*s, radii[j]*s
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
