# EVOLVE-BLOCK-START
"""Hexagonal grid circle packing for n=26 circles in unit square"""
import numpy as np

def construct_packing():
    """Hexagonal-inspired packing with 5 rows: 5-6-5-6-4 pattern."""
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    dy = 0.19  # vertical spacing between rows
    y_pos = [0.11, 0.30, 0.49, 0.68, 0.87]
    counts = [5, 6, 5, 6, 4]
    for row, (y, cnt) in enumerate(zip(y_pos, counts)):
        dx = 0.85 / max(cnt - 1, 1)
        x0 = 0.075 if cnt == 6 else 0.11
        for i in range(cnt):
            centers[idx] = [x0 + i * dx, y]
            idx += 1
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Compute max radii with iterative equilibrium."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    for _ in range(50):
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
