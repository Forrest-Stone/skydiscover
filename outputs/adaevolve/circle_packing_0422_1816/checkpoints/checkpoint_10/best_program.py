# EVOLVE-BLOCK-START
"""Optimized hexagonal packing for n=26 circles with 4-5-6-6-5 pattern."""
import numpy as np

def construct_packing():
    """
    Hexagonal packing with 4-5-6-6-5 pattern using optimized spacing.
    Pattern places circles to maximize corner space for larger radii.
    Uses dx=0.18 with hexagonal vertical spacing for better density.
    """
    n = 26
    centers = np.zeros((n, 2))
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    idx = 0
    
    # Pattern: 4-5-6-6-5 = 26 circles
    # Start near bottom edge to maximize corner expansion
    y_start = 0.10
    rows = [(4, y_start), (5, y_start + dy), (6, y_start + 2*dy),
            (6, y_start + 3*dy), (5, y_start + 4*dy)]
    
    for count, y in rows:
        row_width = (count - 1) * dx
        x_start = 0.5 - row_width / 2
        for col in range(count):
            centers[idx] = [x_start + col * dx, y]
            idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Iteratively compute maximum feasible radii with convergence."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with boundary constraints
    for i in range(n):
        radii[i] = min(centers[i, 0], centers[i, 1], 
                       1 - centers[i, 0], 1 - centers[i, 1])
    
    # Iteratively refine with circle-circle constraints
    for _ in range(200):
        for i in range(n):
            max_r = min(centers[i, 0], centers[i, 1], 
                       1 - centers[i, 0], 1 - centers[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(0.001, max_r)
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
