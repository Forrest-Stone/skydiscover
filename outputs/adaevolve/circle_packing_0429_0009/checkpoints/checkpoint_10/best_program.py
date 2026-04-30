# EVOLVE-BLOCK-START
"""Hexagonal grid circle packing for n=26 circles in a unit square"""
import numpy as np

def construct_packing():
    """
    Properly spaced hexagonal packing for 26 circles.
    Pattern: 5-6-5-6-4 rows, centered in unit square.
    Uses expansion phase in radius computation for larger radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Target radius based on hexagonal packing theory
    # For 26 circles, optimal average radius ~0.088-0.095
    r = 0.092
    h = 2 * r  # horizontal spacing (centers 2r apart)
    v = np.sqrt(3) * r  # vertical spacing for hex pattern
    
    # Pattern: 5-6-5-6-4 = 26 circles
    rows = [(5, 0), (6, h/2), (5, 0), (6, h/2), (4, h)]
    
    # Center the pattern vertically
    y_start = 0.5 - 2 * v
    idx = 0
    
    for row_idx, (count, x_off) in enumerate(rows):
        y = y_start + row_idx * v
        # Center each row horizontally
        x_start = 0.5 - (count - 1) * h / 2
        for col in range(count):
            x = x_start + x_off + col * h
            centers[idx] = [x, y]
            idx += 1
    
    # Ensure centers stay within valid bounds
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum non-overlapping radii with shrink-then-expand approach."""
    n = centers.shape[0]
    
    # Initialize with border constraints
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Phase 1: Shrink to resolve overlaps
    for _ in range(100):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    scale = d / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
    
    # Phase 2: Expand circles where space allows
    for _ in range(50):
        for i in range(n):
            # Maximum radius limited by borders
            max_r = min(centers[i][0], centers[i][1], 1-centers[i][0], 1-centers[i][1])
            # Maximum radius limited by other circles
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d - radii[j])
            # Expand if possible
            radii[i] = max(radii[i], max_r)
    
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