# EVOLVE-BLOCK-START
"""Optimized hexagonal grid packing for n=26 circles in a unit square."""
import numpy as np

def construct_packing():
    """
    Hexagonal grid arrangement for 26 circles maximizing sum of radii.
    Uses pattern [5,4,5,4,5,3] with optimal spacing for unit square.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal pattern: 5, 4, 5, 4, 5, 3 = 26 circles
    rows = [5, 4, 5, 4, 5, 3]
    
    # Optimal radius for hexagonal packing: 2r + 5*r*sqrt(3) = 1
    r = 1.0 / (2 + 5 * np.sqrt(3))
    h_spacing = 2 * r
    v_spacing = r * np.sqrt(3)
    
    idx = 0
    for row_idx, count in enumerate(rows):
        y = r + row_idx * v_spacing
        row_width = (count - 1) * h_spacing
        x_start = (1 - row_width) / 2
        # Hexagonal offset for alternating rows
        if row_idx % 2 == 1:
            x_start += r
        
        for col in range(count):
            x = x_start + col * h_spacing
            centers[idx] = [x, y]
            idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Compute maximum non-overlapping radii within unit square using iterative growth."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Iteratively grow each circle to its maximum
    for _ in range(100):
        for i in range(n):
            # Max radius from borders
            r_max = min(centers[i, 0], centers[i, 1], 1-centers[i, 0], 1-centers[i, 1])
            # Reduce based on distance to other circles
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    r_max = min(r_max, max(0, dist - radii[j]))
            radii[i] = r_max
    
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
