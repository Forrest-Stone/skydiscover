# EVOLVE-BLOCK-START
"""Hexagonal circle packing for n=26 circles with optimized radii"""
import numpy as np

def construct_packing():
    """Hexagonal pattern [5,4,5,4,5,3] rows with corner optimization."""
    n = 26
    centers = np.zeros((n, 2))
    r = 0.1  # Base radius for hexagonal spacing
    
    # Hexagonal pattern: rows of 5,4,5,4,5,3 circles
    row_counts = [5, 4, 5, 4, 5, 3]
    y_spacing = r * np.sqrt(3)
    idx = 0
    
    for row, count in enumerate(row_counts):
        y = r + row * y_spacing
        if count == 5:
            x_positions = [r + i * 2 * r for i in range(5)]
        elif count == 4:
            x_positions = [2 * r + i * 2 * r for i in range(4)]
        else:  # count == 3
            x_positions = [r + i * 2 * r for i in range(3)]
        
        for x in x_positions:
            centers[idx] = [x, y]
            idx += 1
    
    # Compute optimized radii
    radii = compute_radii(centers)
    return centers, radii, np.sum(radii)

def compute_radii(centers):
    """Compute maximum valid radii through iterative constraint satisfaction."""
    n = len(centers)
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Iterate to expand radii while respecting constraints
    for _ in range(100):
        for i in range(n):
            # Boundary constraint
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            # Circle-circle constraints
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
