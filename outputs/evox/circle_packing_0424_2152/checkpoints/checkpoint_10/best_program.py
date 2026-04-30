# EVOLVE-BLOCK-START
"""Hexagonal packing for 26 circles in a unit square with iterative radius optimization."""
import numpy as np


def construct_packing():
    """
    Hexagonal packing pattern for 26 circles maximizing sum of radii.
    Uses 5-4-5-4-5-3 row pattern with optimized spacing parameters.
    Hexagonal arrangement allows efficient space utilization.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Optimized hexagonal spacing parameters
    dx = 0.19  # horizontal spacing
    dy = dx * np.sqrt(3) / 2  # vertical spacing for hexagonal packing
    y_start = 0.1  # start position, gives room to expand
    
    idx = 0
    
    # Row 0: 5 circles along bottom
    for i in range(5):
        centers[idx] = [0.12 + i * dx, y_start]
        idx += 1
    
    # Row 1: 4 circles (offset for hexagonal pattern)
    for i in range(4):
        centers[idx] = [0.12 + dx/2 + i * dx, y_start + dy]
        idx += 1
    
    # Row 2: 5 circles
    for i in range(5):
        centers[idx] = [0.12 + i * dx, y_start + 2 * dy]
        idx += 1
    
    # Row 3: 4 circles (offset)
    for i in range(4):
        centers[idx] = [0.12 + dx/2 + i * dx, y_start + 3 * dy]
        idx += 1
    
    # Row 4: 5 circles
    for i in range(5):
        centers[idx] = [0.12 + i * dx, y_start + 4 * dy]
        idx += 1
    
    # Row 5: 3 circles (centered at top)
    for i in range(3):
        centers[idx] = [0.12 + dx/2 + i * dx, y_start + 5 * dy]
        idx += 1
    
    radii = compute_max_radii_iterative(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii_iterative(centers, max_iterations=200):
    """
    Iteratively compute maximum radii for all circles.
    Each iteration refines radii based on current state.
    Converges to locally optimal radius distribution.
    Uses Gauss-Seidel style updates for faster convergence.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with boundary constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Precompute pairwise distances for efficiency
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Iteratively refine radii using Gauss-Seidel updates
    for iteration in range(max_iterations):
        max_change = 0.0
        
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            
            # Constraint from each other circle
            for j in range(n):
                if i != j:
                    max_r = min(max_r, distances[i, j] - radii[j])
            
            new_r = max(max_r, 1e-6)
            max_change = max(max_change, abs(new_r - radii[i]))
            radii[i] = new_r
        
        # Check convergence
        if max_change < 1e-8:
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
