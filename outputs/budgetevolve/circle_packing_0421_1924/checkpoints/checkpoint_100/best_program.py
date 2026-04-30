# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles in a unit square"""
import numpy as np

def construct_packing():
    """
    Strategic placement with variable radii optimization.
    Uses hexagonal-inspired layout with edge-optimized positioning.
    Iteratively expands and contracts radii to maximize sum.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal pattern: 5-4-5-4-5-3 rows, optimized positions
    # Place circles to maximize distance from boundaries and neighbors
    row_counts = [5, 4, 5, 4, 5, 3]
    y_positions = [0.095, 0.27, 0.45, 0.62, 0.80, 0.935]
    
    idx = 0
    for row_idx, (count, y) in enumerate(zip(row_counts, y_positions)):
        if count == 5:
            x_positions = [0.095, 0.28, 0.46, 0.64, 0.84]
        elif count == 4:
            x_positions = [0.18, 0.38, 0.58, 0.80]
        else:  # count == 3
            x_positions = [0.19, 0.50, 0.81]
        
        for x in x_positions:
            centers[idx] = [x, y]
            idx += 1
    
    radii = optimize_radii(centers)
    return centers, radii, np.sum(radii)


def optimize_radii(centers):
    """Maximize radii through iterative expansion-contraction."""
    n = centers.shape[0]
    
    # Initialize with max possible (distance to nearest boundary)
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dists[i, j] = dists[j, i] = np.linalg.norm(centers[i] - centers[j])
    
    # Iterative optimization: expand then contract
    for iteration in range(500):
        # Sort circles by potential for growth (smaller circles first)
        order = np.argsort(radii)
        
        for i in order:
            # Max radius from boundaries
            x, y = centers[i]
            max_r = min(x, y, 1-x, 1-y)
            
            # Max radius from neighbors
            for j in range(n):
                if i != j:
                    max_r = min(max_r, dists[i, j] - radii[j])
            
            # Expand slightly beyond constraint for pressure
            radii[i] = min(max_r * 1.02, max_r + 0.001)
        
        # Contract to resolve overlaps
        for _ in range(10):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    overlap = radii[i] + radii[j] - dists[i, j]
                    if overlap > 0:
                        # Proportional reduction
                        reduction = overlap / 2 + 0.0001
                        radii[i] = max(0.001, radii[i] - reduction)
                        radii[j] = max(0.001, radii[j] - reduction)
                        changed = True
            if not changed:
                break
    
    # Final constraint check
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1-x, 1-y)
    
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
