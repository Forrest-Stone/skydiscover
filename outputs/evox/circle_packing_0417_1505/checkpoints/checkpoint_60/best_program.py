# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized packing for 26 circles using strategic placement
    with edge-biased positioning. Places circles at corners (largest),
    edges (medium), and interior (smaller) to maximize total radii.
    Then applies local position optimization to refine.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corner circles - can be largest due to 2-edge constraint
    r_corner = 0.12
    centers[0] = [r_corner, r_corner]
    centers[1] = [1 - r_corner, r_corner]
    centers[2] = [r_corner, 1 - r_corner]
    centers[3] = [1 - r_corner, 1 - r_corner]
    
    # 8 edge circles - medium size due to 1-edge constraint
    e = 0.07
    # Bottom and top edges
    centers[4] = [0.28, e]
    centers[5] = [0.72, e]
    centers[6] = [0.28, 1 - e]
    centers[7] = [0.72, 1 - e]
    # Left and right edges
    centers[8] = [e, 0.28]
    centers[9] = [e, 0.72]
    centers[10] = [1 - e, 0.28]
    centers[11] = [1 - e, 0.72]
    
    # 14 interior circles - hexagonal pattern with 5-4-5 pattern
    idx = 12
    row_counts = [5, 4, 5]
    y_coords = [0.25, 0.50, 0.75]
    
    for row, (count, y) in enumerate(zip(row_counts, y_coords)):
        if count == 5:
            x_coords = [0.18, 0.34, 0.50, 0.66, 0.82]
        else:
            # Staggered for hexagonal effect
            x_coords = [0.26, 0.42, 0.58, 0.74]
        for x in x_coords:
            centers[idx] = [x, y]
            idx += 1
    
    # Local optimization: refine positions to maximize radii sum
    centers = optimize_positions(centers)
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def optimize_positions(centers):
    """
    Hill-climbing optimization of circle positions.
    Perturbs each circle slightly and keeps improvements.
    """
    n = centers.shape[0]
    best_centers = centers.copy()
    best_sum = np.sum(compute_max_radii(best_centers))
    
    step = 0.02
    for _ in range(50):  # Multiple passes
        improved = False
        for i in range(n):
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                          (step/np.sqrt(2), step/np.sqrt(2)),
                          (-step/np.sqrt(2), -step/np.sqrt(2))]:
                new_centers = best_centers.copy()
                new_x = np.clip(best_centers[i, 0] + dx, 0.01, 0.99)
                new_y = np.clip(best_centers[i, 1] + dy, 0.01, 0.99)
                new_centers[i] = [new_x, new_y]
                new_sum = np.sum(compute_max_radii(new_centers))
                if new_sum > best_sum + 1e-6:
                    best_centers = new_centers
                    best_sum = new_sum
                    improved = True
        if not improved:
            step *= 0.7  # Reduce step size
            if step < 1e-4:
                break
    return best_centers


def compute_max_radii(centers):
    """
    Compute maximum radii via shrink-grow iteration.
    First shrinks overlapping pairs, then grows each circle to its
    maximum possible radius given neighbors and borders. Repeats
    until convergence for true maximum radii.
    """
    n = centers.shape[0]
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Precompute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = dists[j, i] = d
    
    # Iterate: shrink overlaps, then grow where possible
    for _ in range(300):
        changed = False
        
        # Shrink overlapping pairs proportionally
        for i in range(n):
            for j in range(i+1, n):
                if radii[i] + radii[j] > dists[i, j]:
                    scale = dists[i, j] / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        
        # Grow each circle to maximum possible radius
        for i in range(n):
            x, y = centers[i]
            border_limit = min(x, y, 1-x, 1-y)
            neighbor_limits = [dists[i, j] - radii[j] for j in range(n) if j != i]
            max_r = min(border_limit, min(neighbor_limits))
            if max_r > radii[i]:
                radii[i] = max_r
                changed = True
        
        if not changed:
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
