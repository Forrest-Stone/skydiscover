# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized hexagonal packing for 26 circles in a unit square.
    Uses 6 rows with pattern 4-5-4-5-4-4 for better edge coverage.
    Hexagonal staggering for odd rows. More rows = more circles near
    top/bottom edges where they can grow larger due to border proximity.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 6 rows with alternating counts: better edge coverage than 5 rows
    row_counts = [4, 5, 4, 5, 4, 4]
    n_rows = 6
    
    # Radius estimate based on row spacing
    r_estimate = 1.0 / 11.0
    v_gap = r_estimate * np.sqrt(3)
    
    # Position rows to span full height
    y_positions = [r_estimate + i * v_gap for i in range(n_rows)]
    scale_y = (1 - 2*r_estimate) / (y_positions[-1] - r_estimate) if y_positions[-1] > r_estimate else 1.0
    y_positions = [r_estimate + scale_y * (y - r_estimate) for y in y_positions]
    
    idx = 0
    for row, count in enumerate(row_counts):
        y = y_positions[row]
        
        # Horizontal spacing centered in unit square
        total_width = 1 - 2*r_estimate
        h_spacing = total_width / (count - 1) if count > 1 else 0
        
        # Hexagonal staggering for odd rows
        x_positions = [r_estimate + i * h_spacing for i in range(count)]
        if row % 2 == 1 and count > 1:
            x_positions = [x + h_spacing/2 for x in x_positions]
            if x_positions[-1] > 1 - r_estimate:
                x_positions = [x - (x_positions[-1] - (1-r_estimate)) for x in x_positions]
        
        for x in x_positions:
            centers[idx] = [x, y]
            idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


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
    for _ in range(200):
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
