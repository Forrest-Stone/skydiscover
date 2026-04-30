# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized hexagonal packing for 26 circles in a unit square.
    Uses 5 rows with pattern 5-6-5-6-4, placing circles at radius
    distance from boundaries for maximum space utilization.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Optimal hexagonal arrangement: 5 rows alternating 5,6,5,6,4 circles
    row_counts = [5, 6, 5, 6, 4]
    n_rows = 5
    
    # For hexagonal packing: horizontal spacing = 2r, vertical = r*sqrt(3)
    # With max row of 6 circles: width = 10r + 2r = 12r = 1, so r = 1/12
    # Vertical: 4 gaps of r*sqrt(3) + 2r = 1, giving r ≈ 0.1
    
    r_estimate = 1.0 / 12  # Conservative estimate based on width constraint
    v_gap = r_estimate * np.sqrt(3)  # Vertical spacing for hexagonal packing
    
    # Position rows to use full height: first at y=r, last at y=1-r
    y_positions = [r_estimate + i * v_gap for i in range(n_rows)]
    # Adjust last row to be at 1-r
    scale_y = (1 - 2*r_estimate) / (y_positions[-1] - r_estimate)
    y_positions = [r_estimate + scale_y * (y - r_estimate) for y in y_positions]
    
    idx = 0
    for row, count in enumerate(row_counts):
        y = y_positions[row]
        # Horizontal spacing: place first and last circles at r from edges
        if count == 6:
            x_positions = np.linspace(r_estimate, 1 - r_estimate, count)
        else:
            # Center the row with 'count' circles
            h_spacing = (1 - 2*r_estimate) / (count - 1) if count > 1 else 0
            x_positions = [r_estimate + i * h_spacing for i in range(count)]
        
        for x in x_positions:
            centers[idx] = [x, y]
            idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum radii using iterative constraint satisfaction.
    Each circle's radius is limited by border distance and neighbor distances.
    Uses multiple iterations with proper convergence for optimal sizing.
    """
    n = centers.shape[0]
    # Initialize with border constraints
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Iterate to satisfy all pairwise constraints
    for iteration in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                max_sum = dist
                if radii[i] + radii[j] > max_sum:
                    scale = max_sum / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
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
