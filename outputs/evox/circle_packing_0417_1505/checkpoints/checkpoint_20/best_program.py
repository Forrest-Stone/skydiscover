# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized hexagonal packing for 26 circles in a unit square.
    Uses 5 rows with pattern 5-6-5-6-4 with proper hexagonal staggering.
    Alternating rows are offset by half the horizontal spacing for optimal
    density. Uses larger initial radius estimate for better results.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal arrangement: 5 rows alternating 5,6,5,6,4 circles
    row_counts = [5, 6, 5, 6, 4]
    n_rows = 5
    
    # Use optimistic radius estimate - iterative solver will adjust
    r_estimate = 1.0 / 10.5
    v_gap = r_estimate * np.sqrt(3)
    
    # Position rows to span full height
    y_positions = [r_estimate + i * v_gap for i in range(n_rows)]
    scale_y = (1 - 2*r_estimate) / (y_positions[-1] - r_estimate) if y_positions[-1] > r_estimate else 1.0
    y_positions = [r_estimate + scale_y * (y - r_estimate) for y in y_positions]
    
    idx = 0
    for row, count in enumerate(row_counts):
        y = y_positions[row]
        
        # Horizontal spacing for this row
        h_spacing = (1 - 2*r_estimate) / (count - 1) if count > 1 else 0
        x_positions = [r_estimate + i * h_spacing for i in range(count)]
        
        # Apply hexagonal staggering: offset alternate rows by half spacing
        if row % 2 == 1:
            offset = h_spacing / 2
            x_positions = [x + offset for x in x_positions]
            # Ensure circles stay within bounds while maintaining spacing
            if x_positions[-1] > 1 - r_estimate:
                shift = x_positions[-1] - (1 - r_estimate)
                x_positions = [x - shift for x in x_positions]
        
        for x in x_positions:
            centers[idx] = [x, y]
            idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum radii using iterative constraint satisfaction.
    Initializes with border constraints, then iteratively resolves
    overlaps by scaling pairs of overlapping circles proportionally.
    """
    n = centers.shape[0]
    # Initialize with border constraints
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    # Precompute distances for efficiency
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Iterate to satisfy all pairwise constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                max_sum = dists[i, j]
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
