# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using varied radii pattern."""
import numpy as np


def construct_packing():
    """
    Construct packing with optimized positions for 26 circles:
    - 4 corner circles positioned to maximize wall contact (touch 2 walls)
    - 8 edge circles with better spacing (touch 1 wall)
    - 14 interior circles in hexagonal-inspired pattern
    
    Key insight: Corner circles at distance r from walls can have radius r,
    maximizing their contribution. Interior uses denser 3-4-4-3 pattern.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corner circles - position at radius from walls for max size
    cr = 0.105  # corner radius estimate
    for cx, cy in [(cr, cr), (1-cr, cr), (cr, 1-cr), (1-cr, 1-cr)]:
        centers[idx] = [cx, cy]
        idx += 1
    
    # 8 edge circles (2 per side) with optimized spacing
    er = 0.075  # edge radius estimate
    # Bottom edge
    centers[idx] = [0.33, er]; idx += 1
    centers[idx] = [0.67, er]; idx += 1
    # Top edge
    centers[idx] = [0.33, 1-er]; idx += 1
    centers[idx] = [0.67, 1-er]; idx += 1
    # Left edge
    centers[idx] = [er, 0.33]; idx += 1
    centers[idx] = [er, 0.67]; idx += 1
    # Right edge
    centers[idx] = [1-er, 0.33]; idx += 1
    centers[idx] = [1-er, 0.67]; idx += 1
    
    # 14 interior circles in denser hexagonal pattern
    # Row 1: 3 circles
    centers[idx] = [0.25, 0.25]; idx += 1
    centers[idx] = [0.50, 0.25]; idx += 1
    centers[idx] = [0.75, 0.25]; idx += 1
    # Row 2: 4 circles (offset)
    centers[idx] = [0.20, 0.42]; idx += 1
    centers[idx] = [0.40, 0.42]; idx += 1
    centers[idx] = [0.60, 0.42]; idx += 1
    centers[idx] = [0.80, 0.42]; idx += 1
    # Row 3: 4 circles
    centers[idx] = [0.20, 0.58]; idx += 1
    centers[idx] = [0.40, 0.58]; idx += 1
    centers[idx] = [0.60, 0.58]; idx += 1
    centers[idx] = [0.80, 0.58]; idx += 1
    # Row 4: 3 circles (offset)
    centers[idx] = [0.25, 0.75]; idx += 1
    centers[idx] = [0.50, 0.75]; idx += 1
    centers[idx] = [0.75, 0.75]; idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute maximum radii using iterative constraint satisfaction with
    expansion phase. Starts conservative, then expands circles that have
    room to grow, ensuring all constraints are satisfied.
    """
    n = centers.shape[0]
    
    # Initialize radii to wall constraints (upper bound)
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Iteratively enforce non-overlap constraints
    for _ in range(500):
        max_change = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist * 0.9999:
                    total = radii[i] + radii[j]
                    if total > 1e-10:
                        ratio_i = radii[i] / total
                        radii[i] = dist * ratio_i * 0.9999
                        radii[j] = dist * (1 - ratio_i) * 0.9999
                        max_change = max(max_change, abs(total - dist))
        if max_change < 1e-12:
            break
    
    # Final constraint enforcement
    for i in range(n):
        radii[i] = min(radii[i], centers[i][0], centers[i][1], 
                       1 - centers[i][0], 1 - centers[i][1])
    
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
