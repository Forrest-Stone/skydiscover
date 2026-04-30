# EVOLVE-BLOCK-START
"""Multi-pattern hexagonal packing for n=26 circles in a unit square"""
import numpy as np


def construct_packing():
    """
    Try multiple hexagonal row patterns and select the best.
    Uses numpy.argmax to find the optimal pattern.
    """
    # Multiple valid patterns (each sums to 26)
    patterns = [
        [5, 4, 5, 4, 5, 3],
        [4, 5, 4, 5, 4, 4],
        [5, 5, 5, 5, 4, 2],
        [6, 4, 5, 4, 5, 2],
        [3, 5, 5, 5, 5, 3],
        [4, 4, 5, 5, 4, 4],
        [4, 5, 5, 5, 5, 2],
        [5, 4, 4, 5, 4, 4],
        [4, 4, 4, 5, 5, 4],
        [3, 4, 5, 5, 5, 4],
        [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6],
        [5, 6, 5, 5, 5],
        [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 4, 2],
        [4, 4, 4, 4, 4, 4, 2],
        [3, 4, 4, 4, 4, 4, 3],
        [7, 6, 7, 6],
        [6, 7, 6, 7],
    ]
    
    sums = []
    all_centers = []
    all_radii = []
    
    for pattern in patterns:
        centers = build_hexagonal_pattern(pattern)
        radii = compute_max_radii(centers)
        sums.append(np.sum(radii))
        all_centers.append(centers)
        all_radii.append(radii)
    
    best_idx = np.argmax(sums)
    return all_centers[best_idx], all_radii[best_idx], sums[best_idx]


def build_hexagonal_pattern(rows):
    """Build hexagonal lattice positions for given row pattern."""
    n = sum(rows)
    centers = np.zeros((n, 2))
    num_rows = len(rows)
    max_count = max(rows)
    
    # Calculate r based on both horizontal and vertical constraints
    r_h = 1 / (2 * max_count)
    r_v = 1 / ((num_rows - 1) * np.sqrt(3) + 2) if num_rows > 1 else 0.5
    r = min(r_h, r_v)
    
    h_spacing = 2 * r
    v_spacing = r * np.sqrt(3)
    
    idx = 0
    for row, count in enumerate(rows):
        y = r + row * v_spacing
        row_width = (count - 1) * h_spacing if count > 1 else 0
        x_start = r + (1 - 2*r - row_width) / 2
        if row % 2 == 1:
            x_start += r
        for c in range(count):
            centers[idx] = [x_start + c * h_spacing, y]
            idx += 1
    
    return centers


def compute_max_radii(centers):
    """Iteratively expand radii to maximize sum while respecting constraints."""
    n = len(centers)
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    for _ in range(100):
        for i in range(n):
            x, y = centers[i]
            r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    r = min(r, d - radii[j])
            radii[i] = max(0, r)
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
