# EVOLVE-BLOCK-START
"""Circle packing for n=26 using linear programming for exact radii computation"""
import numpy as np
from scipy.optimize import linprog

def construct_packing():
    """
    Hexagonal packing with LP-computed exact maximum radii.
    Pattern: 5-4-5-4-5-3 = 26 circles.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    spacing = 0.2
    y_step = spacing * np.sqrt(3) / 2
    
    idx = 0
    for i in range(5):
        centers[idx] = [0.1 + i * spacing, 0.1]
        idx += 1
    for i in range(4):
        centers[idx] = [0.2 + i * spacing, 0.1 + y_step]
        idx += 1
    for i in range(5):
        centers[idx] = [0.1 + i * spacing, 0.1 + 2*y_step]
        idx += 1
    for i in range(4):
        centers[idx] = [0.2 + i * spacing, 0.1 + 3*y_step]
        idx += 1
    for i in range(5):
        centers[idx] = [0.1 + i * spacing, 0.1 + 4*y_step]
        idx += 1
    for x in [0.1, 0.5, 0.9]:
        centers[idx] = [x, 0.95]
        idx += 1
    
    radii = compute_max_radii_lp(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii_lp(centers):
    """Compute exact maximum radii using linear programming.
    
    Formulates the problem as: max sum(r_i) subject to:
    - r_i + r_j <= d_ij for all pairs (non-overlap)
    - 0 <= r_i <= wall_dist[i] (stay within walls)
    """
    n = len(centers)
    wall_dist = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Build pairwise constraints: r_i + r_j <= d_ij
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    row = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            A_ub[row, i] = 1
            A_ub[row, j] = 1
            b_ub[row] = d
            row += 1
    
    # Maximize sum(r_i) -> minimize -sum(r_i)
    c = -np.ones(n)
    bounds = [(0, wall_dist[i]) for i in range(n)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        # Fallback to iterative method
        radii = wall_dist.copy()
        for _ in range(100):
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if radii[i] + radii[j] > d and d > 1e-10:
                        total = radii[i] + radii[j]
                        radii[i] = d * radii[i] / total
                        radii[j] = d * radii[j] / total
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