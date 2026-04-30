# EVOLVE-BLOCK-START
"""Linear programming approach for optimal radii given positions, with position optimization"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree


def construct_packing():
    """
    Use linear programming to optimally solve for radii given positions.
    For fixed positions, maximizing sum of radii is a LINEAR PROGRAM:
    - Maximize sum(r_i) subject to:
      - r_i + r_j <= dist(i,j) for all pairs (non-overlap)
      - r_i <= min(x_i, y_i, 1-x_i, 1-y_i) (boundary constraints)
    Use scipy.optimize.linprog with method='highs' for robustness.
    Positions are optimized via local search with LP solving at each step.
    """
    n = 26
    centers = generate_initial_positions(n)
    radii = solve_radii_lp(centers)
    
    # Local search: move each circle to improve total sum
    centers, radii = optimize_positions_with_lp(centers)
    
    return centers, radii, np.sum(radii)


def generate_initial_positions(n):
    """Generate well-spaced initial positions using hexagonal pattern + perturbation."""
    centers = []
    
    # Hexagonal lattice parameters
    r_base = 0.095
    dx = 2.0 * r_base
    dy = r_base * np.sqrt(3)
    
    # Row pattern for 26 circles: 5-4-5-4-5-3
    rows = [(5, 0), (4, 1), (5, 0), (4, 1), (5, 0), (3, 1)]
    
    # Center in unit square
    total_height = 5 * dy + 2 * r_base
    y_start = (1.0 - total_height) / 2 + r_base
    
    idx = 0
    for row_idx, (count, x_mult) in enumerate(rows):
        y = y_start + row_idx * dy
        row_width = 2 * r_base * (count - 1) if count > 1 else 0
        x_start = (1.0 - row_width) / 2 + x_mult * r_base
        for col in range(count):
            x = x_start + col * dx
            centers.append([x, y])
            idx += 1
    
    return np.array(centers)


def solve_radii_lp(centers):
    """
    Solve linear program for optimal radii given fixed positions.
    
    Variables: r_0, r_1, ..., r_{n-1}
    Objective: maximize sum(r_i) = minimize -sum(r_i)
    Constraints:
      - r_i + r_j <= dist(i,j) for all pairs (non-overlap)
      - r_i <= boundary_dist_i for each circle (boundary)
      - r_i >= 0 (non-negativity)
    """
    n = len(centers)
    if n == 0:
        return np.array([])
    
    # Objective: minimize -sum(r_i)
    c = -np.ones(n)
    
    # Build inequality constraints: A_ub @ r <= b_ub
    constraints_A = []
    constraints_b = []
    
    # Pair constraints: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            dist = np.linalg.norm(centers[i] - centers[j])
            constraints_A.append(row)
            constraints_b.append(dist)
    
    # Boundary constraints: r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        boundary_dist = min(centers[i, 0], centers[i, 1], 
                          1 - centers[i, 0], 1 - centers[i, 1])
        constraints_A.append(row)
        constraints_b.append(boundary_dist)
    
    A_ub = np.array(constraints_A)
    b_ub = np.array(constraints_b)
    
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP using HiGHS solver
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        # Fallback: simple boundary-constrained radii
        return np.array([min(centers[i, 0], centers[i, 1], 
                            1 - centers[i, 0], 1 - centers[i, 1]) 
                        for i in range(n)])


def optimize_positions_with_lp(centers, max_iters=15):
    """Local coordinate descent using LP to evaluate each position change."""
    n = len(centers)
    centers = centers.copy()
    radii = solve_radii_lp(centers)
    best_sum = np.sum(radii)
    
    for iteration in range(max_iters):
        improved = False
        
        for i in range(n):
            # Try moving circle i in various directions
            step = 0.015
            for _ in range(8):
                found_better = False
                for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                              (step, step), (-step, -step), (step, -step), (-step, step)]:
                    new_x = centers[i, 0] + dx
                    new_y = centers[i, 1] + dy
                    
                    # Check bounds
                    if not (0.02 < new_x < 0.98 and 0.02 < new_y < 0.98):
                        continue
                    
                    # Try new position
                    old_pos = centers[i].copy()
                    centers[i] = [new_x, new_y]
                    new_radii = solve_radii_lp(centers)
                    new_sum = np.sum(new_radii)
                    
                    if new_sum > best_sum + 1e-8:
                        radii = new_radii
                        best_sum = new_sum
                        improved = True
                        found_better = True
                    else:
                        centers[i] = old_pos
                
                step *= 0.7
                if not found_better:
                    break
        
        if not improved:
            break
    
    return centers, radii


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
