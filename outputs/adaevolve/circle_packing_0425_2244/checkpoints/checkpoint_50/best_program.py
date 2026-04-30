# EVOLVE-BLOCK-START
"""Linear programming for optimal radii with multi-restart position optimization"""
import numpy as np
from scipy.optimize import linprog, minimize


def construct_packing():
    """
    Use linear programming to optimally solve for radii given positions.
    Multi-restart approach with different initial patterns and scipy refinement.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0
    
    # Try multiple initial configurations
    for trial in range(6):
        centers = generate_initial_positions(n, trial)
        centers, radii = optimize_positions_with_lp(centers)
        
        # Final polish with scipy.optimize
        centers, radii = scipy_refine(centers)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


def generate_initial_positions(n, pattern=0):
    """Generate initial positions using different hexagonal patterns."""
    centers = []
    r_base = 0.095
    dx = 2.0 * r_base
    dy = r_base * np.sqrt(3)
    
    # Different row patterns for 26 circles
    if pattern == 0:
        rows = [(5, 0), (4, 1), (5, 0), (4, 1), (5, 0), (3, 1)]
    elif pattern == 1:
        rows = [(6, 0), (4, 1), (6, 0), (4, 1), (6, 0)]
    elif pattern == 2:
        rows = [(5, 0), (5, 0.5), (5, 0), (5, 0.5), (5, 0), (1, 0)]
    elif pattern == 3:
        rows = [(4, 0), (5, 1), (5, 0), (4, 1), (5, 0), (3, 1)]
    elif pattern == 4:
        rows = [(5, 0), (4, 1), (4, 0), (5, 1), (4, 0), (4, 1)]
    else:
        rows = [(6, 0), (5, 0.5), (5, 0), (5, 0.5), (5, 0)]
    
    total_height = (len(rows) - 1) * dy + 2 * r_base
    y_start = (1.0 - total_height) / 2 + r_base
    
    for row_idx, (count, x_mult) in enumerate(rows):
        y = y_start + row_idx * dy
        row_width = 2 * r_base * (count - 1) if count > 1 else 0
        x_start = (1.0 - row_width) / 2 + x_mult * r_base
        for col in range(count):
            x = x_start + col * dx
            centers.append([x, y])
    
    result = np.array(centers[:n])
    if pattern > 0:
        result += np.random.randn(len(result), 2) * 0.008
        result = np.clip(result, 0.05, 0.95)
    return result


def solve_radii_lp(centers):
    """Solve LP for optimal radii given fixed positions."""
    n = len(centers)
    if n == 0:
        return np.array([])
    
    c = -np.ones(n)
    A_list = []
    b_list = []
    
    # Pair constraints
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i], row[j] = 1.0, 1.0
            A_list.append(row)
            b_list.append(np.linalg.norm(centers[i] - centers[j]))
    
    # Boundary constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        b_list.append(min(centers[i, 0], centers[i, 1], 1-centers[i, 0], 1-centers[i, 1]))
        A_list.append(row)
    
    result = linprog(c, A_ub=np.array(A_list), b_ub=np.array(b_list),
                    bounds=[(0, None)]*n, method='highs')
    return result.x if result.success else np.zeros(n)


def optimize_positions_with_lp(centers, max_iters=30):
    """Enhanced local search with more directions and larger steps."""
    n = len(centers)
    centers = centers.copy()
    radii = solve_radii_lp(centers)
    best_sum = np.sum(radii)
    
    # 16 directions including diagonals
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1),
            (0.7,0.7),(-0.7,-0.7),(0.7,-0.7),(-0.7,0.7),
            (0.5,1),(-0.5,-1),(0.5,-1),(-0.5,1)]
    
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            step = 0.03  # Larger initial step
            for _ in range(12):
                found_better = False
                for dx, dy in dirs:
                    new_x = centers[i, 0] + dx * step
                    new_y = centers[i, 1] + dy * step
                    if not (0.02 < new_x < 0.98 and 0.02 < new_y < 0.98):
                        continue
                    old_pos = centers[i].copy()
                    centers[i] = [new_x, new_y]
                    new_radii = solve_radii_lp(centers)
                    new_sum = np.sum(new_radii)
                    if new_sum > best_sum + 1e-9:
                        radii, best_sum = new_radii, new_sum
                        improved = found_better = True
                    else:
                        centers[i] = old_pos
                step *= 0.65
                if not found_better:
                    break
        if not improved:
            break
    return centers, radii


def scipy_refine(centers):
    """Final polish using scipy.optimize.minimize with Powell method."""
    n = len(centers)
    
    def objective(x):
        c = np.clip(x.reshape(n, 2), 0.02, 0.98)
        return -np.sum(solve_radii_lp(c))
    
    result = minimize(objective, centers.flatten(), method='Powell',
                     options={'maxiter': 150, 'xtol': 1e-5})
    centers = np.clip(result.x.reshape(n, 2), 0.02, 0.98)
    return centers, solve_radii_lp(centers)


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
