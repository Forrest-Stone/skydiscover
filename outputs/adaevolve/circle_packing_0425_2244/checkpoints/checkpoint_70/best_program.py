# EVOLVE-BLOCK-START
"""SLSQP optimization with LP refinement for circle packing"""
import numpy as np
from scipy.optimize import minimize, linprog


def construct_packing():
    """
    Joint optimization using SLSQP (Sequential Quadratic Programming).
    SLSQP solves QP subproblems at each iteration, potentially finding
    different local optima than trust-constr's barrier method.
    Variables: 52 position vars + 26 radius vars = 78 total.
    LP refinement ensures optimal radii for final positions.
    """
    n = 26
    best_positions, best_radii, best_sum = None, None, 0
    
    for trial in range(5):  # More trials with SLSQP's faster convergence
        x0 = generate_initial_config(n, trial)
        positions, _ = optimize_slsqp(x0, n)
        radii = solve_radii_lp(positions)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_positions, best_radii, best_sum = positions.copy(), radii.copy(), current_sum
    
    return best_positions, best_radii, best_sum


def generate_initial_config(n, trial):
    """Generate initial configuration from hexagonal pattern with perturbations."""
    positions = hexagonal_positions(n)
    if trial > 0:
        np.random.seed(trial * 42)
        positions += np.random.randn(n, 2) * 0.03 * trial
        positions = np.clip(positions, 0.08, 0.92)
    radii = np.ones(n) * 0.095
    return np.concatenate([positions.flatten(), radii])


def hexagonal_positions(n):
    """Generate hexagonal lattice positions for 26 circles."""
    r_base, centers = 0.095, []
    dx, dy = 2.0 * r_base, r_base * np.sqrt(3)
    rows = [(5, 0), (4, 1), (5, 0), (4, 1), (5, 0), (3, 1)]
    y_start = (1.0 - (len(rows)-1)*dy - 2*r_base) / 2 + r_base
    
    for row_idx, (count, x_mult) in enumerate(rows):
        y = y_start + row_idx * dy
        row_width = 2 * r_base * (count - 1) if count > 1 else 0
        x_start = (1.0 - row_width) / 2 + x_mult * r_base
        for col in range(count):
            centers.append([x_start + col * dx, y])
    
    return np.array(centers[:n])


def optimize_slsqp(x0, n):
    """SLSQP optimization with dictionary format constraints."""
    def objective(x):
        return -np.sum(x[52:])
    
    def objective_grad(x):
        grad = np.zeros(78)
        grad[52:] = -1.0
        return grad
    
    def constraints_fun(x):
        pos = x[:52].reshape(n, 2)
        rad = x[52:]
        # Non-overlap: dist(i,j) >= ri + rj  =>  dist - ri - rj >= 0
        non_overlap = [np.linalg.norm(pos[i] - pos[j]) - rad[i] - rad[j]
                       for i in range(n) for j in range(i+1, n)]
        # Boundary: xi - ri >= 0, yi - ri >= 0, 1 - xi - ri >= 0, 1 - yi - ri >= 0
        boundary = []
        for i in range(n):
            boundary.extend([pos[i, 0] - rad[i],      # xi - ri >= 0
                            pos[i, 1] - rad[i],        # yi - ri >= 0
                            1 - pos[i, 0] - rad[i],    # 1 - xi - ri >= 0
                            1 - pos[i, 1] - rad[i]])    # 1 - yi - ri >= 0
        return np.array(non_overlap + boundary)
    
    constraints = {'type': 'ineq', 'fun': constraints_fun}
    bounds = [(0.01, 0.99)] * 52 + [(0.01, 0.49)] * 26
    
    result = minimize(objective, x0, method='SLSQP',
                     jac=objective_grad, constraints=constraints,
                     bounds=bounds, options={'maxiter': 300, 'ftol': 1e-9})
    
    return result.x[:52].reshape(n, 2), result.x[52:]


def solve_radii_lp(centers):
    """Solve LP for optimal radii given fixed positions."""
    n = len(centers)
    c = -np.ones(n)
    A_list, b_list = [], []
    
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i], row[j] = 1.0, 1.0
            A_list.append(row)
            b_list.append(np.linalg.norm(centers[i] - centers[j]))
    
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_list.append(row)
        b_list.append(min(centers[i, 0], centers[i, 1], 1-centers[i, 0], 1-centers[i, 1]))
    
    result = linprog(c, A_ub=np.array(A_list), b_ub=np.array(b_list),
                    bounds=[(0, None)]*n, method='highs')
    return result.x if result.success else np.zeros(n)


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
