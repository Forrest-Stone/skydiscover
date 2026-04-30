# EVOLVE-BLOCK-START
"""Joint optimization using scipy trust-constr with LP refinement"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, linprog


def construct_packing():
    """
    Joint optimization of positions and radii using trust-constr method.
    Variables: 52 position vars + 26 radius vars = 78 total.
    Nonlinear constraints: dist(i,j) >= ri + rj for all pairs.
    Linear constraints: ri <= min(xi, yi, 1-xi, 1-yi) for boundaries.
    LP refinement ensures optimal radii for final positions.
    """
    n = 26
    best_positions, best_radii, best_sum = None, None, 0
    
    for trial in range(3):
        x0 = generate_initial_config(n, trial)
        positions, _ = optimize_trust_constr(x0, n)
        radii = solve_radii_lp(positions)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_positions, best_radii, best_sum = positions.copy(), radii.copy(), current_sum
    
    return best_positions, best_radii, best_sum


def generate_initial_config(n, trial):
    """Generate initial configuration from hexagonal pattern."""
    positions = hexagonal_positions(n)
    if trial > 0:
        positions += np.random.randn(n, 2) * 0.02 * trial
        positions = np.clip(positions, 0.12, 0.88)
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


def optimize_trust_constr(x0, n):
    """Trust-constr optimization with nonlinear constraints."""
    def objective(x):
        return -np.sum(x[52:])
    
    def objective_grad(x):
        grad = np.zeros(78)
        grad[52:] = -1.0
        return grad
    
    def constraints_fun(x):
        pos = x[:52].reshape(n, 2)
        rad = x[52:]
        return np.array([np.linalg.norm(pos[i] - pos[j]) - rad[i] - rad[j]
                        for i in range(n) for j in range(i+1, n)])
    
    def constraints_jac(x):
        pos = x[:52].reshape(n, 2)
        nc = n * (n - 1) // 2
        jac = np.zeros((nc, 78))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d > 1e-10:
                    u = diff / d
                    jac[idx, 2*i:2*i+2] = u
                    jac[idx, 2*j:2*j+2] = -u
                jac[idx, 52+i] = jac[idx, 52+j] = -1.0
                idx += 1
        return jac
    
    nlc = NonlinearConstraint(constraints_fun, lb=0.0, ub=np.inf, jac=constraints_jac)
    
    # Linear constraints: ri <= xi, ri <= yi, ri + xi <= 1, ri + yi <= 1
    A, b_u = np.zeros((4*n, 78)), np.zeros(4*n)
    for i in range(n):
        A[4*i, 52+i], A[4*i, 2*i] = 1.0, -1.0
        A[4*i+1, 52+i], A[4*i+1, 2*i+1] = 1.0, -1.0
        A[4*i+2, 52+i], A[4*i+2, 2*i], b_u[4*i+2] = 1.0, 1.0, 1.0
        A[4*i+3, 52+i], A[4*i+3, 2*i+1], b_u[4*i+3] = 1.0, 1.0, 1.0
    
    lc = LinearConstraint(A, lb=-np.inf, ub=b_u)
    bounds = [(0.01, 0.99)] * 52 + [(0.01, 0.49)] * 26
    
    result = minimize(objective, x0, method='trust-constr',
                     jac=objective_grad, constraints=[nlc, lc],
                     bounds=bounds, options={'maxiter': 100, 'xtol': 1e-6, 'gtol': 1e-6})
    
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
