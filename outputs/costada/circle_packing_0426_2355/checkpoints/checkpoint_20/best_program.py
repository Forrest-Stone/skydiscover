# EVOLVE-BLOCK-START
"""Joint optimization of positions and radii using trust-constr method"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, linprog


def construct_packing():
    """
    Joint optimization using trust-constr with nonlinear constraints.
    78 variables: 52 position coords + 26 radii.
    Constraints: distance >= r_i + r_j (non-overlap), edge distance >= r_i (boundary).
    Multiple restarts to avoid local minima.
    """
    n = 26
    best_sum = 0
    best_pos, best_rad = None, None
    
    for seed in range(3):
        np.random.seed(seed)
        x0 = create_initial_guess(n, seed)
        constraints = build_constraints(n)
        bounds = [(0.02, 0.98)] * (2*n) + [(0.005, 0.48)] * n
        
        try:
            result = minimize(
                lambda x: -np.sum(x[2*n:]), x0, method='trust-constr',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 400, 'verbose': 0}
            )
            pos = np.clip(result.x[:2*n].reshape(n, 2), 0.02, 0.98)
            rad = compute_optimal_radii_lp(pos)
            if np.sum(rad) > best_sum:
                best_sum, best_pos, best_rad = np.sum(rad), pos.copy(), rad.copy()
        except:
            continue
    
    if best_pos is None:
        best_pos = create_initial_guess(n, 0)[:2*n].reshape(n, 2)
        best_rad = compute_optimal_radii_lp(best_pos)
    
    return best_pos, np.maximum(best_rad, 0), np.sum(best_rad)


def create_initial_guess(n, seed=0):
    """Create initial guess with hexagonal pattern + small perturbation."""
    pos = np.zeros((n, 2))
    rows = [5, 4, 5, 4, 5, 3]
    y_sp = 1.0 / (len(rows) + 1)
    
    idx = 0
    for r, cnt in enumerate(rows):
        y = (r + 1) * y_sp
        x_sp = 1.0 / (cnt + 1)
        for c in range(cnt):
            pos[idx] = [(c + 1) * x_sp, y]
            idx += 1
    
    if seed > 0:
        pos += np.random.randn(n, 2) * 0.02
        pos = np.clip(pos, 0.05, 0.95)
    
    r_init = np.array([min(p[0], p[1], 1-p[0], 1-p[1]) * 0.6 for p in pos])
    return np.concatenate([pos.flatten(), r_init])


def build_constraints(n):
    """Build nonlinear constraints for trust-constr."""
    n_pairs = n * (n - 1) // 2
    n_con = n_pairs + 4 * n
    
    def constraint_fn(x):
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        c = np.zeros(n_con)
        
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                c[idx] = d - r[i] - r[j]
                idx += 1
        
        for i in range(n):
            c[idx:idx+4] = [pos[i,0]-r[i], pos[i,1]-r[i], 1-pos[i,0]-r[i], 1-pos[i,1]-r[i]]
            idx += 4
        
        return c
    
    def constraint_jac(x):
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        J = np.zeros((n_con, 3*n))
        
        row = 0
        for i in range(n):
            for j in range(i+1, n):
                dx, dy = pos[i,0]-pos[j,0], pos[i,1]-pos[j,1]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                J[row, 2*i], J[row, 2*i+1] = dx/d, dy/d
                J[row, 2*j], J[row, 2*j+1] = -dx/d, -dy/d
                J[row, 2*n+i], J[row, 2*n+j] = -1, -1
                row += 1
        
        for i in range(n):
            J[row, 2*i], J[row, 2*n+i] = 1, -1
            J[row+1, 2*i+1], J[row+1, 2*n+i] = 1, -1
            J[row+2, 2*i], J[row+2, 2*n+i] = -1, -1
            J[row+3, 2*i+1], J[row+3, 2*n+i] = -1, -1
            row += 4
        
        return J
    
    return NonlinearConstraint(constraint_fn, lb=0, ub=np.inf, jac=constraint_jac)


def compute_optimal_radii_lp(centers):
    """Compute optimal radii using linear programming."""
    n = centers.shape[0]
    c = -np.ones(n)
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs + n, n))
    b_ub = np.zeros(n_pairs + n)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            A_ub[idx, i] = A_ub[idx, j] = 1
            b_ub[idx] = d
            idx += 1
    
    for i in range(n):
        x, y = centers[i]
        A_ub[n_pairs + i, i] = 1
        b_ub[n_pairs + i] = max(min(x, y, 1-x, 1-y), 1e-6)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs')
    return res.x if res.success else np.ones(n) * 0.05


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
