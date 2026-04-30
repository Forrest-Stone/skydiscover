# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using trust-constr with simultaneous radius optimization"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.sparse import lil_matrix

def construct_packing():
    """
    Optimize all 78 variables (26 centers + 26 radii) simultaneously.
    Uses trust-constr with explicit linear wall constraints and nonlinear overlap constraints.
    """
    n = 26
    best_sum, best_result = 0, None
    
    def objective(vars):
        """Minimize negative sum of radii."""
        radii = vars[2::3]
        return -np.sum(radii)
    
    def objective_grad(vars):
        """Gradient of objective."""
        grad = np.zeros(3 * n)
        grad[2::3] = -1.0
        return grad
    
    def overlap_constraints(vars):
        """Nonlinear constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2."""
        radii = vars[2::3]
        centers = vars.reshape(n, 3)[:, :2]
        
        m = n * (n - 1) // 2
        vals = np.zeros(m)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                vals[idx] = dx*dx + dy*dy - (radii[i] + radii[j])**2
                idx += 1
        return vals
    
    def overlap_jacobian(vars):
        """Sparse Jacobian for overlap constraints."""
        radii = vars[2::3]
        centers = vars.reshape(n, 3)[:, :2]
        
        m = n * (n - 1) // 2
        jac = lil_matrix((m, 3 * n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                r_sum = radii[i] + radii[j]
                
                jac[idx, 3*i] = 2*dx
                jac[idx, 3*i+1] = 2*dy
                jac[idx, 3*i+2] = -2*r_sum
                jac[idx, 3*j] = -2*dx
                jac[idx, 3*j+1] = -2*dy
                jac[idx, 3*j+2] = -2*r_sum
                idx += 1
        return jac
    
    # Wall constraints as linear: x_i - r_i >= 0, -x_i - r_i >= -1, etc.
    A = lil_matrix((4*n, 3*n))
    b = np.zeros(4*n)
    for i in range(n):
        A[4*i, 3*i] = 1; A[4*i, 3*i+2] = -1; b[4*i] = 0
        A[4*i+1, 3*i] = -1; A[4*i+1, 3*i+2] = -1; b[4*i+1] = -1
        A[4*i+2, 3*i+1] = 1; A[4*i+2, 3*i+2] = -1; b[4*i+2] = 0
        A[4*i+3, 3*i+1] = -1; A[4*i+3, 3*i+2] = -1; b[4*i+3] = -1
    
    wall_constraint = LinearConstraint(A.tocsr(), lb=b, ub=np.inf)
    overlap_constraint = NonlinearConstraint(overlap_constraints, lb=0, ub=np.inf, jac=overlap_jacobian)
    
    for seed in range(12):
        np.random.seed(seed)
        x0 = np.zeros(3 * n)
        
        # Hexagonal grid initialization
        rows = [5, 4, 5, 4, 5, 3]
        y_step = 1.0 / (len(rows) + 1)
        idx = 0
        for row_idx, count in enumerate(rows):
            y = y_step * (row_idx + 1)
            x_step = 1.0 / (count + 1)
            for i in range(count):
                x = x_step * (i + 1)
                if row_idx % 2 == 1:
                    x += x_step * 0.5
                x0[3*idx] = np.clip(x + np.random.uniform(-0.02, 0.02), 0.03, 0.97)
                x0[3*idx+1] = np.clip(y + np.random.uniform(-0.02, 0.02), 0.03, 0.97)
                x0[3*idx+2] = 0.04 + np.random.uniform(0, 0.02)
                idx += 1
        
        bounds = [(0.02, 0.98), (0.02, 0.98), (0.005, 0.25)] * n
        
        try:
            result = minimize(objective, x0, method='trust-constr',
                            jac=objective_grad,
                            constraints=[wall_constraint, overlap_constraint],
                            bounds=bounds,
                            options={'maxiter': 800, 'gtol': 1e-7, 'verbose': 0})
            
            radii = result.x[2::3]
            sum_radii = np.sum(radii[radii > 0])
            
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result.x.copy()
        except:
            continue
    
    if best_result is None:
        best_result = x0
    
    centers = best_result.reshape(n, 3)[:, :2]
    radii = np.maximum(best_result[2::3], 1e-8)
    
    return centers, radii, np.sum(radii)
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