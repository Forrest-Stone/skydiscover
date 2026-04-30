# EVOLVE-BLOCK-START
"""SLSQP-based optimization for circle packing n=26 circles"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Use scipy.optimize.minimize with SLSQP to jointly optimize circle positions
    and radii. This is a fundamentally different approach from heuristic placement.
    Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    Objective: maximize sum of radii (minimize negative sum)
    Constraints: boundary (circles in square) and no-overlap constraints.
    """
    n = 26
    
    # Generate good initial guess using hexagonal-inspired pattern
    x0 = generate_initial_guess(n)
    
    # Define objective: minimize negative sum of radii
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)
    
    def objective_grad(vars):
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0  # derivative of -sum(r) w.r.t. each r
        return grad
    
    # Build constraints
    constraints = []
    
    # Boundary constraints: r <= x, r <= y, r <= 1-x, r <= 1-y
    # Rewritten as: x - r >= 0, y - r >= 0, 1 - x - r >= 0, 1 - y - r >= 0
    for i in range(n):
        idx = i * 3
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=idx: v[i] - v[i+2]})
        # y - r >= 0  
        constraints.append({'type': 'ineq', 'fun': lambda v, i=idx: v[i+1] - v[i+2]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=idx: 1 - v[i] - v[i+2]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=idx: 1 - v[i+1] - v[i+2]})
    
    # No-overlap constraints: ||ci - cj|| >= ri + rj
    for i in range(n):
        for j in range(i + 1, n):
            idx_i, idx_j = i * 3, j * 3
            def no_overlap(v, ii=idx_i, jj=idx_j):
                dx = v[ii] - v[jj]
                dy = v[ii+1] - v[jj+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - v[ii+2] - v[jj+2]
            constraints.append({'type': 'ineq', 'fun': no_overlap})
    
    # Variable bounds: positions in [0,1], radii positive
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    
    # Run SLSQP optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
    )
    
    # Extract solution
    vars_opt = result.x
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        idx = i * 3
        centers[i] = [vars_opt[idx], vars_opt[idx + 1]]
        radii[i] = max(vars_opt[idx + 2], 1e-8)
    
    # Ensure validity with final cleanup
    radii = enforce_validity(centers, radii)
    
    return centers, radii, np.sum(radii)


def generate_initial_guess(n):
    """Generate initial guess using layered hexagonal pattern."""
    vars0 = []
    # 4 corners
    for cx, cy in [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]:
        vars0.extend([cx, cy, 0.10])
    # 8 edge
    for x in [0.35, 0.50, 0.65]:
        vars0.extend([x, 0.07, 0.06])
    for x in [0.35, 0.50, 0.65]:
        vars0.extend([x, 0.93, 0.06])
    vars0.extend([0.07, 0.50, 0.06])
    vars0.extend([0.93, 0.50, 0.06])
    # 14 interior
    for x in [0.22, 0.42, 0.58, 0.78]:
        vars0.extend([x, 0.25, 0.07])
    for x in [0.32, 0.50, 0.68]:
        vars0.extend([x, 0.42, 0.07])
    for x in [0.22, 0.42, 0.58, 0.78]:
        vars0.extend([x, 0.58, 0.07])
    for x in [0.32, 0.50, 0.68]:
        vars0.extend([x, 0.75, 0.07])
    
    return np.array(vars0)


def enforce_validity(centers, radii):
    """Ensure all constraints are satisfied by shrinking radii if needed."""
    n = len(radii)
    radii = radii.copy()
    
    # Check boundary constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1 - x, 1 - y)
    
    # Check overlap constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist - 1e-9:
                    total = radii[i] + radii[j]
                    if total > 1e-9:
                        scale = (dist - 1e-9) / total
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
        if not changed:
            break
    
    return np.maximum(radii, 1e-8)


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
