# EVOLVE-BLOCK-START
"""SLSQP joint optimization for n=26 circle packing in a unit square"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """
    Joint optimization of all 78 variables (26 centers × 2 + 26 radii)
    using SLSQP with explicit nonlinear constraints. Parameters are
    flattened as [x0,y0,r0,x1,y1,r1,...].
    """
    n = 26
    best_sum, best_params = 0, None
    
    # Multiple restarts with perturbed initial positions
    for trial in range(5):
        params0 = init_params(n, perturb=(trial > 0))
        
        result = minimize(
            lambda p: -np.sum(p[2::3]),  # Maximize sum of radii
            params0,
            method='SLSQP',
            constraints=build_constraints(n),
            bounds=build_bounds(n),
            options={'ftol': 1e-9, 'maxiter': 500, 'disp': False}
        )
        
        if result.success or result.fun < -best_sum:
            s = -result.fun
            if s > best_sum:
                best_sum, best_params = s, result.x
    
    # Extract centers and radii from flattened params
    centers = best_params.reshape(n, 3)[:, :2]
    radii = best_params[2::3]
    return centers, radii, np.sum(radii)


def build_constraints(n):
    """Build wall and non-overlap constraints for SLSQP."""
    cons = []
    
    # Wall constraints: x_i - r_i >= 0, y_i - r_i >= 0
    # and 1 - x_i - r_i >= 0, 1 - y_i - r_i >= 0
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
    
    # Non-overlap: ||c_i - c_j|| >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt((p[3*i]-p[3*j])**2 + 
                         (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
            })
    return cons


def build_bounds(n):
    """Build bounds: x,y in [0,1], r in [0, 0.5]."""
    return [(0, 1), (0, 1), (0, 0.5)] * n


def init_params(n, perturb=False):
    """Initialize from 5-6-5-6-4 hexagonal pattern."""
    r = 0.095
    h, v = 2 * r, np.sqrt(3) * r
    rows = [(5, 0), (6, h/2), (5, 0), (6, h/2), (4, h)]
    y_start = 0.5 - 2 * v
    
    params = np.zeros(3 * n)
    idx = 0
    for row_idx, (count, x_off) in enumerate(rows):
        y = y_start + row_idx * v
        x_start = 0.5 - (count - 1) * h / 2
        for col in range(count):
            x = x_start + x_off + col * h
            params[3*idx] = np.clip(x, 0.05, 0.95)
            params[3*idx+1] = np.clip(y, 0.05, 0.95)
            params[3*idx+2] = r
            idx += 1
    
    if perturb:
        params += np.random.uniform(-0.02, 0.02, 3*n)
        params[2::3] = np.clip(params[2::3], 0.01, 0.2)
    
    return params

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