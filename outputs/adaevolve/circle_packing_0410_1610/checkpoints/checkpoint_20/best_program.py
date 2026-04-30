# EVOLVE-BLOCK-START
"""Optimized circle packing using scipy SLSQP for joint position-radius optimization"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Use scipy.optimize.minimize with SLSQP for joint optimization of positions and radii."""
    n = 26
    
    def get_initial_positions(seed=42):
        """Generate hexagonal initial positions with optional perturbation."""
        np.random.seed(seed)
        centers = np.zeros((n, 2))
        r = 0.095
        row_counts = [5, 4, 5, 4, 5, 3]
        y_spacing = r * np.sqrt(3)
        idx = 0
        for row, count in enumerate(row_counts):
            y = r + row * y_spacing
            if count == 5:
                x_positions = [r + i * 2 * r for i in range(5)]
            elif count == 4:
                x_positions = [2 * r + i * 2 * r for i in range(4)]
            else:
                x_positions = [r + i * 2 * r for i in range(3)]
            for x in x_positions:
                centers[idx] = [x, y]
                idx += 1
        # Add perturbation for restarts
        if seed != 42:
            centers += np.random.uniform(-0.02, 0.02, centers.shape)
            centers = np.clip(centers, 0.05, 0.95)
        return centers
    
    def pack_vars(centers, radii):
        """Pack centers and radii into flat array."""
        x = np.zeros(3 * n)
        for i in range(n):
            x[3*i], x[3*i+1], x[3*i+2] = centers[i, 0], centers[i, 1], radii[i]
        return x
    
    def unpack_vars(x):
        """Unpack flat array into centers and radii."""
        centers = x[0::3].reshape(n, 1)
        centers = np.hstack([centers, x[1::3].reshape(n, 1)])
        radii = x[2::3].copy()
        return centers, radii
    
    def objective(x):
        return -np.sum(x[2::3])  # Minimize negative sum of radii
    
    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        return grad
    
    # Build constraints
    constraints = []
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2] - 1e-8})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2] - 1e-8})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2] - 1e-8})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2] - 1e-8})
    
    # Non-overlap constraints (for all pairs)
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2) - x[3*i+2] - x[3*j+2] - 1e-8
            })
    
    # Bounds: positions in [0,1], radii in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    best_sum, best_solution = 0, None
    
    # Multiple restarts to escape local optima
    for restart in range(12):
        seed = 42 + restart * 17
        centers_init = get_initial_positions(seed)
        radii_init = compute_initial_radii(centers_init)
        x0 = pack_vars(centers_init, radii_init)
        
        result = minimize(
            objective, x0, method='SLSQP', jac=objective_grad,
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.success or result.fun < -best_sum:
            centers, radii = unpack_vars(result.x)
            radii = np.maximum(radii, 0.001)
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = (centers.copy(), radii.copy())
    
    centers, radii = best_solution
    return centers, radii, np.sum(radii)

def compute_initial_radii(centers):
    """Compute initial radii from boundary constraints."""
    n = len(centers)
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    radii = np.maximum(radii, 0.01)
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