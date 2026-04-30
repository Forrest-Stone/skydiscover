# EVOLVE-BLOCK-START
"""Basin hopping global optimization for circle packing."""
import numpy as np
from scipy.optimize import minimize, basinhopping


def construct_packing():
    """
    Basin hopping optimization combining global stochastic search with local SLSQP refinement.
    Uses Metropolis acceptance criterion to escape local optima.
    """
    n = 26
    
    def objective(vars):
        """Minimize negative sum of radii."""
        radii = vars[52:]
        return -np.sum(radii)
    
    def constraints(vars):
        """Constraint values (must be >= 0 for valid solution)."""
        centers = vars[:52].reshape(n, 2)
        radii = vars[52:]
        cons = []
        
        # Boundary constraints
        for i in range(n):
            cons.extend([
                centers[i, 0] - radii[i],
                1 - centers[i, 0] - radii[i],
                centers[i, 1] - radii[i],
                1 - centers[i, 1] - radii[i]
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                r_sum = radii[i] + radii[j]
                cons.append(d2 - r_sum * r_sum)
        
        return np.array(cons)
    
    bounds = [(0.02, 0.98)] * 52 + [(0, 0.15)] * 26
    
    # Local minimizer config for basin hopping
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'constraints': {'type': 'ineq', 'fun': constraints},
        'options': {'maxiter': 800, 'ftol': 1e-9}
    }
    
    # Multi-start with basin hopping
    best_sum = 0
    best_solution = None
    patterns = [[5, 4, 5, 4, 5, 3], [4, 5, 4, 5, 4, 4], [5, 5, 6, 5, 5], [6, 5, 5, 5, 5]]
    
    for rows in patterns:
        for h in np.linspace(0.14, 0.22, 8):
            centers = create_hex_pattern(rows, h)
            radii = compute_max_radii(centers)
            x0 = np.concatenate([centers.flatten(), radii])
            
            # Basin hopping with adaptive step size
            result = basinhopping(
                objective, x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=30,
                stepsize=0.04,
                T=0.5
            )
            
            centers_res = result.x[:52].reshape(n, 2)
            radii_res = np.maximum(result.x[52:], 0)
            s = np.sum(radii_res)
            
            if s > best_sum:
                best_sum = s
                best_solution = result.x.copy()
    
    # Final local refinement
    result = minimize(
        objective, best_solution, method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraints},
        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-11}
    )
    
    centers = result.x[:52].reshape(n, 2)
    radii = np.maximum(result.x[52:], 0)
    
    return centers, radii, np.sum(radii)


def create_hex_pattern(rows, h):
    """Create centered hexagonal pattern with alternating row offsets."""
    n = sum(rows)
    centers = np.zeros((n, 2))
    v = h * np.sqrt(3) / 2
    
    total_height = (len(rows) - 1) * v
    y_start = (1 - total_height) / 2
    
    idx = 0
    y = y_start
    for row_idx, count in enumerate(rows):
        offset = h / 2 if row_idx % 2 == 1 else 0
        row_width = (count - 1) * h
        x_start = (1 - row_width) / 2
        for i in range(count):
            centers[idx] = [x_start + i * h + offset, y]
            idx += 1
        y += v
    
    return centers


def compute_max_radii(centers):
    """Iteratively compute maximum feasible radii."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])
    
    for _ in range(100):
        for i in range(n):
            max_r = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    max_r = min(max_r, dist - radii[j])
            radii[i] = max(0, max_r)
    
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
