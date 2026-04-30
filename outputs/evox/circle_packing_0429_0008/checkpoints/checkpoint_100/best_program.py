# EVOLVE-BLOCK-START
"""SLSQP optimization with growth phase for circle packing n=26"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    SLSQP optimization with multiple initialization patterns and growth phase.
    Uses gradient-based optimization followed by iterative radius expansion.
    """
    n = 26
    
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    def wall_constraints(params):
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        c = np.zeros(4 * n)
        for i in range(n):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            c[4*i] = x - r
            c[4*i+1] = 1 - x - r
            c[4*i+2] = y - r
            c[4*i+3] = 1 - y - r
        return c
    
    def circle_constraints(params):
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        c = []
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                c.append(dist - radii[i] - radii[j])
        return np.array(c)
    
    def create_init_pattern(rows, y_positions, init_r):
        """Create initialization from row pattern."""
        x0 = np.zeros(3 * n)
        idx = 0
        for row, y in zip(rows, y_positions):
            for k in range(row):
                x = (k + 0.5) / row
                x0[2*idx] = x
                x0[2*idx+1] = y
                x0[2*n + idx] = init_r
                idx += 1
        return x0
    
    # Multiple initialization patterns
    patterns = [
        ([4, 6, 6, 6, 4], [0.12, 0.30, 0.50, 0.70, 0.88]),
        ([5, 5, 6, 5, 5], [0.10, 0.30, 0.50, 0.70, 0.90]),
        ([4, 5, 8, 5, 4], [0.11, 0.30, 0.50, 0.70, 0.89]),
        ([5, 6, 4, 6, 5], [0.10, 0.30, 0.50, 0.70, 0.90]),
        ([3, 7, 6, 7, 3], [0.08, 0.28, 0.50, 0.72, 0.92]),
    ]
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    for rows, y_pos in patterns:
        x0 = create_init_pattern(rows, y_pos, 0.10)
        
        # Bounds
        bounds = [(0.01, 0.99)] * (2*n) + [(0.001, 0.25)] * n
        
        # Constraints for SLSQP
        cons = [
            {'type': 'ineq', 'fun': wall_constraints},
            {'type': 'ineq', 'fun': circle_constraints}
        ]
        
        result = minimize(
            objective, x0, method='SLSQP',
            constraints=cons, bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12}
        )
        
        params = result.x
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Growth phase: iteratively expand circles
        for _ in range(200):
            improved = False
            for i in range(n):
                # Find max possible radius for circle i
                max_r = min(centers[i, 0], centers[i, 1], 
                           1 - centers[i, 0], 1 - centers[i, 1])
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                                      (centers[i, 1] - centers[j, 1])**2)
                        max_r = min(max_r, dist - radii[j])
                if max_r > radii[i] + 1e-8:
                    radii[i] = max_r
                    improved = True
            if not improved:
                break
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


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
