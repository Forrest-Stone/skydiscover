# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles using scipy.optimize.minimize"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


def construct_packing():
    """
    Construct optimized packing using trust-constr optimization.
    Optimizes all 78 variables (x, y, r for 26 circles) simultaneously.
    Uses multiple random restarts with adaptive perturbations to escape local optima.
    
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    best_sum = 0
    best_result = None
    
    # Get initial guess from heuristic arrangement
    initial_centers, initial_radii, _ = get_initial_guess()
    
    # Try multiple random perturbations to escape local optima
    np.random.seed(42)
    for trial in range(15):  # Increased from 5 to 15 trials
        # Pack initial guess into variables: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = initial_centers[i, 0]
            x0[3*i + 1] = initial_centers[i, 1]
            x0[3*i + 2] = initial_radii[i]
        
        # Use best result as starting point for refinement trials
        if best_result is not None and trial > 5:
            x0 = best_result.copy()
        
        # Add random perturbation for trials after the first with adaptive scaling
        if trial > 0:
            scale = 0.03 if trial < 10 else 0.01  # Larger perturbations early, smaller later
            perturbation = np.random.randn(3 * n) * scale
            perturbation[2::3] = np.abs(perturbation[2::3]) * 0.3  # Smaller radius perturbations
            x0 = x0 + perturbation
            # Clip to valid ranges
            x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
            x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
            x0[2::3] = np.clip(x0[2::3], 0.01, 0.5)
        
        # Define objective: minimize negative sum of radii
        def objective(x):
            return -np.sum(x[2::3])
        
        # Non-overlapping constraints: ri + rj - dist <= 0
        def constraint_nonoverlap(x):
            violations = []
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, ri = x[3*i], x[3*i + 1], x[3*i + 2]
                    xj, yj, rj = x[3*j], x[3*j + 1], x[3*j + 2]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    violations.append(ri + rj - dist)
            return np.array(violations)
        
        # Boundary constraints: r - x <= 0, x + r - 1 <= 0, etc.
        def constraint_boundary(x):
            violations = []
            for i in range(n):
                xi, yi, ri = x[3*i], x[3*i + 1], x[3*i + 2]
                violations.extend([ri - xi, ri - yi, xi + ri - 1, yi + ri - 1])
            return np.array(violations)
        
        # Bounds: x,y in [0,1], r >= 0
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0, None)])
        
        # Create nonlinear constraints
        nc_nonoverlap = NonlinearConstraint(constraint_nonoverlap, -np.inf, 0)
        nc_boundary = NonlinearConstraint(constraint_boundary, -np.inf, 0)
        
        try:
            result = minimize(
                objective,
                x0,
                method='trust-constr',
                constraints=[nc_nonoverlap, nc_boundary],
                bounds=bounds,
                options={'maxiter': 1000, 'verbose': 0}  # Increased from 500 to 1000
            )
            
            current_sum = -result.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result.x.copy()
        except Exception:
            continue
    
    # If optimization failed, fall back to initial guess
    if best_result is None:
        centers = initial_centers
        radii = initial_radii
    else:
        centers = np.zeros((n, 2))
        centers[:, 0] = best_result[0::3]
        centers[:, 1] = best_result[1::3]
        radii = best_result[2::3]
    
    return centers, radii, np.sum(radii)


def get_initial_guess():
    """Generate initial guess using corner, edge, and interior placement."""
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corner circles
    r_corner = 0.146
    centers[0] = [r_corner, r_corner]
    centers[1] = [1 - r_corner, r_corner]
    centers[2] = [r_corner, 1 - r_corner]
    centers[3] = [1 - r_corner, 1 - r_corner]
    
    # 12 edge circles (3 per edge)
    edge_r = 0.128
    for i, x in enumerate([0.28, 0.50, 0.72]):
        centers[4 + i] = [x, edge_r]
        centers[7 + i] = [x, 1 - edge_r]
        centers[10 + i] = [edge_r, x]
        centers[13 + i] = [1 - edge_r, x]
    
    # 10 interior circles - hexagonal pattern
    h = 0.23
    v = h * np.sqrt(3) / 2
    cx, cy = 0.50, 0.50
    
    centers[16] = [cx - h, cy + v]
    centers[17] = [cx, cy + v]
    centers[18] = [cx + h, cy + v]
    centers[19] = [cx - h/2, cy]
    centers[20] = [cx + h/2, cy]
    centers[21] = [cx - h, cy - v]
    centers[22] = [cx, cy - v]
    centers[23] = [cx + h, cy - v]
    centers[24] = [cx - h/2, cy - 2*v]
    centers[25] = [cx + h/2, cy - 2*v]
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum radii ensuring no overlaps and staying within bounds."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    for _ in range(30):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist > 0:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
    
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
