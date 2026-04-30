# EVOLVE-BLOCK-START
"""Basinhopping global optimization for n=26 circle packing with variable-radius initialization."""
import numpy as np
from scipy.optimize import basinhopping, minimize


def construct_packing():
    """
    Enhanced optimization using basinhopping + SLSQP + trust-constr polish + position refinement.
    Uses more trials and iterations for thorough exploration, plus position perturbation
    around best solution for final refinement.
    """
    n = 26
    best_sum, best_c, best_r = 0, None, None
    
    # Try multiple initial patterns with basinhopping - increased trials
    for trial in range(8):
        centers = create_variable_pattern(n, corner_size=0.07 + 0.015*trial)
        radii = compute_initial_radii(centers)
        
        # Phase 1: Basinhopping global optimization with more iterations
        centers, radii, s1 = basinhopping_optimize(centers, radii, niter=50)
        
        # Phase 2: SLSQP local polish
        centers, radii, s2 = slsqp_polish(centers, radii)
        
        # Phase 3: Trust-constr polish for tighter convergence
        centers, radii, s3 = trustconstr_polish(centers, radii)
        
        # Post-process: aggressive radius expansion
        radii = expand_radii(centers, radii)
        s_val = np.sum(radii)
        
        if s_val > best_sum:
            best_sum, best_c, best_r = s_val, centers.copy(), radii.copy()
    
    # Position perturbation refinement around best solution
    for _ in range(10):
        perturbed_c = best_c + np.random.randn(n, 2) * 0.008
        perturbed_c = np.clip(perturbed_c, 0.02, 0.98)
        perturbed_r = compute_initial_radii(perturbed_c)
        
        centers, radii, _ = slsqp_polish(perturbed_c, perturbed_r)
        centers, radii, _ = trustconstr_polish(centers, radii)
        radii = expand_radii(centers, radii)
        s_val = np.sum(radii)
        
        if s_val > best_sum:
            best_sum, best_c, best_r = s_val, centers.copy(), radii.copy()
    
    return best_c, best_r, best_sum


def create_variable_pattern(n, corner_size):
    """Create pattern with 4 large corner circles + hexagonal interior fill."""
    centers = np.zeros((n, 2))
    cr = corner_size
    
    # 4 corner circles
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    # Hexagonal interior pattern for remaining 22 circles
    idx = 4
    s = 0.16
    dy = s * np.sqrt(3) / 2
    base_y = cr + s * 0.5
    rows = [(base_y, 4), (base_y + dy, 5), (base_y + 2*dy, 4),
            (base_y + 3*dy, 5), (base_y + 4*dy, 4)]
    
    for y_base, count in rows:
        offset = (1 - (count - 1) * s) / 2
        for i in range(count):
            if idx < n:
                centers[idx] = [offset + i * s, y_base]
                idx += 1
    return centers


def compute_initial_radii(centers):
    """Compute initial radii with iterative expansion."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, y, 1 - x, 1 - y)
        for j in range(n):
            if i != j:
                max_r = min(max_r, np.linalg.norm(centers[i] - centers[j]) / 2)
        radii[i] = max_r
    
    for _ in range(50):
        improved = False
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    max_r = min(max_r, np.linalg.norm(centers[i] - centers[j]) - radii[j])
            if max_r > radii[i] + 1e-9:
                radii[i] = max_r
                improved = True
        if not improved:
            break
    return radii


def basinhopping_optimize(init_centers, init_radii, niter=50):
    """Basinhopping: stochastic global optimization with random perturbations."""
    n = len(init_centers)
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    def constraints(x):
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        cons = []
        for i in range(n):
            cons.extend([pos[i,0] - r[i], pos[i,1] - r[i],
                        1 - pos[i,0] - r[i], 1 - pos[i,1] - r[i]])
        for i in range(n):
            for j in range(i+1, n):
                cons.append(np.linalg.norm(pos[i] - pos[j]) - r[i] - r[j])
        return np.array(cons)
    
    def accept_test(f_new, x_new, f_old, x_old):
        """Accept only feasible solutions."""
        cons = constraints(x_new)
        return np.all(cons >= -1e-6)
    
    bounds = [(0.02, 0.98)] * (2*n) + [(0.01, 0.48)] * n
    x0 = np.concatenate([init_centers.flatten(), init_radii])
    
    # Local minimizer for each basinhopping step with more iterations
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {'maxiter': 100}
    }
    
    result = basinhopping(objective, x0, niter=niter, T=0.5, stepsize=0.05,
                          minimizer_kwargs=minimizer_kwargs, accept_test=accept_test,
                          seed=42)
    
    centers = result.x[:2*n].reshape(n, 2)
    radii = result.x[2*n:]
    radii = np.maximum(radii, 0.01)
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1], 1 - centers[i,0], 1 - centers[i,1])
    return centers, radii, np.sum(radii)


def slsqp_polish(init_centers, init_radii):
    """SLSQP local refinement for polishing the solution."""
    n = len(init_centers)
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    def constraints(x):
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        cons = []
        for i in range(n):
            cons.extend([pos[i,0] - r[i], pos[i,1] - r[i],
                        1 - pos[i,0] - r[i], 1 - pos[i,1] - r[i]])
        for i in range(n):
            for j in range(i+1, n):
                cons.append(np.linalg.norm(pos[i] - pos[j]) - r[i] - r[j])
        return np.array(cons)
    
    bounds = [(0.02, 0.98)] * (2*n) + [(0.01, 0.48)] * n
    x0 = np.concatenate([init_centers.flatten(), init_radii])
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': constraints},
                      options={'maxiter': 400, 'ftol': 1e-12})
    
    centers = result.x[:2*n].reshape(n, 2)
    radii = result.x[2*n:]
    radii = np.maximum(radii, 0.01)
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1], 1 - centers[i,0], 1 - centers[i,1])
    return centers, radii, np.sum(radii)


def trustconstr_polish(init_centers, init_radii):
    """Trust-constr polish for tighter convergence on best solution."""
    n = len(init_centers)
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    def constraints(x):
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        cons = []
        for i in range(n):
            cons.extend([pos[i,0] - r[i], pos[i,1] - r[i],
                        1 - pos[i,0] - r[i], 1 - pos[i,1] - r[i]])
        for i in range(n):
            for j in range(i+1, n):
                cons.append(np.linalg.norm(pos[i] - pos[j]) - r[i] - r[j])
        return np.array(cons)
    
    bounds = [(0.02, 0.98)] * (2*n) + [(0.01, 0.48)] * n
    x0 = np.concatenate([init_centers.flatten(), init_radii])
    
    result = minimize(objective, x0, method='trust-constr', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': constraints},
                      options={'maxiter': 200, 'gtol': 1e-10})
    
    centers = result.x[:2*n].reshape(n, 2)
    radii = result.x[2*n:]
    radii = np.maximum(radii, 0.01)
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1], 1 - centers[i,0], 1 - centers[i,1])
    return centers, radii, np.sum(radii)


def expand_radii(centers, radii):
    """Aggressive post-hoc radius expansion with more iterations."""
    n = len(radii)
    for _ in range(500):
        improved = False
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    max_r = min(max_r, np.linalg.norm(centers[i] - centers[j]) - radii[j])
            if max_r > radii[i] + 1e-12:
                radii[i] = max_r
                improved = True
        if not improved:
            break
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
