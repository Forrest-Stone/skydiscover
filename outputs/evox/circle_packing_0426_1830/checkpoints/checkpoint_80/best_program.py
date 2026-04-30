# EVOLVE-BLOCK-START
"""Multi-start SLSQP optimization for n=26 circle packing."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Multi-start SLSQP optimization with diverse initial patterns.
    Tries corner-heavy and balanced configurations, then optimizes.
    """
    n = 26
    best_sum, best_c, best_r = 0, None, None
    
    # Try multiple initial patterns with different parameters
    for trial in range(6):
        centers = create_pattern(n, corner_offset=0.08 + 0.02*trial, 
                                  spacing=0.15 + 0.01*trial)
        radii = compute_max_radii(centers)
        c, r, s_val = optimize_packing(centers, radii)
        # Post-process: aggressive radius expansion
        r = expand_radii(c, r)
        s_val = np.sum(r)
        if s_val > best_sum:
            best_sum, best_c, best_r = s_val, c.copy(), r.copy()
    
    return best_c, best_r, best_sum


def create_pattern(n, corner_offset, spacing):
    """Create hexagonal pattern with 4 corner circles."""
    centers = np.zeros((n, 2))
    cr = corner_offset
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    idx = 4
    dy = spacing * np.sqrt(3) / 2
    base_y = cr + spacing * 0.7
    rows = [(base_y, 4), (base_y + dy, 5), (base_y + 2*dy, 4),
            (base_y + 3*dy, 5), (base_y + 4*dy, 4)]
    
    for y_base, count in rows:
        offset = (1 - (count - 1) * spacing) / 2
        for i in range(count):
            if idx < n:
                centers[idx] = [offset + i * spacing, y_base]
                idx += 1
    return centers


def compute_max_radii(centers):
    """Iteratively compute maximum feasible radii."""
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


def expand_radii(centers, radii):
    """Aggressive post-hoc radius expansion."""
    n = len(radii)
    for _ in range(200):
        improved = False
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    max_r = min(max_r, np.linalg.norm(centers[i] - centers[j]) - radii[j])
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        if not improved:
            break
    return radii


def optimize_packing(init_centers, init_radii):
    """SLSQP optimization maximizing sum of radii."""
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
                      options={'maxiter': 300, 'ftol': 1e-10})
    
    centers = result.x[:2*n].reshape(n, 2)
    radii = result.x[2*n:]
    radii = np.maximum(radii, 0.01)
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1],
                      1 - centers[i,0], 1 - centers[i,1])
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
