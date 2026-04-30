# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using SLSQP with diverse patterns."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Multi-pattern SLSQP optimization with aggressive position perturbation.
    Explores corner-heavy, edge-optimized, and dense interior patterns.
    Uses fine parameter grid and multi-stage refinement.
    """
    n = 26
    best_sum, best_c, best_r = 0, None, None
    np.random.seed(42)
    
    # Pattern 1: Corner-heavy with fine parameter grid
    for co in [0.06, 0.08, 0.10, 0.12]:
        for sp in [0.135, 0.145, 0.155, 0.165]:
            centers = create_corner_pattern(n, offset=co, spacing=sp, row_cfg=[4,5,4,5,4])
            c, r, s_val = optimize_and_expand(centers)
            if s_val > best_sum:
                best_sum, best_c, best_r = s_val, c.copy(), r.copy()
    
    # Pattern 2: Dense interior with different row config
    for sp in [0.13, 0.14, 0.15, 0.16]:
        centers = create_corner_pattern(n, offset=0.05, spacing=sp, row_cfg=[5,4,5,4,4])
        c, r, s_val = optimize_and_expand(centers)
        if s_val > best_sum:
            best_sum, best_c, best_r = s_val, c.copy(), r.copy()
    
    # Pattern 3: Edge-optimized (corners further from edges)
    for sp in [0.14, 0.15, 0.16]:
        centers = create_corner_pattern(n, offset=0.15, spacing=sp, row_cfg=[4,5,4,5,4])
        c, r, s_val = optimize_and_expand(centers)
        if s_val > best_sum:
            best_sum, best_c, best_r = s_val, c.copy(), r.copy()
    
    # Pattern 4: Symmetric hexagonal (no explicit corners)
    centers = create_hexagonal_pattern(n, spacing=0.155)
    c, r, s_val = optimize_and_expand(centers)
    if s_val > best_sum:
        best_sum, best_c, best_r = s_val, c.copy(), r.copy()
    
    # Position perturbation refinement with varying magnitudes
    for mag in [0.01, 0.015, 0.02]:
        for _ in range(8):
            perturbed_c = best_c + np.random.uniform(-mag, mag, (n, 2))
            perturbed_c = np.clip(perturbed_c, 0.02, 0.98)
            c, r, s_val = optimize_and_expand(perturbed_c)
            if s_val > best_sum:
                best_sum, best_c, best_r = s_val, c.copy(), r.copy()
    
    return best_c, best_r, best_sum


def create_corner_pattern(n, offset, spacing, row_cfg):
    """Create pattern with 4 corner circles + configurable hexagonal interior."""
    centers = np.zeros((n, 2))
    centers[0] = [offset, offset]
    centers[1] = [1-offset, offset]
    centers[2] = [offset, 1-offset]
    centers[3] = [1-offset, 1-offset]
    
    idx = 4
    dy = spacing * np.sqrt(3) / 2
    base_y = offset + spacing * 0.65
    
    for row_idx, count in enumerate(row_cfg):
        y_base = base_y + row_idx * dy
        x_offset = (1 - (count - 1) * spacing) / 2
        for i in range(count):
            if idx < n:
                centers[idx] = [x_offset + i * spacing, y_base]
                idx += 1
    return centers


def create_hexagonal_pattern(n, spacing):
    """Create pure hexagonal pattern without explicit corner circles."""
    centers = np.zeros((n, 2))
    dy = spacing * np.sqrt(3) / 2
    idx = 0
    rows = [(0.1 + i*dy, 5 if i%2==0 else 6) for i in range(5)]
    
    for y_base, count in rows:
        x_offset = (1 - (count - 1) * spacing) / 2
        for i in range(count):
            if idx < n:
                centers[idx] = [x_offset + i * spacing, y_base]
                idx += 1
    return centers


def compute_initial_radii(centers):
    """Compute initial feasible radii with iterative expansion."""
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


def optimize_and_expand(centers):
    """Optimize with SLSQP then aggressively expand radii."""
    radii = compute_initial_radii(centers)
    c, r, _ = optimize_slsqp(centers, radii)
    r = expand_radii(c, r)
    return c, r, np.sum(r)


def expand_radii(centers, radii):
    """Aggressive radius expansion for maximum growth."""
    n = len(radii)
    for _ in range(600):
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


def optimize_slsqp(centers, radii):
    """SLSQP optimization for radius maximization."""
    n = len(centers)
    
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
    x0 = np.concatenate([centers.flatten(), radii])
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': constraints},
                      options={'maxiter': 500, 'ftol': 1e-12})
    
    c = result.x[:2*n].reshape(n, 2)
    r = result.x[2*n:]
    r = np.maximum(r, 0.01)
    for i in range(n):
        r[i] = min(r[i], c[i,0], c[i,1], 1 - c[i,0], 1 - c[i,1])
    return c, r, np.sum(r)


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
