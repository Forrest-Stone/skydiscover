# EVOLVE-BLOCK-START
"""SLSQP-optimized circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    SLSQP optimization with multiple hexagonal-based initial guesses.
    Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25].
    Uses greedy radius expansion post-optimization.
    """
    n = 26
    best_sum, best_centers, best_radii = 0, None, None
    constraints, bounds = _build_constraints(n)
    
    for x0 in _generate_initial_guesses(n):
        result = minimize(
            lambda v: -np.sum(v[2::3]), x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False}
        )
        centers = np.array([[result.x[i*3], result.x[i*3+1]] for i in range(n)])
        radii = np.array([max(result.x[i*3+2], 1e-8) for i in range(n)])
        radii = _enforce_validity(centers, radii)
        radii = _expand_radii(centers, radii)
        if np.sum(radii) > best_sum:
            best_sum, best_centers, best_radii = np.sum(radii), centers.copy(), radii.copy()
    
    return best_centers, best_radii, best_sum


def _build_constraints(n):
    """Build boundary and no-overlap constraints for SLSQP."""
    cons = []
    for i in range(n):
        idx = i * 3
        cons.extend([
            {'type': 'ineq', 'fun': lambda v, i=idx: v[i] - v[i+2]},
            {'type': 'ineq', 'fun': lambda v, i=idx: v[i+1] - v[i+2]},
            {'type': 'ineq', 'fun': lambda v, i=idx: 1 - v[i] - v[i+2]},
            {'type': 'ineq', 'fun': lambda v, i=idx: 1 - v[i+1] - v[i+2]}
        ])
    for i in range(n):
        for j in range(i+1, n):
            ii, jj = i*3, j*3
            cons.append({'type': 'ineq', 'fun': lambda v, a=ii, b=jj: 
                np.sqrt((v[a]-v[b])**2 + (v[a+1]-v[b+1])**2) - v[a+2] - v[b+2]})
    bounds = [(0,1), (0,1), (1e-6, 0.5)] * n
    return cons, bounds


def _generate_initial_guesses(n):
    """Generate diverse hexagonal-based initial configurations."""
    guesses = []
    for cr, er, ir in [(0.12, 0.07, 0.07), (0.11, 0.065, 0.075), 
                       (0.10, 0.08, 0.08), (0.115, 0.06, 0.07), (0.09, 0.065, 0.085)]:
        v = []
        for cx, cy in [(cr, cr), (1-cr, cr), (cr, 1-cr), (1-cr, 1-cr)]:
            v.extend([cx, cy, cr])
        for x in [0.35, 0.50, 0.65]:
            v.extend([x, er, er])
        for x in [0.35, 0.50, 0.65]:
            v.extend([x, 1-er, er])
        v.extend([er, 0.50, er])
        v.extend([1-er, 0.50, er])
        for x in [0.22, 0.42, 0.58, 0.78]:
            v.extend([x, 0.25, ir])
        for x in [0.32, 0.50, 0.68]:
            v.extend([x, 0.42, ir])
        for x in [0.22, 0.42, 0.58, 0.78]:
            v.extend([x, 0.58, ir])
        for x in [0.32, 0.50, 0.68]:
            v.extend([x, 0.75, ir])
        guesses.append(np.array(v))
    return guesses


def _enforce_validity(centers, radii):
    """Shrink radii to satisfy all constraints."""
    n = len(radii)
    radii = radii.copy()
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d + 1e-12:
                    s = (d - 1e-9) / (radii[i] + radii[j])
                    radii[i] *= s
                    radii[j] *= s
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 1e-8)


def _expand_radii(centers, radii):
    """Greedily expand each circle to maximum feasible radius."""
    n = len(radii)
    radii = radii.copy()
    for _ in range(300):
        improved = False
        for i in np.argsort(radii):
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    max_r = min(max_r, np.linalg.norm(centers[i]-centers[j]) - radii[j])
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
