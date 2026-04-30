# EVOLVE-BLOCK-START
"""Corner-first placement with SLSQP optimization for n=26 circle packing."""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """
    Corner circles exploit fewer boundary constraints (can grow larger).
    SLSQP handles nonlinear overlap constraints explicitly.
    Multi-start with diverse patterns: corner_hex, hexagonal, grid.
    """
    n = 26
    best_sum, best_c = 0, None
    bounds = [(0.02, 0.98)] * (2 * n)
    
    for init_type in ['corner_hex', 'hexagonal', 'grid']:
        c = init_centers(n, init_type)
        res = minimize(
            lambda x: -np.sum(compute_radii_fast(x.reshape(n, 2))),
            c.flatten(), method='SLSQP', bounds=bounds,
            options={'maxiter': 250, 'ftol': 1e-10})
        c = res.x.reshape(n, 2)
        s = np.sum(compute_radii_fast(c))
        if s > best_sum:
            best_sum, best_c = s, c.copy()
    
    return best_c, compute_radii_fast(best_c), best_sum

def init_centers(n, init_type):
    """Generate initial centers using different strategies."""
    c = np.zeros((n, 2))
    if init_type == 'corner_hex':
        d = 0.07
        c[0], c[1], c[2], c[3] = [d, d], [1-d, d], [d, 1-d], [1-d, 1-d]
        r, row_h, idx = 0.092, 0.092 * np.sqrt(3), 4
        rows = [4, 5, 4, 5, 4]
        y_start = r + 0.04
        for ri, cnt in enumerate(rows):
            y = y_start + ri * row_h
            margin = r * 1.4 if cnt == 4 else r * 0.75
            sp = (1 - 2*margin) / max(cnt-1, 1)
            for ci in range(cnt):
                c[idx] = [margin + ci*sp, y]
                idx += 1
    elif init_type == 'hexagonal':
        r, row_h, idx = 0.094, 0.094 * np.sqrt(3), 0
        rows = [5, 6, 5, 5, 5]
        for ri, cnt in enumerate(rows):
            y = r + ri * row_h
            sp = (1 - 2*r) / (cnt-1) if cnt > 1 else 0
            x0 = r if cnt == 6 else r + sp/2
            for ci in range(cnt):
                c[idx] = [x0 + ci*sp, y]
                idx += 1
    else:
        for i in range(n):
            c[i] = [0.08 + (i % 6) * 0.15, 0.08 + (i // 6) * 0.17]
    return c

def compute_radii_fast(c):
    n = len(c)
    r = np.minimum.reduce([c[:,0], c[:,1], 1-c[:,0], 1-c[:,1]])
    if n > 1:
        d = np.sqrt(np.sum((c[:,None,:] - c[None,:,:])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        r = np.minimum(r, d.min(axis=1)/2)
    return r


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
