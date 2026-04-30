# EVOLVE-BLOCK-START
"""Basinhopping global optimization with SLSQP refinement for n=26 circle packing."""
import numpy as np
from scipy.optimize import minimize, basinhopping

def construct_packing():
    """Global optimization via basinhopping - proven effective for escaping local minima."""
    n = 26
    best_sum, best_c = 0, None
    bounds = [(0.03, 0.97)] * (2 * n)
    
    def obj(x):
        return -np.sum(compute_radii(x.reshape(n, 2)))
    
    minimizer_kwargs = {"method": "SLSQP", "bounds": bounds,
                        "options": {"maxiter": 500, "ftol": 1e-12}}
    
    for init_type in ['corner_hex', 'hexagonal', 'staggered']:
        c = init_centers(n, init_type)
        res = basinhopping(obj, c.flatten(), minimizer_kwargs=minimizer_kwargs,
                          niter=18, stepsize=0.05, seed=42)
        c = res.x.reshape(n, 2)
        res = minimize(obj, c.flatten(), method='SLSQP', bounds=bounds,
                      options={'maxiter': 800, 'ftol': 1e-14})
        c = res.x.reshape(n, 2)
        s = np.sum(compute_radii(c))
        if s > best_sum:
            best_sum, best_c = s, c.copy()
    
    return best_c, compute_radii(best_c), best_sum

def init_centers(n, init_type):
    """Generate initial centers using proven patterns."""
    c = np.zeros((n, 2))
    if init_type == 'corner_hex':
        d = 0.07
        c[0], c[1], c[2], c[3] = [d, d], [1-d, d], [d, 1-d], [1-d, 1-d]
        r, row_h, idx = 0.094, 0.094*np.sqrt(3), 4
        rows = [4, 5, 4, 5, 4]
        y_start = r + 0.04
        for ri, cnt in enumerate(rows):
            y = y_start + ri * row_h
            margin = r * 1.2 if cnt == 4 else r * 0.6
            sp = (1 - 2*margin) / max(cnt-1, 1)
            for ci in range(cnt):
                c[idx] = [margin + ci*sp, y]
                idx += 1
    elif init_type == 'hexagonal':
        r, row_h, idx = 0.094, 0.094*np.sqrt(3), 0
        rows = [5, 6, 5, 5, 5]
        for ri, cnt in enumerate(rows):
            y = r + ri * row_h
            sp = (1 - 2*r) / (cnt-1) if cnt > 1 else 0
            x0 = r if cnt == 6 else r + sp/2
            for ci in range(cnt):
                c[idx] = [x0 + ci*sp, y]
                idx += 1
    else:
        r, row_h, idx = 0.094, 0.094*np.sqrt(3), 0
        rows = [5, 6, 5, 6, 4]
        for ri, cnt in enumerate(rows):
            y = r + ri * row_h
            sp = (1 - 2*r) / max(cnt-1, 1)
            x0 = r if cnt == 6 else r + sp/2
            for ci in range(cnt):
                c[idx] = [x0 + ci*sp, y]
                idx += 1
    return c

def compute_radii(c):
    """Vectorized radius computation from boundary and inter-circle constraints."""
    n = len(c)
    r = np.minimum.reduce([c[:,0], c[:,1], 1-c[:,0], 1-c[:,1]])
    if n > 1:
        d = np.sqrt(np.sum((c[:,None,:]-c[None,:,:])**2, axis=2))
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
