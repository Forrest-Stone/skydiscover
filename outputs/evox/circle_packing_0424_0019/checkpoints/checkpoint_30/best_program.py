# EVOLVE-BLOCK-START
"""Multi-pattern hexagonal packing with multi-start L-BFGS-B for n=26."""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Test multiple row patterns with multi-start optimization."""
    n = 26
    best_sum, best_c = 0, None
    patterns = [[5,6,5,5,5], [6,5,5,5,5], [4,5,5,5,4,3], [5,5,4,5,4,3]]
    bounds = [(0.02, 0.98)] * (2 * n)
    
    for rows in patterns:
        c = init_hex(n, rows)
        for _ in range(3):
            res = minimize(lambda x: -np.sum(compute_radii(x.reshape(n,2))),
                          c.flatten(), method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 120})
            c = res.x.reshape(n, 2)
            s = np.sum(compute_radii(c))
            if s > best_sum:
                best_sum, best_c = s, c.copy()
            c = best_c + np.random.randn(n, 2) * 0.012
            c = np.clip(c, 0.03, 0.97)
    return best_c, compute_radii(best_c), best_sum

def init_hex(n, rows):
    """Initialize centers in hexagonal pattern."""
    c, idx, r = np.zeros((n, 2)), 0, 0.094
    for ri, cnt in enumerate(rows):
        y = r + ri * r * np.sqrt(3)
        sp = (1 - 2*r) / (cnt - 1) if cnt > 1 else 0
        x0 = r if cnt == 6 else r + sp/2
        for ci in range(cnt):
            c[idx] = [x0 + ci*sp, y]
            idx += 1
    return c

def compute_radii(c):
    """Vectorized radius computation."""
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
