# EVOLVE-BLOCK-START
"""SLSQP optimization with iterative radius expansion for n=26 circle packing"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """SLSQP optimizes positions, then iterative expansion computes max radii."""
    n = 26
    best_sum, best_x = 0, None
    
    def objective(x):
        return -np.sum(x[2::3])
    
    cons = []
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                np.sqrt((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2)-x[3*i+2]-x[3*j+2]})
        cons.extend([{'type': 'ineq', 'fun': lambda x, i=i: x[3*i]-x[3*i+2]},
                     {'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1]-x[3*i+2]},
                     {'type': 'ineq', 'fun': lambda x, i=i: 1-x[3*i]-x[3*i+2]},
                     {'type': 'ineq', 'fun': lambda x, i=i: 1-x[3*i+1]-x[3*i+2]}])
    
    bounds = [(0.02, 0.98), (0.02, 0.98), (0.01, 0.20)] * n
    
    for r_base in [0.088, 0.092, 0.096, 0.10]:
        x0 = hex_start(n, r_base)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 1200, 'ftol': 1e-11})
            s = np.sum(res.x[2::3])
            if s > best_sum:
                best_sum, best_x = s, res.x.copy()
        except: pass
    
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        x0 = np.column_stack([np.random.uniform(0.05, 0.95, n),
            np.random.uniform(0.05, 0.95, n), np.random.uniform(0.07, 0.12, n)]).flatten()
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                          options={'maxiter': 1200, 'ftol': 1e-11})
            s = np.sum(res.x[2::3])
            if s > best_sum:
                best_sum, best_x = s, res.x.copy()
        except: pass
    
    if best_x is None:
        best_x = hex_start(n, 0.095)
    
    centers = np.array([[best_x[3*i], best_x[3*i+1]] for i in range(n)])
    radii = expand_radii(centers)
    return centers, radii, np.sum(radii)

def hex_start(n, r):
    c = np.zeros((n, 2))
    dx, dy = 2*r, r*np.sqrt(3)
    idx, row, y = 0, 0, r + 0.01
    while idx < n and y < 1-r:
        x = r + 0.01 + (dx/2 if row % 2 else 0)
        while idx < n and x < 1-r:
            c[idx] = [min(x, 1-r), min(y, 1-r)]; idx += 1; x += dx
        y += dy; row += 1
    np.random.seed(42)
    while idx < n:
        c[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]; idx += 1
    return np.column_stack([c, np.full(n, r*0.9)]).flatten()

def expand_radii(centers, max_iter=150):
    """Iteratively expand each circle to maximum feasible radius."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, max(0, d - radii[j]))
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        if not improved: break
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