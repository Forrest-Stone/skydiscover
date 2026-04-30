# EVOLVE-BLOCK-START
"""Asymmetric packing with corner-heavy distribution and SLSQP optimization."""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Direct construction: 4 corners + 12 edges + 10 interior, optimized with SLSQP.
    Corner circles exploit geometric advantage of touching only 2 edges."""
    n = 26
    best_sum, best_res = 0, None
    
    for trial in range(8):
        c = get_asymmetric_positions(n, trial)
        res = slsqp_optimize(c, n)
        s = np.sum(res.x[2::3])
        if s > best_sum:
            best_sum, best_res = s, res
    
    return np.hstack([best_res.x[0::3].reshape(n,1), best_res.x[1::3].reshape(n,1)]), best_res.x[2::3], best_sum

def get_asymmetric_positions(n, trial):
    """Corner-heavy initial placement: 4 corners + 12 edges + 10 interior."""
    np.random.seed(trial * 41)
    c = np.zeros((n, 2))
    i = 0
    # 4 corner circles - placed to maximize geometric advantage
    rc = 0.115 + np.random.uniform(-0.01, 0.01)
    re = 0.085 + np.random.uniform(-0.01, 0.01)
    g = max(0.1, (1 - 2*rc) / 4)
    for p in [(rc,rc), (1-rc,rc), (rc,1-rc), (1-rc,1-rc)]:
        c[i] = p; i += 1
    # 12 edge circles (3 per side)
    for k in range(3):
        c[i] = [rc+(k+1)*g, re]; i += 1
        c[i] = [rc+(k+1)*g, 1-re]; i += 1
        c[i] = [re, rc+(k+1)*g]; i += 1
        c[i] = [1-re, rc+(k+1)*g]; i += 1
    # 10 interior circles in hexagonal pattern
    for k in range(4): c[i] = [0.18+k*0.21, 0.32]; i += 1
    for k in range(3): c[i] = [0.28+k*0.22, 0.50]; i += 1
    for k in range(3): c[i] = [0.28+k*0.22, 0.68]; i += 1
    if trial > 0:
        c += np.random.uniform(-0.02, 0.02, c.shape)
        c = np.clip(c, 0.05, 0.95)
    return c

def slsqp_optimize(c, n):
    """SLSQP optimization for positions and radii simultaneously."""
    def obj(x): return -np.sum(x[2::3])
    cons = []
    for i in range(n):
        cons.extend([
            {'type':'ineq','fun':lambda x,i=i:x[3*i]-x[3*i+2]},
            {'type':'ineq','fun':lambda x,i=i:1-x[3*i]-x[3*i+2]},
            {'type':'ineq','fun':lambda x,i=i:x[3*i+1]-x[3*i+2]},
            {'type':'ineq','fun':lambda x,i=i:1-x[3*i+1]-x[3*i+2]}
        ])
    for i in range(n):
        for j in range(i+1,n):
            cons.append({'type':'ineq','fun':lambda x,i=i,j=j:
                np.sqrt((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2)-x[3*i+2]-x[3*j+2]})
    x0 = np.zeros(3*n)
    for i in range(n): x0[3*i], x0[3*i+1], x0[3*i+2] = c[i,0], c[i,1], 0.08
    bounds = [(0.01,0.99),(0.01,0.99),(0.01,0.5)]*n
    return minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':2000})
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