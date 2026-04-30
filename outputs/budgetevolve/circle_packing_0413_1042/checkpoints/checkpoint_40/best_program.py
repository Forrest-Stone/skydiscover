# EVOLVE-BLOCK-START
"""Circle packing n=26: basinhopping global optimization with SLSQP local minimizer"""
import numpy as np
from scipy.optimize import basinhopping

def construct_packing():
    """
    Global optimization using basinhopping to escape local optima.
    Uses SLSQP as local minimizer with vectorized constraints.
    Custom take_step maintains feasibility during random walks.
    """
    n = 26
    n_pairs = n * (n - 1) // 2
    n_cons = 4 * n + n_pairs
    best_sum, best_x = 0, None
    
    def objective(v):
        return -np.sum(v[2::3])
    
    def obj_grad(v):
        g = np.zeros(3 * n)
        g[2::3] = -1.0
        return g
    
    def constraints(v):
        c = np.zeros(n_cons)
        for i in range(n):
            c[4*i] = v[3*i] - v[3*i+2]
            c[4*i+1] = 1 - v[3*i] - v[3*i+2]
            c[4*i+2] = v[3*i+1] - v[3*i+2]
            c[4*i+3] = 1 - v[3*i+1] - v[3*i+2]
        k = 4*n
        for i in range(n):
            for j in range(i+1, n):
                dx, dy = v[3*i] - v[3*j], v[3*i+1] - v[3*j+1]
                rs = v[3*i+2] + v[3*j+2]
                c[k] = dx*dx + dy*dy - rs*rs
                k += 1
        return c
    
    def cons_jac(v):
        jac = np.zeros((n_cons, 3*n))
        for i in range(n):
            jac[4*i, 3*i], jac[4*i, 3*i+2] = 1, -1
            jac[4*i+1, 3*i], jac[4*i+1, 3*i+2] = -1, -1
            jac[4*i+2, 3*i+1], jac[4*i+2, 3*i+2] = 1, -1
            jac[4*i+3, 3*i+1], jac[4*i+3, 3*i+2] = -1, -1
        k = 4*n
        for i in range(n):
            for j in range(i+1, n):
                dx, dy, rs = v[3*i]-v[3*j], v[3*i+1]-v[3*j+1], v[3*i+2]+v[3*j+2]
                jac[k, 3*i], jac[k, 3*i+1], jac[k, 3*i+2] = 2*dx, 2*dy, -2*rs
                jac[k, 3*j], jac[k, 3*j+1], jac[k, 3*j+2] = -2*dx, -2*dy, -2*rs
                k += 1
        return jac
    
    class AdaptiveStep:
        def __init__(self, stepsize=0.04):
            self.stepsize = stepsize
        def __call__(self, x):
            x_new = x.copy()
            for i in range(n):
                r = max(x[3*i+2], 0.03)
                step = self.stepsize * (0.5 + r)
                x_new[3*i] = np.clip(x[3*i] + np.random.uniform(-step, step), 0.03, 0.97)
                x_new[3*i+1] = np.clip(x[3*i+1] + np.random.uniform(-step, step), 0.03, 0.97)
                x_new[3*i+2] = np.clip(x[3*i+2] + np.random.uniform(-step*0.3, step*0.3), 0.005, 0.25)
            return x_new
    
    patterns = [[5,4,5,4,5,3], [4,5,4,5,4,4], [3,5,4,5,4,5], [5,5,4,4,4,4], [4,4,4,5,5,4]]
    
    for seed in range(20):
        np.random.seed(seed)
        x0 = np.zeros(3 * n)
        pat = patterns[seed % len(patterns)]
        idx = 0
        y_step = 1.0 / (len(pat) + 1)
        for row, cnt in enumerate(pat):
            y = y_step * (row + 1)
            x_step = 1.0 / (cnt + 1)
            for i in range(cnt):
                x = x_step * (i + 1) + (x_step * 0.5 if row % 2 else 0)
                x0[3*idx] = np.clip(x + np.random.uniform(-0.02, 0.02), 0.05, 0.95)
                x0[3*idx+1] = np.clip(y + np.random.uniform(-0.02, 0.02), 0.05, 0.95)
                x0[3*idx+2] = 0.04 + np.random.uniform(0, 0.02)
                idx += 1
        
        minimizer_kwargs = {
            'method': 'SLSQP', 'jac': obj_grad,
            'constraints': {'type': 'ineq', 'fun': constraints, 'jac': cons_jac},
            'bounds': [(0.02, 0.98), (0.02, 0.98), (0.004, 0.24)] * n,
            'options': {'maxiter': 300, 'ftol': 1e-9}
        }
        
        try:
            res = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs,
                             niter=35, take_step=AdaptiveStep(0.035), stepsize=0.04)
            radii = res.x[2::3]
            s = np.sum(radii[radii > 0])
            if s > best_sum:
                best_sum, best_x = s, res.x.copy()
        except:
            continue
    
    if best_x is None:
        best_x = x0
    centers = best_x.reshape(n, 3)[:, :2]
    radii = np.maximum(best_x[2::3], 1e-8)
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