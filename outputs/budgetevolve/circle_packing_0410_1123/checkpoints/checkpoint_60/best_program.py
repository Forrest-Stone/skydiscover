# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using scipy.optimize.basinhopping"""
import numpy as np
from scipy.optimize import basinhopping, minimize


def construct_packing():
    """
    Optimize packing using basinhopping for global optimization.
    Basinhopping combines local minimization with Monte Carlo jumps
    to escape local optima and explore the solution space globally.
    """
    n = 26
    
    # Get initial guess
    centers0, radii0 = get_initial_guess(0)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i:3*i+3] = [centers0[i,0], centers0[i,1], radii0[i]]
    
    # Objective: minimize negative sum of radii
    def obj(x):
        return -np.sum(x[2::3])
    
    # Constraints for SLSQP (must be >= 0)
    def nonoverlap(x):
        c = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2)
                c.append(d - x[3*i+2] - x[3*j+2])
        return np.array(c)
    
    def boundary(x):
        c = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            c.extend([xi - ri, yi - ri, 1 - xi - ri, 1 - yi - ri])
        return np.array(c)
    
    constraints = [{'type': 'ineq', 'fun': nonoverlap},
                   {'type': 'ineq', 'fun': boundary}]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Custom take_step: perturb while respecting bounds
    class RandomDisplacement:
        def __init__(self, stepsize=0.02):
            self.stepsize = stepsize
        def __call__(self, x):
            x = x.copy()
            x[0::3] = np.clip(x[0::3] + np.random.randn(n) * self.stepsize, 0.01, 0.99)
            x[1::3] = np.clip(x[1::3] + np.random.randn(n) * self.stepsize, 0.01, 0.99)
            x[2::3] = np.clip(x[2::3] + np.random.randn(n) * self.stepsize * 0.5, 0.01, 0.5)
            return x
    
    # Local minimizer using SLSQP
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'constraints': constraints,
        'options': {'maxiter': 300, 'ftol': 1e-10}
    }
    
    # Run basinhopping
    result = basinhopping(
        obj, x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=200,
        T=0.5,
        take_step=RandomDisplacement(0.02)
    )
    
    centers = np.zeros((n, 2))
    centers[:, 0] = result.x[0::3]
    centers[:, 1] = result.x[1::3]
    radii = result.x[2::3]
    
    return centers, radii, np.sum(radii)


def get_initial_guess(seed):
    """Generate initial configuration with corner, edge, and interior circles."""
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corners
    rc = 0.145
    centers[:4] = [[rc, rc], [1-rc, rc], [rc, 1-rc], [1-rc, 1-rc]]
    
    # 12 edge circles (3 per edge)
    re = 0.12
    for i, v in enumerate([0.27, 0.5, 0.73]):
        centers[4+i] = [v, re]
        centers[7+i] = [v, 1-re]
        centers[10+i] = [re, v]
        centers[13+i] = [1-re, v]
    
    # 10 interior - hexagonal pattern
    h, v = 0.22, 0.22 * np.sqrt(3)/2
    cx, cy = 0.5, 0.5
    centers[16:] = [[cx-h,cy+v],[cx,cy+v],[cx+h,cy+v],[cx-h/2,cy],[cx+h/2,cy],
                    [cx-h,cy-v],[cx,cy-v],[cx+h,cy-v],[cx-h/2,cy-2*v],[cx+h/2,cy-2*v]]
    
    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """Compute maximum radii ensuring no overlaps."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    for _ in range(25):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d > 0:
                    s = d / (radii[i] + radii[j])
                    radii[i] *= s
                    radii[j] *= s
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