# EVOLVE-BLOCK-START
"""Smooth residual formulation using log-sum-exp for differentiable optimization."""
import numpy as np
from scipy.optimize import least_squares, minimize

def construct_packing():
    """
    Use smooth residual formulation to avoid gradient discontinuities.
    Replace max(0, x) with log(1 + exp(k*x))/k for smooth approximation.
    This allows trust-region optimizer to compute better search directions.
    """
    n = 26
    centers_init = _hexagonal_initial(n)
    radii_init = np.array([0.5 * min(centers_init[i,0], centers_init[i,1], 
                                     1-centers_init[i,0], 1-centers_init[i,1]) for i in range(n)])
    
    best_sum = 0
    best_centers, best_radii = centers_init.copy(), radii_init.copy()
    
    np.random.seed(42)
    for restart in range(12):
        if restart == 0:
            x0 = np.concatenate([centers_init.flatten(), radii_init])
        else:
            pert = np.random.randn(3*n) * 0.04
            x0 = np.concatenate([centers_init.flatten(), radii_init]) + pert
            x0 = np.clip(x0, 0.01, 0.99)
        
        # Smooth residuals allow better gradient flow
        res = least_squares(_smooth_residuals, x0, args=(n,), method='trf',
                           bounds=(np.zeros(3*n), np.ones(3*n)),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=3000)
        
        x_opt = res.x
        centers = x_opt[:2*n].reshape(n, 2)
        radii = x_opt[2*n:]
        
        # Refine radii with fixed centers
        radii = _max_radii_for_centers(centers)
        
        # Local SLSQP refinement
        x0_ref = np.concatenate([centers.flatten(), radii])
        res2 = minimize(_objective, x0_ref, args=(n,), method='SLSQP',
                       constraints=_build_constraints(n),
                       bounds=[(0,1)]*(2*n) + [(0.001,0.5)]*n,
                       options={'maxiter':500, 'ftol':1e-10})
        
        if -res2.fun > best_sum:
            best_sum = -res2.fun
            best_centers = res2.x[:2*n].reshape(n, 2)
            best_radii = res2.x[2*n:]
    
    return best_centers, best_radii, np.sum(best_radii)

def _smooth_max(x, k=30.0):
    """Smooth approximation of max(0, x) using log-sum-exp."""
    return np.log(1.0 + np.exp(k * x)) / k

def _smooth_residuals(x, n):
    """Smooth residual formulation using log-sum-exp approximation."""
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    residuals = []
    
    # Smooth boundary residuals - avoids gradient discontinuity at constraint boundary
    for i in range(n):
        residuals.extend([
            _smooth_max(radii[i] - centers[i,0]),      # left
            _smooth_max(radii[i] - centers[i,1]),      # bottom
            _smooth_max(radii[i] - (1 - centers[i,0])), # right
            _smooth_max(radii[i] - (1 - centers[i,1]))  # top
        ])
    
    # Smooth overlap residuals
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            residuals.append(_smooth_max(radii[i] + radii[j] - dist))
    
    # Soft penalty encouraging larger radii
    for i in range(n):
        residuals.append(0.05 * (0.5 - radii[i]))
    
    return np.array(residuals)

def _objective(x, n):
    return -np.sum(x[2*n:])

def _build_constraints(n):
    cons = []
    for i in range(n):
        cons.append({'type':'ineq', 'fun': lambda x,i=i: x[2*i] - x[2*n+i]})
        cons.append({'type':'ineq', 'fun': lambda x,i=i: 1 - x[2*i] - x[2*n+i]})
        cons.append({'type':'ineq', 'fun': lambda x,i=i: x[2*i+1] - x[2*n+i]})
        cons.append({'type':'ineq', 'fun': lambda x,i=i: 1 - x[2*i+1] - x[2*n+i]})
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type':'ineq', 'fun': lambda x,i=i,j=j: 
                        (x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2 - (x[2*n+i]+x[2*n+j])**2})
    return cons

def _max_radii_for_centers(centers):
    """Compute maximum radii for fixed centers."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
    for _ in range(100):
        for i in range(n):
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(0.001, max_r)
    return radii

def _hexagonal_initial(n):
    centers = np.zeros((n, 2))
    dx, dy = 0.19, 0.19 * np.sqrt(3) / 2
    idx = 0
    y_start = 0.10
    for cnt, y in [(4,y_start), (5,y_start+dy), (6,y_start+2*dy), (6,y_start+3*dy), (5,y_start+4*dy)]:
        x0 = 0.5 - (cnt-1)*dx/2
        for c in range(cnt):
            centers[idx] = [x0 + c*dx, y]
            idx += 1
    return centers
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
