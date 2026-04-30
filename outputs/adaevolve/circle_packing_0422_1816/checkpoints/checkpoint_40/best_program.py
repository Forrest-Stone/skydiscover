# EVOLVE-BLOCK-START
"""KKT-based optimization using scipy least_squares with trust-region method."""
import numpy as np
from scipy.optimize import least_squares, minimize

def construct_packing():
    """
    Solve KKT conditions for optimal packing using trust-region least squares.
    At optimum, each circle touches boundaries or neighbors (constraints are tight).
    Uses least_squares to minimize constraint violations while maximizing radii.
    """
    n = 26
    centers_init = _hexagonal_initial(n)
    radii_init = np.array([0.5 * min(centers_init[i,0], centers_init[i,1], 
                                     1-centers_init[i,0], 1-centers_init[i,1]) for i in range(n)])
    
    best_sum = 0
    best_centers, best_radii = centers_init.copy(), radii_init.copy()
    
    np.random.seed(42)
    for restart in range(8):
        if restart == 0:
            x0 = np.concatenate([centers_init.flatten(), radii_init])
        else:
            pert = np.random.randn(3*n) * 0.03
            x0 = np.concatenate([centers_init.flatten(), radii_init]) + pert
            x0 = np.clip(x0, 0.01, 0.99)
        
        # Phase 1: Use least_squares to find good configuration
        res = least_squares(_residuals, x0, args=(n,), method='trf',
                           bounds=(np.zeros(3*n), np.ones(3*n)),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
        
        x_opt = res.x
        centers = x_opt[:2*n].reshape(n, 2)
        radii = x_opt[2*n:]
        
        # Phase 2: Refine radii with fixed centers
        radii = _max_radii_for_centers(centers)
        
        # Phase 3: Local SLSQP refinement
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

def _residuals(x, n):
    """Residuals for KKT conditions: tight constraints at optimum."""
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    residuals = []
    
    # Boundary residuals (should be zero at optimum - circles touch edges)
    for i in range(n):
        residuals.extend([
            max(0, radii[i] - centers[i,0]),      # left
            max(0, radii[i] - centers[i,1]),      # bottom
            max(0, radii[i] - (1 - centers[i,0])), # right
            max(0, radii[i] - (1 - centers[i,1]))  # top
        ])
    
    # Overlap residuals (should be non-negative)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            residuals.append(max(0, radii[i] + radii[j] - dist))
    
    # Objective: encourage larger radii (soft penalty)
    for i in range(n):
        residuals.append(0.1 * (0.5 - radii[i]))
    
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
