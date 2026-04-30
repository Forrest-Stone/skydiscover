# EVOLVE-BLOCK-START
"""Circle packing n=26 using scipy.optimize.minimize with COBYLA - optimized constraint handling."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint optimization of positions and radii for 26 circles using COBYLA.
    Optimized constraint handling: 325 pairwise + 104 boundary constraints (429 total).
    Uses vectorized constraint evaluation for efficiency.
    Multi-start with different initial configurations to escape local optima.
    Variables: [x0, y0, r0, ..., x25, y25, r25] (78 total).
    """
    n = 26
    best_sum = 0
    best_c, best_r = None, None
    
    # Try multiple initial configurations
    for config_type in ['hex_corner', 'hex_edge', 'uniform']:
        c0 = get_initial_positions(config_type)
        r0 = safe_radii(c0)
        
        # Pack into variable vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i:3*i+3] = [c0[i, 0], c0[i, 1], r0[i]]
        
        # Build constraints (reduced from 585 to 429)
        cons_list = []
        
        # Pairwise non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                cons_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
                })
        
        # Boundary constraints (merged with bounds)
        for i in range(n):
            cons_list.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2] - 1e-6})
            cons_list.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2] - 1e-6})
            cons_list.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2] - 1e-6})
            cons_list.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2] - 1e-6})
        
        result = minimize(lambda x: -np.sum(x[2::3]), x0, method='cobyla', 
                          constraints=cons_list,
                          options={'rhobeg': 0.05, 'rhoend': 1e-7, 'maxiter': 2000})
        
        # Extract solution
        c = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            c[i] = [result.x[3*i], result.x[3*i+1]]
            r[i] = max(1e-6, result.x[3*i+2])
        
        r = final_adjust(c, r)
        current_sum = np.sum(r)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_c, best_r = c.copy(), r.copy()
    
    return best_c, best_r, best_sum


def get_initial_positions(config_type='hex_corner'):
    """Return initial positions based on configuration type."""
    n = 26
    c = np.zeros((n, 2))
    
    if config_type == 'hex_corner':
        # Hexagonal with larger corner circles
        i = 0
        cr = 0.095
        c[i] = [cr, cr]; i += 1
        c[i] = [1-cr, cr]; i += 1
        c[i] = [cr, 1-cr]; i += 1
        c[i] = [1-cr, 1-cr]; i += 1
        er = 0.06
        c[i] = [0.30, er]; i += 1
        c[i] = [0.70, er]; i += 1
        c[i] = [0.30, 1-er]; i += 1
        c[i] = [0.70, 1-er]; i += 1
        c[i] = [er, 0.30]; i += 1
        c[i] = [er, 0.70]; i += 1
        c[i] = [1-er, 0.30]; i += 1
        c[i] = [1-er, 0.70]; i += 1
        for p in [(0.22,0.22),(0.50,0.22),(0.78,0.22),(0.15,0.40),(0.39,0.40),
                  (0.61,0.40),(0.85,0.40),(0.15,0.60),(0.39,0.60),(0.61,0.60),
                  (0.85,0.60),(0.22,0.78),(0.50,0.78),(0.78,0.78)]:
            c[i] = p; i += 1
    elif config_type == 'hex_edge':
        # More edge circles, smaller corners
        i = 0
        cr = 0.085
        c[i] = [cr, cr]; i += 1
        c[i] = [1-cr, cr]; i += 1
        c[i] = [cr, 1-cr]; i += 1
        c[i] = [1-cr, 1-cr]; i += 1
        er = 0.055
        c[i] = [0.25, er]; i += 1
        c[i] = [0.50, er]; i += 1
        c[i] = [0.75, er]; i += 1
        c[i] = [0.25, 1-er]; i += 1
        c[i] = [0.50, 1-er]; i += 1
        c[i] = [0.75, 1-er]; i += 1
        c[i] = [er, 0.25]; i += 1
        c[i] = [er, 0.50]; i += 1
        c[i] = [er, 0.75]; i += 1
        c[i] = [1-er, 0.25]; i += 1
        c[i] = [1-er, 0.50]; i += 1
        c[i] = [1-er, 0.75]; i += 1
        for p in [(0.20,0.20),(0.50,0.20),(0.80,0.20),(0.35,0.50),(0.65,0.50),
                  (0.20,0.80),(0.50,0.80),(0.80,0.80)]:
            c[i] = p; i += 1
    else:  # uniform
        # Grid-like uniform distribution
        idx = 0
        for row in range(5):
            for col in range(5):
                if idx < n:
                    c[idx] = [0.1 + 0.2*col, 0.1 + 0.2*row]
                    idx += 1
        c[24] = [0.5, 0.5]
        c[25] = [0.5, 0.9]
    
    return c


def safe_radii(c):
    """Compute safe radii ensuring no overlaps for initial guess."""
    n = len(c)
    r = np.array([min(p[0], p[1], 1-p[0], 1-p[1]) * 0.9 for p in c])
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i, j] = d[j, i] = np.linalg.norm(c[i] - c[j])
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if r[i] + r[j] > d[i, j] * 0.99:
                    t = r[i] + r[j]
                    if t > 1e-10:
                        r[i] = d[i, j] * r[i] / t * 0.99
                        r[j] = d[i, j] * r[j] / t * 0.99
                        changed = True
        if not changed:
            break
    return r * 0.99


def final_adjust(c, r):
    """Final adjustment ensuring all constraints satisfied."""
    n = len(c)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i, j] = d[j, i] = np.linalg.norm(c[i] - c[j])
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if r[i] + r[j] > d[i, j] * 0.9999:
                    t = r[i] + r[j]
                    if t > 1e-10:
                        r[i] = d[i, j] * r[i] / t * 0.9999
                        r[j] = d[i, j] * r[j] / t * 0.9999
                        changed = True
        if not changed:
            break
    for i in range(n):
        r[i] = min(r[i], c[i][0], c[i][1], 1-c[i][0], 1-c[i][1]) * 0.9999
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