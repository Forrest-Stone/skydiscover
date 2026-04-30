# EVOLVE-BLOCK-START
"""Circle packing n=26 using scipy.optimize.minimize with trust-constr for joint position-radius optimization."""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


def construct_packing():
    """
    Joint optimization of positions and radii for 26 circles using trust-constr.
    Variables: [x0, y0, r0, ..., x25, y25, r25] (78 total).
    Objective: maximize sum of radii (minimize negative sum).
    Constraints: 325 pairwise non-overlap + 104 boundary constraints.
    Trust-constr handles many nonlinear constraints efficiently via trust-region approach.
    """
    n = 26
    c0 = initial_positions()
    r0 = safe_radii(c0)
    
    # Pack into variable vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i:3*i+3] = [c0[i, 0], c0[i, 1], r0[i]]
    
    # Objective: minimize negative sum of radii
    def obj(x):
        return -np.sum(x[2::3])
    
    # All constraints: pairwise distances >= sum of radii, boundary constraints
    def cons(x):
        vals = np.zeros(429)  # 325 pairwise + 104 boundary
        k = 0
        # Pairwise non-overlap: distance >= r_i + r_j
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2)
                vals[k] = d - x[3*i+2] - x[3*j+2]
                k += 1
        # Boundary: x-r>=0, x+r<=1, y-r>=0, y+r<=1
        for i in range(n):
            vals[k:k+4] = [x[3*i]-x[3*i+2], 1-x[3*i]-x[3*i+2],
                          x[3*i+1]-x[3*i+2], 1-x[3*i+1]-x[3*i+2]]
            k += 4
        return vals
    
    nlc = NonlinearConstraint(cons, lb=0, ub=np.inf)
    bounds = [(0.01, 0.99), (0.01, 0.99), (1e-5, 0.49)] * n
    
    result = minimize(obj, x0, method='trust-constr', constraints=nlc, bounds=bounds,
                      options={'maxiter': 500, 'gtol': 1e-6})
    
    # Extract solution
    c = np.zeros((n, 2))
    r = np.zeros(n)
    for i in range(n):
        c[i] = [result.x[3*i], result.x[3*i+1]]
        r[i] = max(1e-6, result.x[3*i+2])
    
    r = final_adjust(c, r)
    return c, r, np.sum(r)


def initial_positions():
    """Return initial positions from proven hexagonal layout."""
    n = 26
    c = np.zeros((n, 2))
    i = 0
    cr = 0.08
    c[i] = [cr, cr]; i += 1
    c[i] = [1-cr, cr]; i += 1
    c[i] = [cr, 1-cr]; i += 1
    c[i] = [1-cr, 1-cr]; i += 1
    er = 0.055
    c[i] = [0.33, er]; i += 1
    c[i] = [0.67, er]; i += 1
    c[i] = [0.33, 1-er]; i += 1
    c[i] = [0.67, 1-er]; i += 1
    c[i] = [er, 0.33]; i += 1
    c[i] = [er, 0.67]; i += 1
    c[i] = [1-er, 0.33]; i += 1
    c[i] = [1-er, 0.67]; i += 1
    for p in [(0.22,0.22),(0.50,0.22),(0.78,0.22),(0.15,0.40),(0.39,0.40),
              (0.61,0.40),(0.85,0.40),(0.15,0.60),(0.39,0.60),(0.61,0.60),
              (0.85,0.60),(0.22,0.78),(0.50,0.78),(0.78,0.78)]:
        c[i] = p; i += 1
    return c


def safe_radii(c):
    """Compute safe radii ensuring no overlaps for initial guess."""
    n = len(c)
    r = np.array([min(p[0], p[1], 1-p[0], 1-p[1]) * 0.95 for p in c])
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i, j] = d[j, i] = np.linalg.norm(c[i] - c[j])
    for _ in range(30):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if r[i] + r[j] > d[i, j] * 0.98:
                    t = r[i] + r[j]
                    r[i] = d[i, j] * r[i] / t * 0.98
                    r[j] = d[i, j] * r[j] / t * 0.98
                    changed = True
        if not changed: break
    return r * 0.98


def final_adjust(c, r):
    """Final adjustment ensuring all constraints satisfied."""
    n = len(c)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i, j] = d[j, i] = np.linalg.norm(c[i] - c[j])
    for _ in range(5):
        for i in range(n):
            for j in range(i+1, n):
                if r[i] + r[j] > d[i, j] * 0.999:
                    t = r[i] + r[j]
                    r[i] = d[i, j] * r[i] / t * 0.999
                    r[j] = d[i, j] * r[j] / t * 0.999
    for i in range(n):
        r[i] = min(r[i], c[i][0], c[i][1], 1-c[i][0], 1-c[i][1]) * 0.999
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