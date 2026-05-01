# EVOLVE-BLOCK-START
"""Joint optimization of circle positions and radii using SLSQP."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Joint optimization of positions and radii for 26 circles using SLSQP.
    Variables: [x0..x25, y0..y25, r0..r25] = 78 variables.
    Constraints: 325 non-overlap + 104 boundary = 429 inequality constraints.
    Uses multiple initializations to escape local optima.
    """
    n = 26
    best_sum = 0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    # Multiple initializations: hexagonal, corner-heavy, random seeds
    inits = [
        hexagonal_init(n),
        corner_init(n),
        dense_init(n),
        random_init(n, 42),
        random_init(n, 123),
    ]
    
    for x0 in inits:
        result = optimize_packing(n, x0)
        if result is not None:
            centers = np.column_stack([result.x[:n], result.x[n:2*n]])
            radii = result.x[2*n:]
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


def optimize_packing(n, x0):
    """SLSQP optimization for circle packing problem."""
    def objective(v):
        return -np.sum(v[2*n:])
    
    constraints = []
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt(
                    (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2
                ) - v[2*n+i] - v[2*n+j]
            })
    
    # Boundary constraints
    for i in range(n):
        constraints.extend([
            {'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]},      # x >= r
            {'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2*n+i]},  # x <= 1-r
            {'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]},    # y >= r
            {'type': 'ineq', 'fun': lambda v, i=i: 1 - v[n+i] - v[2*n+i]},# y <= 1-r
        ])
    
    bounds = [(0, 1)]*n + [(0, 1)]*n + [(1e-8, 0.5)]*n
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, 
                      options={'maxiter': 2000, 'ftol': 1e-10})
    return result


def hexagonal_init(n):
    """Hexagonal pattern initialization."""
    r = 0.094
    x, y = [], []
    rows = [(5, r), (4, 2*r), (5, r), (4, 2*r), (5, r), (3, 2*r)]
    dy = r * np.sqrt(3)
    yi = r
    for cnt, xs in rows:
        for i in range(cnt):
            x.append(xs + i * 2 * r)
            y.append(yi)
        yi += dy
    return np.array(x + y + [r]*n)


def corner_init(n):
    """Corner-heavy initialization with large corner circles."""
    r = 0.12
    x = [r, 1-r, r, 1-r]  # 4 corners
    y = [r, r, 1-r, 1-r]
    # Edge circles
    x.extend([0.5, 0.5, r, 1-r])
    y.extend([r, 1-r, 0.5, 0.5])
    # Interior grid
    for i in range(18):
        x.append(0.15 + (i % 5) * 0.17)
        y.append(0.22 + (i // 5) * 0.18)
    return np.array(x + y + [r]*n)


def dense_init(n):
    """Dense packing initialization with varied radii."""
    x, y, r = [], [], []
    # Large corner circles
    rc = 0.11
    x.extend([rc, 1-rc, rc, 1-rc])
    y.extend([rc, rc, 1-rc, 1-rc])
    r.extend([rc]*4)
    # Medium edge circles
    re = 0.08
    x.extend([0.3, 0.7, 0.3, 0.7, re, 1-re, 0.5, 0.5])
    y.extend([re, re, 1-re, 1-re, 0.3, 0.3, re, 1-re])
    r.extend([re]*8)
    # Smaller interior circles
    ri = 0.07
    for i in range(14):
        x.append(0.18 + (i % 5) * 0.16)
        y.append(0.25 + (i // 5) * 0.20)
        r.append(ri)
    return np.array(x + y + r)


def random_init(n, seed):
    """Random initialization with feasible starting point."""
    np.random.seed(seed)
    x = np.random.uniform(0.15, 0.85, n)
    y = np.random.uniform(0.15, 0.85, n)
    r = np.ones(n) * 0.06
    return np.concatenate([x, y, r])


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
