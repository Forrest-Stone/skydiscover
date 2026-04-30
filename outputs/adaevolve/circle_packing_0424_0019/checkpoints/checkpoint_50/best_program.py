# EVOLVE-BLOCK-START
"""Circle packing for n=26 using joint position-radius optimization with SLSQP"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """
    Joint optimization of positions AND radii using SLSQP.
    Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25] = 78 variables
    This allows positions to shift to accommodate larger circles.
    """
    n = 26
    centers_init = np.zeros((n, 2))
    
    # Initial hexagonal pattern
    spacing = 0.2
    y_step = spacing * np.sqrt(3) / 2
    
    idx = 0
    for i in range(5):
        centers_init[idx] = [0.1 + i * spacing, 0.1]; idx += 1
    for i in range(4):
        centers_init[idx] = [0.2 + i * spacing, 0.1 + y_step]; idx += 1
    for i in range(5):
        centers_init[idx] = [0.1 + i * spacing, 0.1 + 2*y_step]; idx += 1
    for i in range(4):
        centers_init[idx] = [0.2 + i * spacing, 0.1 + 3*y_step]; idx += 1
    for i in range(5):
        centers_init[idx] = [0.1 + i * spacing, 0.1 + 4*y_step]; idx += 1
    for x in [0.1, 0.5, 0.9]:
        centers_init[idx] = [x, 0.95]; idx += 1
    
    # Pack into vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i], x0[3*i+1] = centers_init[i]
        x0[3*i+2] = 0.08  # Initial radius estimate
    
    def objective(x):
        return -np.sum(x[2::3])  # Maximize sum of radii
    
    def constraints(x):
        cons = []
        # Circle-circle: ||ci - cj|| >= ri + rj
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                dx, dy = xi - x[3*j], yi - x[3*j+1]
                cons.append(np.sqrt(dx*dx + dy*dy) - ri - x[3*j+2])
        # Boundary: 4 linear constraints per circle (smooth, differentiable)
        for i in range(n):
            cx, cy, r = x[3*i], x[3*i+1], x[3*i+2]
            cons.extend([cx - r, 1 - cx - r, cy - r, 1 - cy - r])
        return np.array(cons)
    
    bounds = []
    for _ in range(n):
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.5)])
    
    result = minimize(objective, x0, method='SLSQP',
                     constraints={'type': 'ineq', 'fun': constraints},
                     bounds=bounds, options={'ftol': 1e-9, 'maxiter': 2000})
    
    centers = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
    radii = result.x[2::3]
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