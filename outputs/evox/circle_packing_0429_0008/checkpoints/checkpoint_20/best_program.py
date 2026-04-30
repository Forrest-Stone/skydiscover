# EVOLVE-BLOCK-START
"""Joint optimization of positions and radii using scipy.optimize.minimize"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def construct_packing():
    """
    Joint optimization of circle positions and radii using SLSQP.
    Uses multiple geometric initial configurations and selects the best result.
    Exploits corner/edge advantages for larger circles.
    """
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Multiple initial configurations to try
    initial_configs = [
        create_corner_heavy_config(n),
        create_hexagonal_config(n),
        create_mixed_size_config(n),
    ]
    
    for init_centers, init_radii in initial_configs:
        centers, radii, total = optimize_packing(init_centers, init_radii)
        if total > best_sum:
            best_sum = total
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


def create_corner_heavy_config(n):
    """Create initial config with large corner circles, medium edges, small interior."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 4 corner circles (large)
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for i, (cx, cy) in enumerate(corners):
        centers[i] = [cx * 0.87 + 0.065, cy * 0.87 + 0.065]
        radii[i] = 0.12
    
    # 4 edge circles per side (medium) - positions 4-19
    idx = 4
    # Bottom edge
    for x in [0.25, 0.5, 0.75]:
        centers[idx] = [x, 0.08]
        radii[idx] = 0.08
        idx += 1
    # Top edge
    for x in [0.25, 0.5, 0.75]:
        centers[idx] = [x, 0.92]
        radii[idx] = 0.08
        idx += 1
    # Left edge
    for y in [0.25, 0.5, 0.75]:
        centers[idx] = [0.08, y]
        radii[idx] = 0.08
        idx += 1
    # Right edge
    for y in [0.25, 0.5, 0.75]:
        centers[idx] = [0.92, y]
        radii[idx] = 0.08
        idx += 1
    
    # Interior circles (small) - remaining positions
    interior_positions = [
        (0.33, 0.33), (0.67, 0.33), (0.33, 0.67), (0.67, 0.67),
        (0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)
    ]
    for pos in interior_positions:
        if idx < n:
            centers[idx] = pos
            radii[idx] = 0.05
            idx += 1
    
    return centers, radii


def create_hexagonal_config(n):
    """Create hexagonal lattice pattern with offset rows."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Hexagonal pattern: rows of 5-4-5-4-5-4 alternating
    row_configs = [
        (5, 0.10, 0.08),   # row 0: 5 circles, y=0.08
        (4, 0.20, 0.24),   # row 1: 4 circles, y=0.24
        (5, 0.10, 0.40),   # row 2: 5 circles, y=0.40
        (4, 0.20, 0.56),   # row 3: 4 circles, y=0.56
        (5, 0.10, 0.72),   # row 4: 5 circles, y=0.72
        (4, 0.20, 0.88),   # row 5: 4 circles, y=0.88
    ]
    
    idx = 0
    for num_circles, x_start, y in row_configs:
        spacing = (1.0 - 2 * x_start) / max(1, num_circles - 1) if num_circles > 1 else 0
        for i in range(num_circles):
            if idx < n:
                centers[idx] = [x_start + i * spacing, y]
                radii[idx] = 0.08
                idx += 1
    
    # Fill remaining with small circles
    while idx < n:
        centers[idx] = [0.5, 0.5]
        radii[idx] = 0.03
        idx += 1
    
    return centers, radii


def create_mixed_size_config(n):
    """Create config optimizing for varied circle sizes."""
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 4 large corner circles
    for i, (cx, cy) in enumerate([(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]):
        centers[i] = [cx, cy]
        radii[i] = 0.1
    
    # 8 medium edge circles
    edge_positions = [
        (0.3, 0.05), (0.7, 0.05), (0.3, 0.95), (0.7, 0.95),
        (0.05, 0.3), (0.05, 0.7), (0.95, 0.3), (0.95, 0.7)
    ]
    for i, pos in enumerate(edge_positions):
        centers[4 + i] = pos
        radii[4 + i] = 0.07
    
    # 14 small interior circles in grid
    interior_idx = 0
    for row in range(4):
        for col in range(4):
            if interior_idx < 14:
                centers[12 + interior_idx] = [0.2 + col * 0.2, 0.2 + row * 0.2]
                radii[12 + interior_idx] = 0.05
                interior_idx += 1
    
    return centers, radii


def optimize_packing(init_centers, init_radii):
    """
    Use SLSQP to jointly optimize positions and radii.
    Variables: [x0, y0, r0, x1, y1, r1, ...] flattened.
    """
    n = len(init_radii)
    
    # Pack initial values: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = init_centers[i, 0]
        x0[3*i + 1] = init_centers[i, 1]
        x0[3*i + 2] = init_radii[i]
    
    # Objective: minimize negative sum of radii
    def objective(x):
        return -np.sum(x[2::3])
    
    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2::3] = -1.0
        return grad
    
    # Constraints
    constraints = []
    
    # Boundary constraints: r <= x, r <= y, r <= 1-x, r <= 1-y
    for i in range(n):
        xi, yi, ri = 3*i, 3*i + 1, 3*i + 2
        # r - x <= 0 (circle must fit horizontally from left)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=xi, ridx=ri: x[idx] - x[ridx]})
        # r - y <= 0 (circle must fit vertically from bottom)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=yi, ridx=ri: x[idx] - x[ridx]})
        # r - (1-x) <= 0 (circle must fit horizontally from right)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=xi, ridx=ri: 1 - x[idx] - x[ridx]})
        # r - (1-y) <= 0 (circle must fit vertically from top)
        constraints.append({'type': 'ineq', 'fun': lambda x, idx=yi, ridx=ri: 1 - x[idx] - x[ridx]})
    
    # Circle-circle non-overlap: ||ci - cj|| >= ri + rj
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
            })
    
    # Bounds: x, y in [0, 1], r in [0.001, 0.5]
    bounds = [(0, 1), (0, 1), (0.001, 0.5)] * n
    
    result = minimize(
        objective, x0, method='SLSQP', jac=objective_grad,
        bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # Extract optimized values
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = result.x[3*i]
        centers[i, 1] = result.x[3*i + 1]
        radii[i] = result.x[3*i + 2]
    
    # Ensure validity with final constraint check
    radii = enforce_valid_radii(centers, radii)
    
    return centers, radii, np.sum(radii)


def enforce_valid_radii(centers, radii):
    """Ensure all constraints are satisfied by shrinking radii if needed."""
    n = len(radii)
    radii = radii.copy()
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1 - x, 1 - y)
    
    # Circle-circle constraints
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist + 1e-10:
                    total = radii[i] + radii[j]
                    if total > 1e-10:
                        scale = dist / total
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
        if not changed:
            break
    
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
