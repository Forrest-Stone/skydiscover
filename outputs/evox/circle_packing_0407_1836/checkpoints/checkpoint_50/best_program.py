# EVOLVE-BLOCK-START
"""Enhanced optimization with more patterns, multi-start, and post-processing."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Enhanced optimization with more initial patterns, multi-start refinement,
    and post-processing radius expansion for improved sum of radii.
    """
    n = 26
    
    def objective(vars):
        radii = vars[52:]
        return -np.sum(radii)
    
    def constraints(vars):
        centers = vars[:52].reshape(n, 2)
        radii = vars[52:]
        cons = []
        for i in range(n):
            cons.extend([
                centers[i, 0] - radii[i],
                1 - centers[i, 0] - radii[i],
                centers[i, 1] - radii[i],
                1 - centers[i, 1] - radii[i]
            ])
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                cons.append(d2 - (radii[i] + radii[j])**2)
        return np.array(cons)
    
    # Extended pattern set with asymmetric configurations
    patterns = [
        [5, 4, 5, 4, 5, 3], [4, 5, 4, 5, 4, 4], [5, 5, 6, 5, 5], [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5], [4, 6, 6, 4, 4, 2], [6, 4, 6, 4, 6], [5, 4, 6, 4, 5, 2],
        [4, 5, 5, 5, 5, 2], [6, 6, 4, 6, 4], [5, 5, 5, 5, 4, 2], [4, 4, 6, 6, 4, 2]
    ]
    
    # Collect top initializations for multi-start
    inits = []
    for rows in patterns:
        for h in np.linspace(0.13, 0.24, 20):
            centers = create_hex_pattern(rows, h)
            radii = compute_max_radii(centers)
            s = np.sum(radii)
            inits.append((s, np.concatenate([centers.flatten(), radii])))
    
    # Sort by sum and keep top configurations
    inits.sort(reverse=True, key=lambda x: x[0])
    top_inits = [x[1] for x in inits[:8]]
    
    # Run optimization from top initializations
    best_result = None
    best_sum = 0
    bounds = [(0.01, 0.99)] * 52 + [(0, 0.16)] * 26
    
    for init in top_inits:
        result = minimize(
            objective, init, method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraints},
            bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-11}
        )
        s = np.sum(result.x[52:])
        if s > best_sum:
            best_sum = s
            best_result = result.x.copy()
    
    centers = best_result[:52].reshape(n, 2)
    radii = best_result[52:]
    radii = np.maximum(radii, 0)
    
    # Post-processing: expand radii iteratively
    radii = expand_radii(centers, radii)
    
    return centers, radii, np.sum(radii)


def expand_radii(centers, radii):
    """Post-process to greedily expand each radius to its maximum."""
    n = len(radii)
    for _ in range(300):
        improved = False
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, y, 1 - x, 1 - y)
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] + 1e-9:
                radii[i] = max_r
                improved = True
        if not improved:
            break
    return np.maximum(radii, 0)


def create_hex_pattern(rows, h):
    """Create centered hexagonal pattern with alternating row offsets."""
    n = sum(rows)
    centers = np.zeros((n, 2))
    v = h * np.sqrt(3) / 2
    
    total_height = (len(rows) - 1) * v
    y_start = (1 - total_height) / 2
    
    idx = 0
    y = y_start
    for row_idx, count in enumerate(rows):
        offset = h / 2 if row_idx % 2 == 1 else 0
        row_width = (count - 1) * h
        x_start = (1 - row_width) / 2
        for i in range(count):
            centers[idx] = [x_start + i * h + offset, y]
            idx += 1
        y += v
    
    return centers


def compute_max_radii(centers):
    """Iteratively compute maximum feasible radii."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1 - x, 1 - y) for x, y in centers])
    
    for _ in range(100):
        for i in range(n):
            max_r = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    max_r = min(max_r, dist - radii[j])
            radii[i] = max(0, max_r)
    
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
