# EVOLVE-BLOCK-START
"""Optimized hexagonal packing with position refinement for n=26 circles"""
import numpy as np

def construct_packing():
    """Hexagonal grid with optimized spacing and local position refinement."""
    n = 26
    best_centers, best_radii, best_sum = None, None, 0
    
    # Try multiple base radii for hexagonal spacing
    for r_base in [0.092, 0.094, 0.096, 0.098]:
        centers = create_hex_grid(n, r_base)
        radii = compute_max_radii_iterative(centers)
        centers, radii = refine_positions(centers, radii)
        s = np.sum(radii)
        if s > best_sum:
            best_sum, best_centers, best_radii = s, centers.copy(), radii.copy()
    
    return best_centers, best_radii, best_sum


def create_hex_grid(n, r):
    """Create hexagonal grid with given base radius."""
    centers = np.zeros((n, 2))
    dx, dy = 2.0 * r, r * np.sqrt(3)
    idx, row = 0, 0
    y = r
    while idx < n and y < 1 - r:
        x = r if row % 2 == 0 else r + dx / 2
        while idx < n and x < 1 - r:
            centers[idx] = [x, y]
            idx += 1
            x += dx
        y += dy
        row += 1
    while idx < n:
        y = 1 - r
        x = r if row % 2 == 0 else r + dx / 2
        while idx < n and x < 1 - r:
            centers[idx] = [x, y]
            idx += 1
            x += dx
        row += 1
    return np.clip(centers, 0.001, 0.999)


def compute_max_radii_iterative(centers, n_iter=100):
    """Iteratively expand each circle to its maximum feasible radius."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for _ in range(n_iter):
        improved = False
        for i in range(n):
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, max(0, d - radii[j]))
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        if not improved:
            break
    return radii


def refine_positions(centers, radii, n_steps=15, step_size=0.008):
    """Local position refinement to maximize sum of radii."""
    n = centers.shape[0]
    best_c, best_r, best_s = centers.copy(), radii.copy(), np.sum(radii)
    
    for step in range(n_steps):
        improved = False
        for i in range(n):
            for dx, dy in [(0.006,0), (-0.006,0), (0,0.006), (0,-0.006), (0.004,0.004), (-0.004,-0.004)]:
                new_c = centers.copy()
                new_c[i] = np.clip(centers[i] + np.array([dx, dy]) * step_size / 0.006, 0.001, 0.999)
                new_r = compute_max_radii_iterative(new_c, n_iter=50)
                s = np.sum(new_r)
                if s > best_s + 1e-8:
                    best_s, best_c, best_r = s, new_c.copy(), new_r.copy()
                    improved = True
        if improved:
            centers = best_c.copy()
        step_size *= 0.9
    return best_c, best_r


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
