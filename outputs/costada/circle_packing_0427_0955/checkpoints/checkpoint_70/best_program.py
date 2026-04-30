# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles using multi-start optimization."""
import numpy as np

def construct_packing():
    """Multi-start local optimization with varied hexagonal patterns."""
    n = 26
    best_sum, best_centers = 0, None
    
    # Multiple initial patterns with different y-spacings
    patterns = [
        [(0.07, 5), (0.24, 6), (0.42, 5), (0.60, 6), (0.78, 4)],
        [(0.08, 5), (0.26, 6), (0.44, 5), (0.62, 6), (0.80, 4)],
        [(0.06, 4), (0.22, 6), (0.40, 6), (0.58, 6), (0.76, 4)],
        [(0.09, 5), (0.27, 6), (0.45, 5), (0.63, 6), (0.81, 4)],
    ]
    
    for rows in patterns:
        centers = build_pattern(n, rows)
        centers = optimize(centers)
        s = np.sum(compute_radii(centers))
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    return best_centers, compute_radii(best_centers), best_sum

def build_pattern(n, rows):
    """Build circle centers from row specification."""
    centers = np.zeros((n, 2))
    idx = 0
    for y, cnt in rows:
        dx = 0.86 / max(cnt - 1, 1)
        x0 = 0.07
        for i in range(cnt):
            centers[idx] = [x0 + i * dx, y]
            idx += 1
    return centers

def optimize(centers):
    """Aggressive local search with decreasing step size."""
    n = len(centers)
    best, best_sum = centers.copy(), np.sum(compute_radii(centers))
    
    for iteration in range(100):
        improved = False
        step = 0.025 * (1 - iteration / 100)  # Decreasing step
        for i in range(n):
            moves = [(step,0), (-step,0), (0,step), (0,-step),
                     (step*0.7, step*0.7), (-step*0.7, -step*0.7),
                     (step*0.7, -step*0.7), (-step*0.7, step*0.7)]
            for dx, dy in moves:
                new = best.copy()
                new[i] = np.clip(new[i] + [dx, dy], 0.03, 0.97)
                s = np.sum(compute_radii(new))
                if s > best_sum + 1e-9:
                    best, best_sum, improved = new, s, True
        if not improved and step < 0.003:
            break
    return best

def compute_radii(centers):
    """Compute max valid radii with proportional overlap resolution."""
    n = len(centers)
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for _ in range(80):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d + 1e-10:
                    t = radii[i] + radii[j]
                    radii[i] = d * radii[i] / t
                    radii[j] = d * radii[j] / t
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
