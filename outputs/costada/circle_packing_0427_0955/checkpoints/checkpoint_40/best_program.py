# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles in unit square"""
import numpy as np

def construct_packing():
    """Hexagonal packing with local position optimization to maximize radii sum."""
    n = 26
    centers = np.zeros((n, 2))
    
    # Improved hexagonal pattern with better spacing
    rows = [(0.08, 5), (0.25, 6), (0.44, 5), (0.63, 6), (0.84, 4)]
    
    idx = 0
    for y, cnt in rows:
        dx = 0.90 / (cnt - 1) if cnt > 1 else 0
        x0 = 0.05 if cnt == 6 else 0.08
        for i in range(cnt):
            centers[idx] = [x0 + i * dx, y]
            idx += 1
    
    # Local search optimization
    centers = optimize_positions(centers)
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def optimize_positions(centers):
    """Gradient-free local search to maximize sum of radii."""
    n = centers.shape[0]
    best_centers = centers.copy()
    best_sum = np.sum(compute_max_radii(best_centers))
    
    for _ in range(30):
        improved = False
        for i in range(n):
            for dx, dy in [(0.01,0), (-0.01,0), (0,0.01), (0,-0.01),
                          (0.005,0.005), (-0.005,-0.005), (0.005,-0.005), (-0.005,0.005)]:
                new_centers = best_centers.copy()
                new_pos = np.clip(new_centers[i] + [dx, dy], 0.02, 0.98)
                new_centers[i] = new_pos
                new_sum = np.sum(compute_max_radii(new_centers))
                if new_sum > best_sum + 1e-9:
                    best_centers = new_centers
                    best_sum = new_sum
                    improved = True
        if not improved:
            break
    return best_centers

def compute_max_radii(centers):
    """Compute max valid radii using proportional overlap resolution."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d + 1e-10:
                    total = radii[i] + radii[j]
                    radii[i] = d * radii[i] / total
                    radii[j] = d * radii[j] / total
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
