# EVOLVE-BLOCK-START
"""Multi-pattern circle packing for n=26 circles in a unit square"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """
    Multi-pattern approach: Try multiple fundamentally different row 
    configurations for 22 non-corner circles with various spacing parameters.
    """
    n = 26
    best_sum, best_centers, best_radii = 0, None, None
    
    # Patterns that sum to 22, each with tuned base radius
    pattern_configs = [
        ([4, 5, 5, 4, 4], 0.08),    # Original balanced
        ([5, 5, 5, 5, 2], 0.085),   # Heavy top/bottom
        ([6, 5, 5, 5, 1], 0.08),    # Very heavy top
        ([4, 6, 6, 4, 2], 0.075),   # Wide middle
        ([3, 5, 6, 5, 3], 0.08),    # Hourglass
        ([2, 6, 6, 6, 2], 0.075),   # Edge-heavy
        ([5, 6, 5, 6, 0], 0.075),   # Alternating
        ([4, 5, 4, 5, 4], 0.085),   # Alternating variant
    ]
    
    corner_radii = [0.10, 0.11, 0.12, 0.13]
    
    for pattern, base_r in pattern_configs:
        for corner_r in corner_radii:
            centers = init_pattern(n, pattern, corner_r, base_r)
            radii = compute_max_radii(centers)
            centers, radii = slsqp_refine(centers, radii)
            total = np.sum(radii)
            if total > best_sum:
                best_sum, best_centers, best_radii = total, centers.copy(), radii.copy()
    
    return best_centers, best_radii, best_sum


def init_pattern(n, pattern, corner_r, base_r):
    """Initialize circles with given pattern and hexagonal offset."""
    centers = np.zeros((n, 2))
    
    # 4 corner circles
    centers[0] = [corner_r, corner_r]
    centers[1] = [1 - corner_r, corner_r]
    centers[2] = [corner_r, 1 - corner_r]
    centers[3] = [1 - corner_r, 1 - corner_r]
    
    h = 2 * base_r
    v = np.sqrt(3) * base_r
    
    total_rows = len(pattern)
    y_start = 0.5 - (total_rows - 1) * v / 2
    
    idx = 4
    for ri, cnt in enumerate(pattern):
        if cnt == 0:
            continue
        y = y_start + ri * v
        x_start = 0.5 - (cnt - 1) * h / 2
        offset = h/2 if ri % 2 == 1 else 0
        for c in range(cnt):
            centers[idx] = [x_start + offset + c * h, y]
            idx += 1
    
    return centers


def compute_max_radii(centers):
    """Compute max possible radius for each center iteratively."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        c = centers[i]
        radii[i] = min(c[0], c[1], 1-c[0], 1-c[1])
        for j in range(n):
            if i != j:
                radii[i] = min(radii[i], np.linalg.norm(c - centers[j]) * 0.5)
    # Expand iteratively to fill available space
    for _ in range(50):
        for i in range(n):
            mx = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    mx = min(mx, np.linalg.norm(centers[i] - centers[j]) - radii[j])
            radii[i] = max(radii[i], mx * 0.999)
    return radii


def slsqp_refine(centers, radii):
    """SLSQP optimization of all positions and radii."""
    n = len(centers)
    p0 = np.zeros(3 * n)
    for i in range(n):
        p0[3*i:3*i+3] = [centers[i,0], centers[i,1], radii[i]]
    
    cons = []
    for i in range(n):
        cons.extend([
            {'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]},
            {'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]},
            {'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]},
            {'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]}
        ])
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j:
                np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]})
    
    result = minimize(lambda p: -np.sum(p[2::3]), p0, method='SLSQP',
                     constraints=cons, bounds=[(0,1),(0,1),(0,0.5)]*n,
                     options={'ftol': 1e-10, 'maxiter': 500})
    return result.x.reshape(n, 3)[:, :2], result.x[2::3]
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