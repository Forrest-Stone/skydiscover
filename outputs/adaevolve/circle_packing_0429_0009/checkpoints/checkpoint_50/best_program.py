# EVOLVE-BLOCK-START
"""Corner-priority circle packing for n=26 circles in a unit square"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """
    Corner-priority approach: Place 4 larger circles at corners first,
    then 22 in hexagonal center pattern. Corners allow larger radii
    since circles can touch two walls simultaneously.
    """
    n = 26
    best_sum, best_centers, best_radii = 0, None, None
    
    # Try multiple corner radii to find best starting configuration
    for corner_r in [0.11, 0.12, 0.13, 0.14]:
        centers = init_corner_hex(n, corner_r)
        radii = compute_max_radii(centers)
        centers, radii = slsqp_refine(centers, radii)
        total = np.sum(radii)
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers.copy(), radii.copy()
    
    return best_centers, best_radii, best_sum


def init_corner_hex(n, corner_r):
    """4 corner circles + 22 center circles in 4-5-5-4-4 hexagonal pattern."""
    centers = np.zeros((n, 2))
    # 4 corner circles - corners naturally allow larger radii
    centers[0] = [corner_r, corner_r]
    centers[1] = [1 - corner_r, corner_r]
    centers[2] = [corner_r, 1 - corner_r]
    centers[3] = [1 - corner_r, 1 - corner_r]
    
    # 22 center circles in 4-5-5-4-4 hexagonal pattern
    r = 0.08
    h, v = 2 * r, np.sqrt(3) * r
    rows = [(4, 0), (5, h/2), (5, 0), (4, h/2), (4, 0)]
    y_start = 0.5 - 2 * v
    idx = 4
    for ri, (cnt, xo) in enumerate(rows):
        y = y_start + ri * v
        xs = 0.5 - (cnt - 1) * h / 2
        for c in range(cnt):
            centers[idx] = [xs + xo + c * h, y]
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