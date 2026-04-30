# EVOLVE-BLOCK-START
"""Voronoi-guided iterative packing for n=26 circles in a unit square"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize

def construct_packing():
    """
    Hybrid approach: Voronoi-guided positioning + SLSQP refinement.
    1. Initialize with hexagonal pattern
    2. Lloyd relaxation using Voronoi centroids
    3. Compute max radii from Voronoi cells
    4. Final SLSQP refinement
    """
    n = 26
    centers = init_hex_grid(n)
    
    # Lloyd relaxation: move centers toward Voronoi centroids
    for iteration in range(200):
        vor = Voronoi(centers, qhull_options='Qbb Qc Qz')
        new_centers = lloyd_step(vor, centers)
        new_centers = np.clip(new_centers, 0.02, 0.98)
        if np.max(np.abs(new_centers - centers)) < 1e-8:
            break
        centers = new_centers
    
    # Compute initial radii from Voronoi cells
    radii = compute_max_radii(centers)
    
    # Final SLSQP refinement
    centers, radii = slsqp_refine(centers, radii)
    
    return centers, radii, np.sum(radii)


def lloyd_step(vor, centers, alpha=0.3):
    """Move each center toward its Voronoi cell centroid."""
    n = len(centers)
    new_centers = np.zeros_like(centers)
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if -1 not in region and len(region) >= 3:
            verts = np.clip(vor.vertices[region], 0, 1)
            centroid = np.mean(verts, axis=0)
            new_centers[i] = (1-alpha)*centers[i] + alpha*centroid
        else:
            new_centers[i] = centers[i]
    return new_centers


def compute_max_radii(centers):
    """Compute max possible radius for each center."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        c = centers[i]
        max_r = min(c[0], c[1], 1-c[0], 1-c[1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(c - centers[j])
                max_r = min(max_r, d * 0.5)
        radii[i] = max_r * 0.99
    # Expand iteratively
    for _ in range(50):
        for i in range(n):
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(radii[i], max_r * 0.999)
    return radii


def slsqp_refine(centers, radii):
    """Final SLSQP refinement of positions and radii."""
    n = len(centers)
    params0 = np.zeros(3*n)
    for i in range(n):
        params0[3*i] = centers[i, 0]
        params0[3*i+1] = centers[i, 1]
        params0[3*i+2] = radii[i]
    
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt((p[3*i]-p[3*j])**2 + 
                         (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
            })
    
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    result = minimize(
        lambda p: -np.sum(p[2::3]),
        params0, method='SLSQP', constraints=cons, bounds=bounds,
        options={'ftol': 1e-10, 'maxiter': 500}
    )
    
    new_centers = result.x.reshape(n, 3)[:, :2]
    new_radii = result.x[2::3]
    return new_centers, new_radii


def init_hex_grid(n):
    """Initialize from 5-6-5-6-4 hexagonal pattern."""
    centers = np.zeros((n, 2))
    r = 0.095
    h, v = 2 * r, np.sqrt(3) * r
    rows = [(5, 0), (6, h/2), (5, 0), (6, h/2), (4, h)]
    y_start = 0.5 - 2 * v
    idx = 0
    for row_idx, (count, x_off) in enumerate(rows):
        y = y_start + row_idx * v
        x_start = 0.5 - (count - 1) * h / 2
        for col in range(count):
            x = x_start + x_off + col * h
            centers[idx] = [np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)]
            idx += 1
    return centers

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