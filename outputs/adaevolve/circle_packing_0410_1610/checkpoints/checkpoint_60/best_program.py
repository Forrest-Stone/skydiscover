# EVOLVE-BLOCK-START
"""Voronoi-based circle packing using Lloyd's algorithm for global radius computation"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize

def construct_packing():
    """Use Voronoi iteration: compute radii from Voronoi cells simultaneously.
    Each cell's max radius = min(distance to vertices, distance to boundary).
    Move centers toward cell centroids and iterate until convergence."""
    n = 26
    best_sum, best_sol = 0, None
    
    for trial in range(15):
        centers = get_initial_positions(n, trial)
        
        # Lloyd's algorithm with Voronoi-based radius computation
        for iteration in range(80):
            radii = compute_voronoi_radii(centers, n)
            new_centers = lloyd_step(centers, n)
            if np.max(np.abs(new_centers - centers)) < 1e-7:
                break
            centers = new_centers
        
        # Final SLSQP refinement
        centers, radii = slsqp_refine(centers, radii, n)
        s = np.sum(radii)
        if s > best_sum:
            best_sum, best_sol = s, (centers.copy(), radii.copy())
    
    return best_sol[0], best_sol[1], best_sum

def compute_voronoi_radii(centers, n):
    """Compute max radius for each center from Voronoi cell geometry."""
    vor = Voronoi(centers, qhull_options='Qbb Qc Qz')
    radii = np.zeros(n)
    for i in range(n):
        # Distance to unit square boundary
        bd = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
        # Distance to Voronoi cell vertices
        region_idx = vor.point_region[i]
        if region_idx >= 0 and region_idx < len(vor.regions):
            vertices = vor.regions[region_idx]
            for v_idx in vertices:
                if v_idx >= 0 and v_idx < len(vor.vertices):
                    d = np.linalg.norm(centers[i] - vor.vertices[v_idx])
                    bd = min(bd, d)
        radii[i] = max(bd, 0.01)
    return radii

def lloyd_step(centers, n):
    """Move each center toward its Voronoi cell centroid."""
    vor = Voronoi(centers, qhull_options='Qbb Qc Qz')
    new_centers = centers.copy()
    for i in range(n):
        region_idx = vor.point_region[i]
        if region_idx >= 0 and region_idx < len(vor.regions):
            vertices = vor.regions[region_idx]
            if len(vertices) > 0 and all(v >= 0 for v in vertices):
                pts = vor.vertices[vertices]
                centroid = np.mean(pts, axis=0)
                # Blend toward centroid, staying within bounds
                new_centers[i] = np.clip(centers[i] + 0.3*(centroid - centers[i]), 0.05, 0.95)
    return new_centers

def get_initial_positions(n, trial):
    """Generate hexagonal positions with perturbation."""
    np.random.seed(trial * 17)
    centers = np.zeros((n, 2))
    r, ys = 0.095, 0.095 * np.sqrt(3)
    rows = [5, 4, 5, 4, 5, 3]
    idx = 0
    for row, cnt in enumerate(rows):
        y = r + row * ys
        if cnt == 5:
            for i in range(5): centers[idx] = [r + 2*r*i, y]; idx += 1
        elif cnt == 4:
            for i in range(4): centers[idx] = [2*r + 2*r*i, y]; idx += 1
        else:
            for i in range(3): centers[idx] = [r + 2*r*i, y]; idx += 1
    if trial > 0:
        centers += np.random.uniform(-0.03, 0.03, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
    return centers

def slsqp_refine(centers, radii, n):
    """Final SLSQP optimization for positions and radii."""
    def pack(c, r):
        x = np.zeros(3*n)
        for i in range(n): x[3*i], x[3*i+1], x[3*i+2] = c[i,0], c[i,1], r[i]
        return x
    def unpack(x):
        return np.hstack([x[0::3].reshape(n,1), x[1::3].reshape(n,1)]), x[2::3].copy()
    
    cons = []
    for i in range(n):
        cons.extend([
            {'type':'ineq','fun':lambda x,i=i:x[3*i]-x[3*i+2]},
            {'type':'ineq','fun':lambda x,i=i:1-x[3*i]-x[3*i+2]},
            {'type':'ineq','fun':lambda x,i=i:x[3*i+1]-x[3*i+2]},
            {'type':'ineq','fun':lambda x,i=i:1-x[3*i+1]-x[3*i+2]}
        ])
    for i in range(n):
        for j in range(i+1,n):
            cons.append({'type':'ineq','fun':lambda x,i=i,j=j:
                np.sqrt((x[3*i]-x[3*j])**2+(x[3*i+1]-x[3*j+1])**2)-x[3*i+2]-x[3*j+2]})
    
    bounds = [(0.01,0.99),(0.01,0.99),(0.01,0.5)]*n
    res = minimize(lambda x:-np.sum(x[2::3]), pack(centers,radii), method='SLSQP',
                   bounds=bounds, constraints=cons, options={'maxiter':1500})
    return unpack(res.x)
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