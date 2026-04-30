# EVOLVE-BLOCK-START
"""Multi-start SLSQP optimization for circle packing n=26 in unit square"""
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    """Use SLSQP with multiple restarts to jointly optimize positions and radii."""
    n = 26
    best_sum, best_centers, best_radii = 0, None, None
    
    # Generate multiple initial configurations
    inits = _generate_initializations(n)
    
    for x0 in inits:
        res = _run_slsqp(x0, n)
        if res is None:
            continue
        centers, radii = _extract_solution(res.x, n)
        centers, radii = _ensure_validity(centers, radii)
        radii = _expand_radii(centers, radii)
        total = np.sum(radii)
        if total > best_sum:
            best_sum, best_centers, best_radii = total, centers.copy(), radii.copy()
    
    return best_centers, best_radii, best_sum


def _generate_initializations(n):
    """Create diverse initial configurations for multi-start optimization."""
    inits = []
    np.random.seed(42)
    
    # Base hexagonal pattern with optimized spacing
    base = np.zeros(3*n)
    rows = [5, 4, 5, 4, 5, 3]
    idx = 0
    for row, cnt in enumerate(rows):
        y = 0.08 + row * 0.15
        x_start = 0.1 if row % 2 == 0 else 0.175
        for j in range(cnt):
            base[idx] = x_start + j * 0.165
            base[n+idx] = y
            idx += 1
    base[2*n:] = 0.095
    inits.append(base.copy())
    
    # Perturbed versions with different seeds
    for seed in [123, 456, 789, 101112, 202030]:
        np.random.seed(seed)
        pert = base.copy()
        pert[:2*n] += np.random.uniform(-0.04, 0.04, 2*n)
        pert[:2*n] = np.clip(pert[:2*n], 0.05, 0.95)
        inits.append(pert)
    
    # Grid-based initializations
    for spacing in [0.18, 0.2]:
        grid = np.zeros(3*n)
        idx = 0
        for row in range(6):
            for col in range(5 if row % 2 == 0 else 4):
                if idx >= n:
                    break
                grid[idx] = 0.1 + col * spacing + (0.04 if row % 2 else 0)
                grid[n+idx] = 0.1 + row * spacing * 0.9
                idx += 1
        grid[2*n:] = 0.085
        inits.append(grid)
    
    # Random initializations
    for _ in range(3):
        rand_init = np.zeros(3*n)
        rand_init[:n] = np.random.uniform(0.1, 0.9, n)
        rand_init[n:2*n] = np.random.uniform(0.1, 0.9, n)
        rand_init[2*n:] = 0.08
        inits.append(rand_init)
    
    return inits


def _run_slsqp(x0, n):
    """Execute SLSQP optimization from given starting point."""
    def objective(v):
        return -np.sum(v[2*n:])
    
    def in_square(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:]
        return np.concatenate([x-r, 1-x-r, y-r, 1-y-r])
    
    def no_overlap(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:]
        diffs = np.sqrt((x[:,None]-x)**2 + (y[:,None]-y)**2) - r[:,None] - r
        return diffs[np.triu_indices(n, 1)]
    
    bounds = [(0,1)]*2*n + [(1e-6, 0.5)]*n
    cons = [{'type':'ineq','fun':in_square}, {'type':'ineq','fun':no_overlap}]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter':1500, 'ftol':1e-10, 'disp': False})
        return res if res.success else None
    except:
        return None


def _extract_solution(v, n):
    """Extract centers and radii from solution vector."""
    return np.column_stack([v[:n], v[n:2*n]]), v[2*n:].copy()


def _ensure_validity(centers, radii):
    """Ensure all constraints are satisfied."""
    n = len(radii)
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
    
    for _ in range(100):
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i]-centers[j])
                if radii[i]+radii[j] > d + 1e-10:
                    s = d / (radii[i]+radii[j])
                    radii[i] *= s * 0.9999
                    radii[j] *= s * 0.9999
                    ok = False
        if ok:
            break
    return centers, radii


def _expand_radii(centers, radii):
    """Systematic contact-aware radius expansion with priority queue.
    
    Builds contact graph, uses priority queue for expansion order,
    computes exact max radius considering all constraints simultaneously,
    and performs final validity pass.
    """
    import heapq
    n = len(radii)
    
    # Precompute distance matrix for efficiency
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist[i, j] = dist[j, i] = np.linalg.norm(centers[i] - centers[j])
    
    # Build contact graph - identify close circles for faster updates
    contact_graph = [[j for j in range(n) if i != j and dist[i, j] < 0.35] for i in range(n)]
    
    prev_sum = np.sum(radii)
    for iteration in range(600):
        # Priority queue: circles with most expansion potential first
        pq = []
        for i in range(n):
            # Compute exact max radius: boundary constraints
            max_r = min(centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
            # Neighbor constraints - check all for exact calculation
            for j in range(n):
                if i != j:
                    max_r = min(max_r, dist[i, j] - radii[j])
            potential = max_r - radii[i]
            if potential > 1e-10:
                heapq.heappush(pq, (-potential, i, max_r))
        
        if not pq:
            break
        
        # Expand circle with highest potential, update affects others
        _, i, max_r = heapq.heappop(pq)
        radii[i] = max_r * 0.9999
        
        # Convergence check
        if iteration % 50 == 0:
            curr_sum = np.sum(radii)
            if abs(curr_sum - prev_sum) < 1e-12:
                break
            prev_sum = curr_sum
    
    # Final pass: adjust radii downward if constraints violated
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], centers[i, 1], 1 - centers[i, 0], 1 - centers[i, 1])
    
    for _ in range(50):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if radii[i] + radii[j] > dist[i, j] + 1e-10:
                    scale = dist[i, j] / (radii[i] + radii[j]) * 0.9999
                    radii[i] *= scale
                    radii[j] *= scale
                    ok = False
        if ok:
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
