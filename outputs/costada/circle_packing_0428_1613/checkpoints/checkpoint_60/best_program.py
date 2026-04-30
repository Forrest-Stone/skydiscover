# EVOLVE-BLOCK-START
"""Circle packing for n=26 using LP radii + multi-start position optimization."""
import numpy as np
from scipy.optimize import linprog, minimize

def compute_optimal_radii_lp(centers):
    """Compute optimal radii via LP: max sum(r) s.t. r_i<=wall_dist, r_i+r_j<=pair_dist."""
    n = len(centers)
    if n == 0: return np.array([])
    
    c = -np.ones(n)
    bounds = [(0, max(1e-8, min(x, y, 1-x, 1-y))) for x, y in centers]
    
    A_ub, b_ub = [], []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
    
    if not A_ub: return np.array([b[1] for b in bounds])
    
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    return res.x if res.success else np.zeros(n)

def create_initial_config(config_type, n=26):
    """Create different initial configurations for optimization."""
    centers = np.zeros((n, 2))
    
    if config_type == 'hex_rows':
        # Pattern: 5-4-5-4-5-3 = 26
        rows = [5, 4, 5, 4, 5, 3]
        dy = 0.9 / len(rows)
        idx = 0
        for ri, cnt in enumerate(rows):
            y = 0.05 + dy * (ri + 0.5)
            dx = 0.9 / cnt
            for ci in range(cnt):
                centers[idx] = [0.05 + dx * (ci + 0.5), y]
                idx += 1
    
    elif config_type == 'wall_focused':
        # 4 corners + 8 edges + 14 interior
        rc = 0.09
        centers[0] = [rc, rc]
        centers[1] = [1-rc, rc]
        centers[2] = [rc, 1-rc]
        centers[3] = [1-rc, 1-rc]
        
        re = 0.065
        centers[4] = [0.25, re]
        centers[5] = [0.5, re]
        centers[6] = [0.75, re]
        centers[7] = [0.25, 1-re]
        centers[8] = [0.5, 1-re]
        centers[9] = [0.75, 1-re]
        centers[10] = [re, 0.25]
        centers[11] = [re, 0.5]
        centers[12] = [re, 0.75]
        centers[13] = [1-re, 0.25]
        centers[14] = [1-re, 0.5]
        centers[15] = [1-re, 0.75]
        
        # Interior hexagonal
        interior = [[0.2,0.2], [0.5,0.2], [0.8,0.2],
                     [0.35,0.35], [0.65,0.35],
                     [0.2,0.5], [0.5,0.5], [0.8,0.5],
                     [0.35,0.65], [0.65,0.65]]
        for i, pos in enumerate(interior):
            centers[16+i] = pos
    
    elif config_type == 'symmetric':
        # Symmetric 5x5 grid minus corners + 1 center
        idx = 0
        for i in range(5):
            for j in range(5):
                x, y = 0.1 + i*0.2, 0.1 + j*0.2
                if not ((i==0 or i==4) and (j==0 or j==4)):  # skip corners
                    centers[idx] = [x, y]
                    idx += 1
        # Add center circle
        centers[25] = [0.5, 0.5]
    
    return centers

def construct_packing():
    """Optimize 26 circles using multi-start LP + position optimization."""
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    def obj(pos):
        return -np.sum(compute_optimal_radii_lp(pos.reshape(-1, 2)))
    
    configs = ['hex_rows', 'wall_focused', 'symmetric']
    
    for cfg in configs:
        init = create_initial_config(cfg, n)
        
        res = minimize(obj, init.flatten(), method='Powell', 
                      options={'maxiter': 500, 'ftol': 1e-8})
        
        centers = np.clip(res.x.reshape(-1, 2), 0.01, 0.99)
        radii = compute_optimal_radii_lp(centers)
        total = np.sum(radii)
        
        if total > best_sum:
            best_sum = total
            best_centers = centers
            best_radii = radii
    
    return best_centers, best_radii, best_sum

# EVOLVE-BLOCK-END

def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.5))
        ax.text(c[0], c[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")