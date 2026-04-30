# EVOLVE-BLOCK-START
"""Compact circle packing for n=26: wall-focused placement + LP optimization."""
import numpy as np
from scipy.optimize import linprog, minimize

def lp_radii(c):
    """Compute optimal radii via linear programming."""
    n = len(c)
    if n == 0: return np.array([])
    bounds = [(0, max(1e-9, min(x, y, 1-x, 1-y))) for x, y in c]
    A, b = [], []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i] - c[j])
            row = np.zeros(n); row[i] = row[j] = 1
            A.append(row); b.append(d)
    if not A: return np.array([b[1] for b in bounds])
    res = linprog(-np.ones(n), A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method='highs')
    return res.x if res.success else np.zeros(n)

def init_config(cfg, n=26):
    """Generate initial configurations optimized for wall contact."""
    c = np.zeros((n, 2))
    if cfg == 'wall_opt':
        # Optimized wall-touching: corners + edges + hex interior
        rc = 0.092  # corner radius
        c[0], c[1], c[2], c[3] = [rc, rc], [1-rc, rc], [rc, 1-rc], [1-rc, 1-rc]
        re = 0.068  # edge radius
        for i, x in enumerate([0.22, 0.5, 0.78]):
            c[4+i] = [x, re]; c[7+i] = [x, 1-re]
            c[10+i] = [re, x]; c[13+i] = [1-re, x]
        # Interior hexagonal pattern with staggered rows
        y_vals = [0.22, 0.38, 0.5, 0.62, 0.78]
        x_offsets = [[0.22, 0.5, 0.78], [0.35, 0.65], [0.22, 0.5, 0.78], [0.35, 0.65], [0.22, 0.5, 0.78]]
        idx = 16
        for y, xs in zip(y_vals, x_offsets):
            for x in xs:
                if idx < n: c[idx] = [x, y]; idx += 1
    elif cfg == 'hex_dense':
        # Dense hexagonal: 5-4-5-4-5-3 pattern
        rows = [5, 4, 5, 4, 5, 3]
        idx = 0
        for ri, cnt in enumerate(rows):
            y = 0.08 + 0.84 * ri / (len(rows) - 1)
            dx = 0.84 / (cnt - 1) if cnt > 1 else 0
            x0 = 0.08 + (0.84 - dx * (cnt - 1)) / 2
            for ci in range(cnt):
                c[idx] = [x0 + dx * ci, y]; idx += 1
    elif cfg == 'corner_heavy':
        # Emphasize large corner circles
        rc = 0.11
        c[0], c[1], c[2], c[3] = [rc, rc], [1-rc, rc], [rc, 1-rc], [1-rc, 1-rc]
        # Remaining 22 in hexagonal grid avoiding corners
        idx = 4
        for row in range(5):
            cnt = 5 if row in [0, 2, 4] else 4
            y = 0.15 + 0.7 * row / 4
            for col in range(cnt):
                x = 0.15 + 0.7 * col / max(cnt-1, 1)
                if idx < n: c[idx] = [x, y]; idx += 1
    return c

def construct_packing():
    """Multi-start optimization with wall-focused initializations."""
    n = 26
    best_sum, best_c, best_r = 0, None, None
    
    for cfg in ['wall_opt', 'hex_dense', 'corner_heavy']:
        c0 = init_config(cfg, n)
        # Quick Powell optimization
        res = minimize(lambda p: -np.sum(lp_radii(p.reshape(-1, 2))), 
                      c0.flatten(), method='Powell', options={'maxiter': 300})
        c = np.clip(res.x.reshape(-1, 2), 0.005, 0.995)
        r = lp_radii(c)
        total = np.sum(r)
        if total > best_sum:
            best_sum, best_c, best_r = total, c, r
    return best_c, best_r, best_sum

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