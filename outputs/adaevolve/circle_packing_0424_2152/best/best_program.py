# EVOLVE-BLOCK-START
"""Basinhopping optimization for n=26 circle packing in unit square"""
import numpy as np
from scipy.optimize import basinhopping

def construct_packing():
    n = 26
    best_sum, best_c, best_r = 0, None, None
    for x0 in _inits(n):
        res = _bh(x0, n)
        if res is None: continue
        c, r = _extract(res.x, n)
        c, r = _valid(c, r)
        r = _expand(c, r)
        s = np.sum(r)
        if s > best_sum: best_sum, best_c, best_r = s, c.copy(), r.copy()
    return best_c, best_r, best_sum

def _inits(n):
    np.random.seed(42)
    base = np.zeros(3*n)
    rows = [5, 4, 5, 4, 5, 3]
    idx = 0
    for row, cnt in enumerate(rows):
        y = 0.08 + row * 0.15
        x_start = 0.1 if row % 2 == 0 else 0.175
        for j in range(cnt):
            base[idx], base[n+idx] = x_start + j * 0.165, y
            idx += 1
    base[2*n:] = 0.095
    inits = [base.copy()]
    for seed in [123, 456, 789, 101, 202]:
        np.random.seed(seed)
        p = base.copy()
        p[:2*n] = np.clip(p[:2*n] + np.random.uniform(-0.03, 0.03, 2*n), 0.05, 0.95)
        inits.append(p)
    return inits

def _bh(x0, n):
    def obj(v): return -np.sum(v[2*n:])
    def in_sq(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:]
        return np.concatenate([x-r, 1-x-r, y-r, 1-y-r])
    def no_ov(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:]
        return (np.sqrt((x[:,None]-x)**2 + (y[:,None]-y)**2) - r[:,None] - r)[np.triu_indices(n,1)]
    bounds = [(0,1)]*2*n + [(1e-6, 0.5)]*n
    cons = [{'type':'ineq','fun':in_sq}, {'type':'ineq','fun':no_ov}]
    kw = {'method':'SLSQP', 'bounds':bounds, 'constraints':cons, 'options':{'maxiter':500, 'ftol':1e-9}}
    try: return basinhopping(obj, x0, minimizer_kwargs=kw, niter=100, stepsize=0.08, T=0.5)
    except: return None

def _extract(v, n): return np.column_stack([v[:n], v[n:2*n]]), v[2*n:].copy()

def _valid(c, r):
    n = len(r)
    for i in range(n): r[i] = min(r[i], c[i,0], c[i,1], 1-c[i,0], 1-c[i,1])
    for _ in range(100):
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i]-c[j])
                if r[i]+r[j] > d + 1e-10:
                    s = d / (r[i]+r[j]) * 0.9999
                    r[i] *= s; r[j] *= s; ok = False
        if ok: break
    return c, r

def _expand(c, r):
    n = len(r)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n): dist[i, j] = dist[j, i] = np.linalg.norm(c[i] - c[j])
    for _ in range(500):
        changed = False
        for i in range(n):
            max_r = min(c[i,0], c[i,1], 1-c[i,0], 1-c[i,1])
            for j in range(n):
                if i != j: max_r = min(max_r, dist[i,j] - r[j])
            if max_r > r[i] + 1e-10: r[i] = max_r * 0.9999; changed = True
        if not changed: break
    for i in range(n): r[i] = min(r[i], c[i,0], c[i,1], 1-c[i,0], 1-c[i,1])
    for _ in range(50):
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                if r[i]+r[j] > dist[i,j] + 1e-10:
                    sc = dist[i,j] / (r[i]+r[j]) * 0.9999
                    r[i] *= sc; r[j] *= sc; ok = False
        if ok: break
    return r
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