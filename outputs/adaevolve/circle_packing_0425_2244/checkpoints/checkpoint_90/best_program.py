# EVOLVE-BLOCK-START
"""SLSQP with corner-anchored init and local search refinement for n=26 circle packing"""
import numpy as np
from scipy.optimize import minimize, linprog

def construct_packing():
    n = 26
    best_sum, best_pos, best_rad = 0, None, None
    for trial in range(12):
        pos = corner_init(n, trial)
        pos = optimize(pos, n)
        rad = solve_radii_lp(pos)
        pos, rad = local_refine(pos, rad)
        s = np.sum(rad)
        if s > best_sum:
            best_sum, best_pos, best_rad = s, pos.copy(), rad.copy()
    return best_pos, best_rad, best_sum

def corner_init(n, trial):
    pos = []
    rc = 0.095 + (trial % 4) * 0.006
    pos.extend([[rc,rc], [1-rc,rc], [rc,1-rc], [1-rc,1-rc]])
    re = 0.06 + (trial % 3) * 0.008
    ex = [0.2 + (trial%5)*0.015, 0.5, 0.8 - (trial%5)*0.015]
    for x in ex: pos.extend([[x,re], [x,1-re]])
    for y in ex: pos.extend([[re,y], [1-re,y]])
    ri = 0.085 + (trial % 5) * 0.004
    dy = ri * np.sqrt(3)
    for row, (cnt, yo) in enumerate([(3,-dy),(2,0),(3,dy),(2,2*dy)]):
        y = 0.5 + yo
        xs = 0.5 - (cnt-1)*ri if cnt > 2 else 0.5 - ri*0.5 + (row%2)*ri
        for c in range(cnt): pos.append([xs + c*2*ri, y])
    pos = np.array(pos[:n])
    if trial > 0:
        rng = np.random.RandomState(trial*37)
        pos = np.clip(pos + rng.randn(n,2)*0.015, 0.05, 0.95)
    return pos

def optimize(pos, n):
    def obj(x): return -np.sum(x[52:])
    def cons(x):
        p, r = x[:52].reshape(n,2), x[52:]
        c = [np.linalg.norm(p[i]-p[j])-r[i]-r[j] for i in range(n) for j in range(i+1,n)]
        for i in range(n): c.extend([p[i,0]-r[i], p[i,1]-r[i], 1-p[i,0]-r[i], 1-p[i,1]-r[i]])
        return np.array(c)
    x0 = np.concatenate([pos.flatten(), np.full(n, 0.09)])
    bnds = [(0.02,0.98)]*52 + [(0.01,0.48)]*26
    res = minimize(obj, x0, method='SLSQP', constraints={'type':'ineq','fun':cons},
                  bounds=bnds, options={'maxiter':500,'ftol':1e-10})
    return res.x[:52].reshape(n,2)

def local_refine(pos, rad, max_iter=12):
    """Local search to improve positions after global optimization"""
    pos = pos.copy()
    best_sum = np.sum(rad)
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for _ in range(max_iter):
        improved = False
        for i in range(len(pos)):
            step = 0.02
            for __ in range(5):
                for dx, dy in dirs:
                    new_pos = pos.copy()
                    new_pos[i] = np.clip([pos[i,0]+dx*step, pos[i,1]+dy*step], 0.03, 0.97)
                    new_rad = solve_radii_lp(new_pos)
                    if np.sum(new_rad) > best_sum + 1e-8:
                        pos, rad, best_sum, improved = new_pos, new_rad, np.sum(new_rad), True
                step *= 0.6
        if not improved: break
    return pos, rad

def solve_radii_lp(centers):
    n = len(centers)
    A, b = [], []
    for i in range(n):
        for j in range(i+1,n):
            row = np.zeros(n); row[i], row[j] = 1, 1
            A.append(row); b.append(np.linalg.norm(centers[i]-centers[j]))
    for i in range(n):
        row = np.zeros(n); row[i] = 1
        A.append(row); b.append(min(centers[i,0],centers[i,1],1-centers[i,0],1-centers[i,1]))
    res = linprog(-np.ones(n), A_ub=np.array(A), b_ub=np.array(b), bounds=[(0,None)]*n, method='highs')
    return res.x if res.success else np.zeros(n)


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
