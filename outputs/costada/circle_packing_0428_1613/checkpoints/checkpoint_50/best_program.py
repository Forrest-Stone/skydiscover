# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using hexagonal pattern."""
import numpy as np

def construct_packing():
    """Pack 26 circles: 4 corners + 8 edge + 14 interior in hexagonal grid."""
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 corner circles (0-3): touch 2 walls
    rc = 0.095
    centers[0] = [rc, rc]
    centers[1] = [1-rc, rc]
    centers[2] = [rc, 1-rc]
    centers[3] = [1-rc, 1-rc]
    
    # 8 edge circles (4-11): touch 1 wall
    re = 0.07
    centers[4] = [0.28, re]
    centers[5] = [0.72, re]
    centers[6] = [0.28, 1-re]
    centers[7] = [0.72, 1-re]
    centers[8] = [re, 0.28]
    centers[9] = [re, 0.72]
    centers[10] = [1-re, 0.28]
    centers[11] = [1-re, 0.72]
    
    # 14 interior circles (12-25): hexagonal grid
    centers[12] = [0.18, 0.18]
    centers[13] = [0.50, 0.18]
    centers[14] = [0.82, 0.18]
    centers[15] = [0.34, 0.34]
    centers[16] = [0.66, 0.34]
    centers[17] = [0.18, 0.50]
    centers[18] = [0.50, 0.50]
    centers[19] = [0.82, 0.50]
    centers[20] = [0.34, 0.66]
    centers[21] = [0.66, 0.66]
    centers[22] = [0.18, 0.82]
    centers[23] = [0.50, 0.82]
    centers[24] = [0.82, 0.82]
    centers[25] = [0.50, 0.34]
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Compute max radii ensuring no overlap and within bounds."""
    n = len(centers)
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d > 0:
                    tot = radii[i] + radii[j]
                    radii[i] = radii[i] * d / tot
                    radii[j] = radii[j] * d / tot
                    changed = True
        if not changed: break
    return radii

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