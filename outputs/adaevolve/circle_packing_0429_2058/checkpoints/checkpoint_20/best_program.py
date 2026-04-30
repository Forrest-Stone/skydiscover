# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 using corner-edge-interior pattern with shrink-expand optimization."""
import numpy as np


def construct_packing():
    """
    Layout: 4 corners (touch 2 walls) + 8 edges (touch 1 wall) + 14 interior = 26.
    Wall-touching circles can be larger since they don't compete for space outside.
    Uses alternating shrink-expand phases for optimal radii computation.
    """
    n = 26
    c = np.zeros((n, 2))
    i = 0
    
    # 4 corners - positioned to touch both adjacent walls (highest value)
    cr = 0.11
    c[i] = [cr, cr]; i += 1
    c[i] = [1-cr, cr]; i += 1
    c[i] = [cr, 1-cr]; i += 1
    c[i] = [1-cr, 1-cr]; i += 1
    
    # 8 edge circles (2 per side) - positioned away from corners
    er = 0.07
    c[i] = [0.35, er]; i += 1
    c[i] = [0.65, er]; i += 1
    c[i] = [0.35, 1-er]; i += 1
    c[i] = [0.65, 1-er]; i += 1
    c[i] = [er, 0.35]; i += 1
    c[i] = [er, 0.65]; i += 1
    c[i] = [1-er, 0.35]; i += 1
    c[i] = [1-er, 0.65]; i += 1
    
    # 14 interior circles in 3-4-4-3 hexagonal pattern
    # Row 1: 3 circles
    c[i] = [0.25, 0.23]; i += 1
    c[i] = [0.50, 0.23]; i += 1
    c[i] = [0.75, 0.23]; i += 1
    # Row 2: 4 circles (staggered)
    c[i] = [0.175, 0.42]; i += 1
    c[i] = [0.39, 0.42]; i += 1
    c[i] = [0.61, 0.42]; i += 1
    c[i] = [0.825, 0.42]; i += 1
    # Row 3: 4 circles (staggered)
    c[i] = [0.175, 0.58]; i += 1
    c[i] = [0.39, 0.58]; i += 1
    c[i] = [0.61, 0.58]; i += 1
    c[i] = [0.825, 0.58]; i += 1
    # Row 4: 3 circles
    c[i] = [0.25, 0.77]; i += 1
    c[i] = [0.50, 0.77]; i += 1
    c[i] = [0.75, 0.77]; i += 1
    
    r = compute_max_radii(c)
    return c, r, np.sum(r)


def compute_max_radii(c):
    """
    Alternating shrink-expand optimization for maximum sum of radii.
    Phase 1: Shrink overlapping circles proportionally.
    Phase 2: Expand each circle to fill available space.
    """
    n = len(c)
    r = np.array([min(p[0], p[1], 1-p[0], 1-p[1]) for p in c])
    
    # Pre-compute pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d[i, j] = d[j, i] = np.linalg.norm(c[i] - c[j])
    
    for _ in range(200):
        # Shrink phase: resolve all overlaps
        for __ in range(50):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if r[i] + r[j] > d[i, j] * 0.9999:
                        t = r[i] + r[j]
                        if t > 1e-10:
                            r[i] = d[i, j] * r[i] / t * 0.9999
                            r[j] = d[i, j] * r[j] / t * 0.9999
                            changed = True
            if not changed: break
        
        # Expand phase: grow circles where space allows
        for i in range(n):
            w = min(c[i][0], c[i][1], 1-c[i][0], 1-c[i][1])
            lim = w
            for j in range(n):
                if i != j: lim = min(lim, (d[i, j] - r[j]) * 0.9999)
            r[i] = min(w, max(r[i], lim))
    
    # Final constraint enforcement
    for i in range(n):
        r[i] = min(r[i], c[i][0], c[i][1], 1-c[i][0], 1-c[i][1]) * 0.9999
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