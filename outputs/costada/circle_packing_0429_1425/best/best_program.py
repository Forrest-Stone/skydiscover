# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized packing for 26 circles using:
    - 4 corner circles touching 2 walls each
    - 12 edge circles (3 per edge) touching 1 wall
    - 10 interior circles in staggered hexagonal pattern
    
    Strategy: Corner circles get larger radii (touch 2 walls),
    edge circles get medium radii (touch 1 wall),
    interior circles fill remaining space efficiently.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # CORNER CIRCLES (4): positioned to touch 2 walls
    cr = 0.12
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    # EDGE CIRCLES (12): 3 per edge, positioned to touch 1 wall
    e = 0.07
    # Bottom edge
    centers[4] = [0.30, e]
    centers[5] = [0.50, e]
    centers[6] = [0.70, e]
    # Top edge
    centers[7] = [0.30, 1-e]
    centers[8] = [0.50, 1-e]
    centers[9] = [0.70, 1-e]
    # Left edge
    centers[10] = [e, 0.30]
    centers[11] = [e, 0.50]
    centers[12] = [e, 0.70]
    # Right edge
    centers[13] = [1-e, 0.30]
    centers[14] = [1-e, 0.50]
    centers[15] = [1-e, 0.70]
    
    # INTERIOR CIRCLES (10): staggered hexagonal pattern
    # Row 1: 3 circles
    centers[16] = [0.25, 0.25]
    centers[17] = [0.50, 0.25]
    centers[18] = [0.75, 0.25]
    # Row 2: 4 circles (offset)
    centers[19] = [0.15, 0.50]
    centers[20] = [0.38, 0.50]
    centers[21] = [0.62, 0.50]
    centers[22] = [0.85, 0.50]
    # Row 3: 3 circles
    centers[23] = [0.25, 0.75]
    centers[24] = [0.50, 0.75]
    centers[25] = [0.75, 0.75]
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers, max_iter=500):
    """Compute maximum valid radii with iterative overlap resolution."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                total = radii[i] + radii[j]
                if total > dist and total > 1e-12:
                    scale = dist / total
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, y, 1-x, 1-y)
    
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
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")