# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized packing: 4 corner + 12 edge + 10 interior circles.
    Corner circles positioned to touch 2 walls each.
    Edge circles touch 1 wall, interior uses hexagonal pattern.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 CORNER CIRCLES - touch 2 walls, positioned for max radius
    cr = 0.103
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    # 12 EDGE CIRCLES - touch 1 wall each
    e = 0.072
    # Bottom edge
    centers[4] = [0.28, e]
    centers[5] = [0.50, e]
    centers[6] = [0.72, e]
    # Top edge
    centers[7] = [0.28, 1-e]
    centers[8] = [0.50, 1-e]
    centers[9] = [0.72, 1-e]
    # Left edge
    centers[10] = [e, 0.28]
    centers[11] = [e, 0.50]
    centers[12] = [e, 0.72]
    # Right edge
    centers[13] = [1-e, 0.28]
    centers[14] = [1-e, 0.50]
    centers[15] = [1-e, 0.72]
    
    # 10 INTERIOR CIRCLES - hexagonal pattern
    centers[16] = [0.22, 0.38]
    centers[17] = [0.42, 0.38]
    centers[18] = [0.58, 0.38]
    centers[19] = [0.78, 0.38]
    centers[20] = [0.32, 0.54]
    centers[21] = [0.50, 0.54]
    centers[22] = [0.68, 0.54]
    centers[23] = [0.22, 0.70]
    centers[24] = [0.42, 0.70]
    centers[25] = [0.78, 0.70]
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute max valid radii respecting boundaries and no overlap."""
    n = centers.shape[0]
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
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