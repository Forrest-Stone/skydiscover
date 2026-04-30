# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized packing for 26 circles with improved spacing.
    4 corners + 12 edges + 10 interior with tuned positions
    to minimize overlap constraints and maximize radii.
    Uses larger corner circles and spread-out interior pattern.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # 4 CORNER CIRCLES - larger for more wall contact
    cr = 0.12
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    # 12 EDGE CIRCLES - positioned to avoid corner overlap
    e = 0.075
    centers[4] = [0.33, e]
    centers[5] = [0.50, e]
    centers[6] = [0.67, e]
    centers[7] = [0.33, 1-e]
    centers[8] = [0.50, 1-e]
    centers[9] = [0.67, 1-e]
    centers[10] = [e, 0.33]
    centers[11] = [e, 0.50]
    centers[12] = [e, 0.67]
    centers[13] = [1-e, 0.33]
    centers[14] = [1-e, 0.50]
    centers[15] = [1-e, 0.67]
    
    # 10 INTERIOR CIRCLES - spread hexagonal pattern
    centers[16] = [0.23, 0.23]
    centers[17] = [0.43, 0.23]
    centers[18] = [0.57, 0.23]
    centers[19] = [0.77, 0.23]
    centers[20] = [0.33, 0.43]
    centers[21] = [0.50, 0.43]
    centers[22] = [0.67, 0.43]
    centers[23] = [0.23, 0.63]
    centers[24] = [0.50, 0.63]
    centers[25] = [0.77, 0.63]
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers, max_iter=500):
    """Compute max valid radii with iterative overlap resolution."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1-x, 1-y)
    
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