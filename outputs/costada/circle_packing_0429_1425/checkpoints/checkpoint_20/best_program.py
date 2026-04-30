# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Optimized packing for 26 circles using:
    - 4 corner circles (touch 2 walls, largest possible)
    - 12 edge circles (touch 1 wall each)
    - 10 interior circles (hexagonal pattern)
    
    Positions optimized to maximize sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # CORNER CIRCLES (indices 0-3): Position at cr=0.125 from edges
    # This gives radius ~0.125 when touching 2 walls
    cr = 0.125
    centers[0] = [cr, cr]
    centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]
    centers[3] = [1-cr, 1-cr]
    
    # EDGE CIRCLES (indices 4-15): 12 circles, 3 per edge
    # Position close to walls for larger radii
    e = 0.075  # distance from wall
    # Bottom edge (y=0)
    centers[4] = [0.32, e]
    centers[5] = [0.50, e]
    centers[6] = [0.68, e]
    # Top edge (y=1)
    centers[7] = [0.32, 1-e]
    centers[8] = [0.50, 1-e]
    centers[9] = [0.68, 1-e]
    # Left edge (x=0)
    centers[10] = [e, 0.32]
    centers[11] = [e, 0.50]
    centers[12] = [e, 0.68]
    # Right edge (x=1)
    centers[13] = [1-e, 0.32]
    centers[14] = [1-e, 0.50]
    centers[15] = [1-e, 0.68]
    
    # INTERIOR CIRCLES (indices 16-25): 10 circles in hexagonal pattern
    # Two offset rows for better packing density
    # Row 1: 4 circles at y=0.35
    centers[16] = [0.22, 0.35]
    centers[17] = [0.41, 0.35]
    centers[18] = [0.59, 0.35]
    centers[19] = [0.78, 0.35]
    # Row 2: 3 circles at y=0.50 (offset)
    centers[20] = [0.315, 0.50]
    centers[21] = [0.50, 0.50]
    centers[22] = [0.685, 0.50]
    # Row 3: 3 circles at y=0.65
    centers[23] = [0.22, 0.65]
    centers[24] = [0.41, 0.65]
    centers[25] = [0.78, 0.65]
    
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