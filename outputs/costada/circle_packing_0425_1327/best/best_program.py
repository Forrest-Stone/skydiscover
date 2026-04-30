# EVOLVE-BLOCK-START
"""Optimized hexagonal circle packing for n=26 with staggered interior pattern."""
import numpy as np

def construct_packing():
    """4 corners + 8 edges + 14 interior in staggered hexagonal rows."""
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corners - well-positioned for large radii
    rc = 0.11
    centers[idx] = [rc, rc]; idx += 1
    centers[idx] = [1-rc, rc]; idx += 1
    centers[idx] = [rc, 1-rc]; idx += 1
    centers[idx] = [1-rc, 1-rc]; idx += 1
    
    # 8 edge circles (2 per side, positioned between corners)
    for x in [0.35, 0.65]: centers[idx] = [x, 0.06]; idx += 1
    for x in [0.35, 0.65]: centers[idx] = [x, 0.94]; idx += 1
    for y in [0.35, 0.65]: centers[idx] = [0.06, y]; idx += 1
    for y in [0.35, 0.65]: centers[idx] = [0.94, y]; idx += 1
    
    # 14 interior - hexagonal staggered pattern (rows of 5-4-5)
    y0, dy = 0.22, 0.17
    for i in range(5): centers[idx] = [0.15 + i*0.175, y0]; idx += 1
    for i in range(4): centers[idx] = [0.2375 + i*0.175, y0 + dy]; idx += 1
    for i in range(5): centers[idx] = [0.15 + i*0.175, y0 + 2*dy]; idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Iteratively expand circles to maximum respecting boundaries and overlaps."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for _ in range(500):
        for i in range(n):
            r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    r = min(r, np.linalg.norm(centers[i] - centers[j]) - radii[j])
            radii[i] = max(0, r)
    return radii


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
