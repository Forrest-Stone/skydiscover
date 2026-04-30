# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """Corner-edge-interior pattern exploiting that corners allow larger radii."""
    n = 26
    c = np.zeros((n, 2))
    
    # 4 corner circles - positioned to allow larger radii
    corner_r = 0.1
    c[0] = [corner_r, corner_r]
    c[1] = [1-corner_r, corner_r]
    c[2] = [corner_r, 1-corner_r]
    c[3] = [1-corner_r, 1-corner_r]
    
    # 12 edge circles (3 per edge, avoiding corners)
    # Bottom edge (y ~ 0.06)
    c[4] = [0.22, 0.06]
    c[5] = [0.5, 0.05]
    c[6] = [0.78, 0.06]
    # Top edge (y ~ 0.94)
    c[7] = [0.22, 0.94]
    c[8] = [0.5, 0.95]
    c[9] = [0.78, 0.94]
    # Left edge (x ~ 0.06)
    c[10] = [0.06, 0.22]
    c[11] = [0.05, 0.5]
    c[12] = [0.06, 0.78]
    # Right edge (x ~ 0.94)
    c[13] = [0.94, 0.22]
    c[14] = [0.95, 0.5]
    c[15] = [0.94, 0.78]
    
    # 10 interior circles in hexagonal-like pattern
    # Row 1 (y ~ 0.25)
    c[16] = [0.25, 0.25]
    c[17] = [0.5, 0.22]
    c[18] = [0.75, 0.25]
    # Row 2 (y ~ 0.42) - offset
    c[19] = [0.35, 0.42]
    c[20] = [0.65, 0.42]
    # Row 3 (y ~ 0.58) - offset
    c[21] = [0.35, 0.58]
    c[22] = [0.65, 0.58]
    # Row 4 (y ~ 0.75)
    c[23] = [0.25, 0.75]
    c[24] = [0.5, 0.78]
    c[25] = [0.75, 0.75]
    
    # Compute max radii constrained by borders
    r = np.array([min(x, y, 1-x, 1-y) for x, y in c])
    
    # Reduce radii for non-overlapping pairs
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i] - c[j])
            if r[i] + r[j] > d:
                sc = d / (r[i] + r[j])
                r[i], r[j] = r[i]*sc, r[j]*sc
    return c, r, np.sum(r)
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
