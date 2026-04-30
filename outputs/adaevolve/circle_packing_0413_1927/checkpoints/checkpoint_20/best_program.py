# EVOLVE-BLOCK-START
"""Shell-based packing exploiting corner/edge advantages for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Shell-based packing: corners + edges + interior.
    Corners allow largest radii (2 boundary constraints).
    Edges allow medium radii (1 boundary constraint).
    Interior circles fill remaining space efficiently.
    """
    n = 26
    c = np.zeros((n, 2))
    
    # 4 CORNERS - positioned to maximize radius while leaving edge space
    # Distance from corner determines initial radius potential
    corner_d = 0.095
    c[0] = [corner_d, corner_d]
    c[1] = [1-corner_d, corner_d]
    c[2] = [corner_d, 1-corner_d]
    c[3] = [1-corner_d, 1-corner_d]
    
    # 12 EDGE CIRCLES - 3 per edge, between corners
    # Positioned closer to edge for larger radii
    edge_d = 0.055  # distance from edge
    
    # Bottom edge (y = edge_d)
    c[4] = [0.22, edge_d]
    c[5] = [0.50, edge_d]
    c[6] = [0.78, edge_d]
    
    # Top edge (y = 1 - edge_d)
    c[7] = [0.22, 1-edge_d]
    c[8] = [0.50, 1-edge_d]
    c[9] = [0.78, 1-edge_d]
    
    # Left edge (x = edge_d)
    c[10] = [edge_d, 0.22]
    c[11] = [edge_d, 0.50]
    c[12] = [edge_d, 0.78]
    
    # Right edge (x = 1 - edge_d)
    c[13] = [1-edge_d, 0.22]
    c[14] = [1-edge_d, 0.50]
    c[15] = [1-edge_d, 0.78]
    
    # 10 INTERIOR CIRCLES - hexagonal-inspired pattern
    # Row 1 (lower)
    c[16] = [0.22, 0.22]
    c[17] = [0.50, 0.19]
    c[18] = [0.78, 0.22]
    
    # Row 2 (middle-low) - offset
    c[19] = [0.35, 0.38]
    c[20] = [0.65, 0.38]
    
    # Row 3 (middle-high) - offset
    c[21] = [0.35, 0.62]
    c[22] = [0.65, 0.62]
    
    # Row 4 (upper)
    c[23] = [0.22, 0.78]
    c[24] = [0.50, 0.81]
    c[25] = [0.78, 0.78]
    
    # Compute max radii constrained by borders
    r = np.array([min(x, y, 1-x, 1-y) for x, y in c])
    
    # Iteratively reduce radii for non-overlapping pairs
    for _ in range(10):  # multiple passes for convergence
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
