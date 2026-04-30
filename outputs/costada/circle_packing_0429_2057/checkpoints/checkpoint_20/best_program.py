# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using hexagonal grid"""
import numpy as np

def construct_packing():
    """
    Hexagonal grid packing with corner optimization for 26 circles.
    Uses 4 large corner circles + hexagonal interior pattern.
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 large corner circles - key optimization
    r_corner = 0.1464  # Optimal corner radius
    centers[idx] = [r_corner, r_corner]; idx += 1
    centers[idx] = [1-r_corner, r_corner]; idx += 1
    centers[idx] = [r_corner, 1-r_corner]; idx += 1
    centers[idx] = [1-r_corner, 1-r_corner]; idx += 1
    
    # Edge circles (4 per edge, avoiding corners)
    r_edge = 0.08
    for i in range(4):
        t = 0.22 + i * 0.19
        centers[idx] = [t, r_edge]; idx += 1  # bottom
        centers[idx] = [t, 1-r_edge]; idx += 1  # top
        centers[idx] = [r_edge, t]; idx += 1  # left
        centers[idx] = [1-r_edge, t]; idx += 1  # right
    
    # Interior hexagonal grid (6 circles)
    h = 0.5 + 0.12 * np.sqrt(3)
    for row in range(2):
        y = 0.35 + row * 0.30
        for col in range(3):
            x = 0.30 + col * 0.20 + (row * 0.10)
            centers[idx] = [x, y]; idx += 1
    
    centers = np.clip(centers, 0.02, 0.98)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    """Compute maximum valid radii respecting boundaries and non-overlap."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d:
                s = d / (radii[i] + radii[j])
                radii[i] *= s
                radii[j] *= s
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
