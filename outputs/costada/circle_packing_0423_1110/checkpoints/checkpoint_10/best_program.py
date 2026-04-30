# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using hexagonal grid"""
import numpy as np

def construct_packing():
    """Hexagonal grid packing with corner optimization for n=26 circles."""
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # Corner circles (4) - can be larger as they touch two walls
    r_corner = 0.12
    corners = [(r_corner, r_corner), (1-r_corner, r_corner), 
               (r_corner, 1-r_corner), (1-r_corner, 1-r_corner)]
    for cx, cy in corners:
        centers[idx] = [cx, cy]
        idx += 1
    
    # Edge circles along each side (12 total)
    r_edge = 0.08
    # Bottom edge (excluding corners): 3 circles
    for i in range(3):
        centers[idx] = [0.25 + i*0.25, r_edge]
        idx += 1
    # Top edge: 3 circles
    for i in range(3):
        centers[idx] = [0.25 + i*0.25, 1-r_edge]
        idx += 1
    # Left edge: 3 circles
    for i in range(3):
        centers[idx] = [r_edge, 0.25 + i*0.25]
        idx += 1
    # Right edge: 3 circles
    for i in range(3):
        centers[idx] = [1-r_edge, 0.25 + i*0.25]
        idx += 1
    
    # Interior hexagonal packing (10 circles)
    r_int = 0.07
    dy = r_int * np.sqrt(3)
    # Row 1
    for i in range(4):
        centers[idx] = [0.2 + i*0.2, 0.3]
        idx += 1
    # Row 2 (offset)
    for i in range(3):
        centers[idx] = [0.3 + i*0.2, 0.5]
        idx += 1
    # Row 3
    for i in range(3):
        centers[idx] = [0.2 + i*0.2, 0.7]
        idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

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
