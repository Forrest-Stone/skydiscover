# EVOLVE-BLOCK-START
"""Hexagonal circle packing for n=26 with optimized spacing"""
import numpy as np

def construct_packing():
    """
    Hexagonal packing pattern: alternating rows of 5 and 4 circles.
    Optimized spacing allows circles to expand to ~0.1 radius each.
    Pattern: 5-4-5-4-5-3 = 26 circles total.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal packing parameters
    spacing = 0.2  # Horizontal spacing between adjacent circles
    y_step = spacing * np.sqrt(3) / 2  # Vertical step for hexagonal packing
    
    idx = 0
    # Row 1: 5 circles at y=0.1
    for i in range(5):
        centers[idx] = [0.1 + i * spacing, 0.1]
        idx += 1
    # Row 2: 4 circles (offset) at y=0.1+y_step
    for i in range(4):
        centers[idx] = [0.2 + i * spacing, 0.1 + y_step]
        idx += 1
    # Row 3: 5 circles at y=0.1+2*y_step
    for i in range(5):
        centers[idx] = [0.1 + i * spacing, 0.1 + 2*y_step]
        idx += 1
    # Row 4: 4 circles (offset)
    for i in range(4):
        centers[idx] = [0.2 + i * spacing, 0.1 + 3*y_step]
        idx += 1
    # Row 5: 5 circles
    for i in range(5):
        centers[idx] = [0.1 + i * spacing, 0.1 + 4*y_step]
        idx += 1
    # Row 6: 3 circles at top
    for x in [0.1, 0.5, 0.9]:
        centers[idx] = [x, 0.95]
        idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum valid radii respecting wall and circle constraints."""
    n = len(centers)
    # Initialize with distance to nearest wall
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    # Iterate to resolve circle-circle constraints
    for _ in range(100):
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d and d > 1e-10:
                    total = radii[i] + radii[j]
                    radii[i] = d * radii[i] / total
                    radii[j] = d * radii[j] / total
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