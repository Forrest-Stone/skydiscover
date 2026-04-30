# EVOLVE-BLOCK-START
"""Optimized hexagonal circle packing for n=26 circles in a unit square."""
import numpy as np

def construct_packing():
    """Dense hexagonal packing pattern with 5-4-5-4-5-3 row structure."""
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # Hexagonal packing: rows of 5,4,5,4,5,3 circles
    # Optimized spacing for maximum radii
    r = 0.101  # Target radius
    dx = 2 * r  # Horizontal spacing
    dy = r * np.sqrt(3)  # Vertical spacing for hex pattern
    
    # Row 1: 5 circles (bottom)
    y1 = r
    for i in range(5):
        centers[idx] = [r + i * dx, y1]
        idx += 1
    
    # Row 2: 4 circles (offset by r)
    y2 = y1 + dy
    for i in range(4):
        centers[idx] = [2*r + i * dx, y2]
        idx += 1
    
    # Row 3: 5 circles
    y3 = y2 + dy
    for i in range(5):
        centers[idx] = [r + i * dx, y3]
        idx += 1
    
    # Row 4: 4 circles (offset)
    y4 = y3 + dy
    for i in range(4):
        centers[idx] = [2*r + i * dx, y4]
        idx += 1
    
    # Row 5: 5 circles
    y5 = y4 + dy
    for i in range(5):
        centers[idx] = [r + i * dx, y5]
        idx += 1
    
    # Row 6: 3 circles (top, centered)
    y6 = y5 + dy
    for i in range(3):
        centers[idx] = [2*r + i * dx, y6]
        idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum radii ensuring no overlap and containment in unit square."""
    n = centers.shape[0]
    radii = np.ones(n)
    
    # Limit by distance to borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iteratively resolve overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    
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
