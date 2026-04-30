# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles using radius expansion."""
import numpy as np

def construct_packing():
    """Hexagonal packing with optimized spacing and radius expansion."""
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # Optimized hexagonal pattern: 5-4-5-4-5-3 rows
    # Use spacing that allows maximum expansion
    r0 = 0.095  # Initial spacing parameter
    dx = 2.0 * r0
    dy = r0 * np.sqrt(3)
    
    # Row 1: 5 circles (bottom)
    y = r0
    for i in range(5):
        centers[idx] = [r0 + i * dx, y]
        idx += 1
    
    # Row 2: 4 circles (offset)
    y += dy
    for i in range(4):
        centers[idx] = [2*r0 + i * dx, y]
        idx += 1
    
    # Row 3: 5 circles
    y += dy
    for i in range(5):
        centers[idx] = [r0 + i * dx, y]
        idx += 1
    
    # Row 4: 4 circles (offset)
    y += dy
    for i in range(4):
        centers[idx] = [2*r0 + i * dx, y]
        idx += 1
    
    # Row 5: 5 circles
    y += dy
    for i in range(5):
        centers[idx] = [r0 + i * dx, y]
        idx += 1
    
    # Row 6: 3 circles (top, centered)
    y += dy
    for i in range(3):
        centers[idx] = [2*r0 + i * dx, y]
        idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum radii via iterative expansion from zero."""
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Iteratively expand each circle to its maximum feasible radius
    for _ in range(500):
        improved = False
        for i in range(n):
            x, y = centers[i]
            # Maximum radius from walls
            max_r = min(x, y, 1 - x, 1 - y)
            # Reduce by distances to other circles
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d - radii[j])
            max_r = max(0, max_r)
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        if not improved:
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
