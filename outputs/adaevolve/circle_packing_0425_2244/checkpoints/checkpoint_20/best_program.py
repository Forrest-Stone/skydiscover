# EVOLVE-BLOCK-START
"""Dense hexagonal packing for n=26 circles maximizing radii sum"""
import numpy as np


def construct_packing():
    """
    Construct dense hexagonal grid: 5-4-5-4-5-3 row pattern (26 circles).
    Alternating row offsets create optimal hexagonal packing density.
    Centers positioned to maximize boundary distance.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Hexagonal grid with alternating 5/4 circle rows
    # Vertical spacing: sqrt(3)/2 * horizontal for true hexagonal lattice
    r_base = 0.098  # Target radius for dense packing
    dx = 2.0 * r_base  # Horizontal center spacing
    dy = r_base * np.sqrt(3)  # Vertical spacing for hex lattice
    
    # Row pattern: (count, x_offset) - offset creates hexagonal stagger
    rows = [(5, 0.0), (4, r_base), (5, 0.0), (4, r_base), (5, 0.0), (3, r_base)]
    
    # Center the pattern in unit square
    total_height = 5 * dy + 2 * r_base
    y_start = (1.0 - total_height) / 2 + r_base
    
    idx = 0
    for row_idx, (count, x_off) in enumerate(rows):
        # Center each row horizontally
        row_width = 2 * r_base * (count - 1) + 2 * r_base if count > 1 else 2 * r_base
        x_start = (1.0 - row_width) / 2 + r_base + x_off
        y = y_start + row_idx * dy
        
        for col in range(count):
            x = x_start + col * dx
            centers[idx] = [x, y]
            idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """
    Iterative constraint satisfaction for maximum radii.
    Uses proportional reduction to fairly resolve overlaps.
    """
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for _ in range(500):
        max_change = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 1e-10:
                    continue
                total = radii[i] + radii[j]
                if total > dist:
                    excess = total - dist
                    if radii[i] > 1e-9 and radii[j] > 1e-9:
                        ratio = radii[i] / total
                        radii[i] = max(1e-9, radii[i] - excess * ratio)
                        radii[j] = max(1e-9, radii[j] - excess * (1 - ratio))
                        max_change = max(max_change, excess)
        if max_change < 1e-12:
            break
    
    return np.maximum(radii, 0)


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
