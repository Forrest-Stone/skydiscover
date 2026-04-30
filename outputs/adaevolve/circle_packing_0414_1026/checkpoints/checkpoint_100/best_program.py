# EVOLVE-BLOCK-START
"""Systematic multi-pattern hexagonal packing for n=26 circles"""
import numpy as np

def construct_packing():
    """Try patterns systematically with both offset strategies, select best."""
    patterns = [
        [5,4,5,4,5,3],[4,5,4,5,4,4],[5,5,5,5,4,2],[6,4,5,4,5,2],
        [3,5,5,5,5,3],[4,4,5,5,4,4],[4,5,5,5,5,2],[5,4,4,5,4,4],
        [4,4,4,5,5,4],[3,4,5,5,5,4],[6,5,5,5,5],[5,5,5,5,6],
        [5,6,5,5,5],[5,5,6,5,5],[4,5,6,5,4,2],[4,4,4,4,4,4,2],
        [3,4,4,4,4,4,3],[7,6,7,6],[6,7,6,7],[4,6,6,6,4],
        [5,6,5,6,4],[5,5,6,5,5],[4,6,6,5,5],[6,6,7,7],
        [5,7,7,7],[7,7,6,6],[4,4,6,6,6],[5,5,5,6,5],
        [3,5,6,6,6],[4,5,5,6,6],[6,5,5,5,5],[5,6,4,5,6],
        [7,6,6,7],[6,6,5,4,5],[5,4,5,6,6],[4,5,4,6,7],
    ]
    best_sum, best_c, best_r = 0, None, None
    for p in patterns:
        for off in [True, False]:
            c = build_pattern(p, off)
            r = compute_radii(c)
            s = np.sum(r)
            if s > best_sum:
                best_sum, best_c, best_r = s, c, r
    return best_c, best_r, best_sum

def build_pattern(rows, offset=True):
    """Build hexagonal lattice positions for given row pattern."""
    n = sum(rows)
    centers = np.zeros((n, 2))
    r = min(1/(2*max(rows)), 1/((len(rows)-1)*np.sqrt(3)+2)) if len(rows)>1 else 0.25
    idx = 0
    for row, cnt in enumerate(rows):
        y = r + row * r * np.sqrt(3)
        x_start = r + (1 - 2*r - (cnt-1)*2*r) / 2
        if offset and row % 2 == 1:
            x_start += r
        for c in range(cnt):
            centers[idx] = [x_start + c * 2 * r, y]
            idx += 1
    return centers

def compute_radii(centers):
    """Expand radii iteratively with more iterations for better convergence."""
    n = len(centers)
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    for _ in range(300):
        for i in range(n):
            r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    r = min(r, np.linalg.norm(centers[i]-centers[j]) - radii[j])
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
