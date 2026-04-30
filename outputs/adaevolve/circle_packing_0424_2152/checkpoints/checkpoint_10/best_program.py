# EVOLVE-BLOCK-START
"""Strategic circle packing exploiting corners and edges for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Strategic placement: 4 corners + 12 edges + 10 interior.
    Corners touch 2 edges, edges touch 1 edge, interior uses hexagonal.
    Optimized positions to maximize sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # 4 corner circles - positioned to maximize radius potential
    r_c = 0.103
    centers[idx] = [r_c, r_c]; idx += 1
    centers[idx] = [1-r_c, r_c]; idx += 1
    centers[idx] = [1-r_c, 1-r_c]; idx += 1
    centers[idx] = [r_c, 1-r_c]; idx += 1
    
    # 12 edge circles (3 per edge)
    r_e = 0.103
    # Bottom edge
    for x in [0.28, 0.5, 0.72]:
        centers[idx] = [x, r_e]; idx += 1
    # Right edge
    for y in [0.28, 0.5, 0.72]:
        centers[idx] = [1-r_e, y]; idx += 1
    # Top edge
    for x in [0.72, 0.5, 0.28]:
        centers[idx] = [x, 1-r_e]; idx += 1
    # Left edge
    for y in [0.72, 0.5, 0.28]:
        centers[idx] = [r_e, y]; idx += 1
    
    # 10 interior circles in optimized hexagonal pattern
    # Row 1: 4 circles
    y1 = 0.345
    for x in [0.22, 0.40, 0.60, 0.78]:
        centers[idx] = [x, y1]; idx += 1
    # Row 2: 3 circles (offset)
    y2 = 0.5
    for x in [0.31, 0.50, 0.69]:
        centers[idx] = [x, y2]; idx += 1
    # Row 3: 3 circles
    y3 = 0.655
    for x in [0.31, 0.50, 0.69]:
        centers[idx] = [x, y3]; idx += 1
    
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)


def compute_max_radii(centers):
    """Compute maximum valid radii via iterative constraint satisfaction."""
    n = centers.shape[0]
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for _ in range(200):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                total = radii[i] + radii[j]
                if total > dist > 1e-10:
                    scale = dist / total
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
