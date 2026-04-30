# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """
    Corner-first packing strategy for 26 circles in a unit square.
    Places 4 large corner circles (touch 2 walls), 12 edge circles 
    (touch 1 wall), and 10 interior circles in optimized positions.
    Corner circles use radius ≈ 0.1464 for optimal corner packing.
    """
    n = 26
    centers = np.zeros((n, 2))
    
    # Optimal corner radius: corner circle at (r,r) with radius r
    # Two adjacent corners: distance = 1-2r, so 2r ≤ 1-2r means r ≤ 0.207
    # But diagonal corners interact: distance = sqrt(2)*(1-2r), so 2r ≤ sqrt(2)*(1-2r)
    # This gives r ≤ sqrt(2)/(2+2*sqrt(2)) ≈ 0.207
    # Optimal when corner circles just touch: r = (2-sqrt(2))/2 ≈ 0.1464
    cr = 0.1464  # corner radius/position
    
    # 4 CORNER CIRCLES (indices 0-3) - touch 2 walls each
    centers[0] = [cr, cr]           # bottom-left
    centers[1] = [1-cr, cr]         # bottom-right
    centers[2] = [cr, 1-cr]         # top-left
    centers[3] = [1-cr, 1-cr]       # top-right
    
    # 12 EDGE CIRCLES (indices 4-15) - touch 1 wall each
    # Bottom edge (y=0): 3 circles between corners
    for i in range(3):
        centers[4+i] = [0.30 + i*0.20, 0.07]
    # Top edge (y=1): 3 circles
    for i in range(3):
        centers[7+i] = [0.30 + i*0.20, 0.93]
    # Left edge (x=0): 3 circles
    for i in range(3):
        centers[10+i] = [0.07, 0.30 + i*0.20]
    # Right edge (x=1): 3 circles
    for i in range(3):
        centers[13+i] = [0.93, 0.30 + i*0.20]
    
    # 10 INTERIOR CIRCLES (indices 16-25) - hexagonal pattern
    # Two offset rows for dense packing
    for i in range(4):
        centers[16+i] = [0.25 + i*0.17, 0.35]
    for i in range(3):
        centers[20+i] = [0.33 + i*0.17, 0.50]
    for i in range(3):
        centers[23+i] = [0.25 + i*0.17, 0.65]
    
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii


def compute_max_radii(centers, max_iterations=200):
    """
    Compute maximum valid radii using iterative constraint satisfaction.
    Handles overlaps proportionally and respects boundary constraints.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Initialize with distance to nearest boundary
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)
    
    # Iteratively resolve overlaps
    for _ in range(max_iterations):
        max_change = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                total = radii[i] + radii[j]
                if total > dist and total > 1e-10:
                    # Scale both radii proportionally to resolve overlap
                    scale = dist / total
                    change = max(radii[i] * (1 - scale), radii[j] * (1 - scale))
                    radii[i] *= scale
                    radii[j] *= scale
                    max_change = max(max_change, change)
        
        if max_change < 1e-10:
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
