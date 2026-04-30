# EVOLVE-BLOCK-START
"""Circle packing n=26: simulated annealing with corner-optimized patterns."""
import numpy as np

def construct_packing():
    """Simulated annealing with diverse initial patterns including corner-heavy designs."""
    n = 26
    best_sum, best_centers = 0, None
    
    # Pattern 1: Corner-heavy with hexagonal interior
    centers1 = build_corner_hex()
    # Pattern 2: Uniform hexagonal grid
    centers2 = build_hex_grid()
    # Pattern 3: Edge-dense pattern
    centers3 = build_edge_dense()
    # Pattern 4: Spiral-like placement
    centers4 = build_spiral()
    
    for centers in [centers1, centers2, centers3, centers4]:
        centers = simulated_annealing(centers)
        s = np.sum(compute_radii(centers))
        if s > best_sum:
            best_sum, best_centers = s, centers.copy()
    
    return best_centers, compute_radii(best_centers), best_sum

def build_corner_hex():
    """4 large corners + hexagonal interior packing."""
    centers = np.zeros((26, 2))
    r_corner = 0.12
    centers[0] = [r_corner, r_corner]
    centers[1] = [1-r_corner, r_corner]
    centers[2] = [r_corner, 1-r_corner]
    centers[3] = [1-r_corner, 1-r_corner]
    # Hexagonal interior: 5-6-6-5 pattern
    idx = 4
    rows = [(0.22, 5), (0.38, 6), (0.58, 6), (0.78, 5)]
    for y, cnt in rows:
        x0 = 0.5 - (cnt-1)*0.075/2
        for i in range(cnt):
            centers[idx] = [x0 + i*0.075, y]
            idx += 1
    return centers

def build_hex_grid():
    """Dense hexagonal grid with 5-6-5-6-4 rows."""
    centers = np.zeros((26, 2))
    idx = 0
    rows = [(0.10, 5), (0.28, 6), (0.46, 5), (0.64, 6), (0.82, 4)]
    for y, cnt in rows:
        x0 = 0.5 - (cnt-1)*0.085/2
        for i in range(cnt):
            centers[idx] = [x0 + i*0.085, y]
            idx += 1
    return centers

def build_edge_dense():
    """Circles along edges with sparse interior."""
    centers = np.zeros((26, 2))
    idx = 0
    # Bottom edge: 6 circles
    for i in range(6):
        centers[idx] = [0.08 + i*0.17, 0.08]
        idx += 1
    # Top edge: 6 circles
    for i in range(6):
        centers[idx] = [0.08 + i*0.17, 0.92]
        idx += 1
    # Left edge: 4 circles
    for i in range(4):
        centers[idx] = [0.08, 0.25 + i*0.17]
        idx += 1
    # Right edge: 4 circles
    for i in range(4):
        centers[idx] = [0.92, 0.25 + i*0.17]
        idx += 1
    # Interior: 6 circles
    int_pos = [(0.3,0.3), (0.5,0.3), (0.7,0.3), (0.4,0.5), (0.6,0.5), (0.5,0.7)]
    for x, y in int_pos:
        centers[idx] = [x, y]
        idx += 1
    return centers

def build_spiral():
    """Spiral-like placement from corners inward."""
    centers = np.zeros((26, 2))
    # 4 corners
    r = 0.11
    centers[0] = [r, r]
    centers[1] = [1-r, r]
    centers[2] = [r, 1-r]
    centers[3] = [1-r, 1-r]
    # Edge midpoints
    centers[4] = [0.5, 0.08]
    centers[5] = [0.5, 0.92]
    centers[6] = [0.08, 0.5]
    centers[7] = [0.92, 0.5]
    # Second ring
    for i, (x, y) in enumerate([(0.25,0.15), (0.75,0.15), (0.15,0.25), (0.85,0.25),
                                 (0.15,0.75), (0.85,0.75), (0.25,0.85), (0.75,0.85)]):
        centers[8+i] = [x, y]
    # Interior spiral
    int_pos = [(0.35,0.35), (0.65,0.35), (0.35,0.65), (0.65,0.65), (0.5,0.5),
               (0.5,0.35), (0.5,0.65), (0.35,0.5), (0.65,0.5)]
    for i, (x, y) in enumerate(int_pos):
        centers[16+i] = [x, y]
    return centers

def simulated_annealing(centers, max_iter=200):
    """SA with adaptive cooling and occasional random jumps."""
    n = len(centers)
    best = centers.copy()
    best_sum = np.sum(compute_radii(best))
    current = best.copy()
    current_sum = best_sum
    T = 0.02
    
    for iteration in range(max_iter):
        T = 0.02 * (1 - iteration/max_iter)**2
        for _ in range(n*3):
            i = np.random.randint(n)
            dx, dy = np.random.randn(2) * (0.03 + T)
            new = current.copy()
            new[i] = np.clip(new[i] + [dx, dy], 0.01, 0.99)
            new_sum = np.sum(compute_radii(new))
            if new_sum > current_sum or np.random.rand() < np.exp((new_sum-current_sum)/T):
                current, current_sum = new, new_sum
                if current_sum > best_sum:
                    best, best_sum = current.copy(), current_sum
    return best

def compute_radii(centers):
    """Compute max valid radii with iterative constraint resolution."""
    n = len(centers)
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])
    
    for _ in range(100):
        max_change = 0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    excess = radii[i] + radii[j] - d
                    radii[i] = max(1e-8, radii[i] - excess * radii[i]/(radii[i]+radii[j]))
                    radii[j] = max(1e-8, radii[j] - excess * radii[j]/(radii[i]+radii[j]))
                    max_change = max(max_change, excess)
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
