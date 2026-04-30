# EVOLVE-BLOCK-START
"""Circle packing for n=26 using multi-start optimization with diverse patterns"""
import numpy as np


def construct_packing():
    """
    Multi-start optimization with 5 diverse geometric patterns.
    Uses 20 trials for thorough exploration, aggressive hill-climbing
    with 12 directions, fine-tuning, and radius inflation pass.
    """
    n = 26
    np.random.seed(42)
    best_centers = None
    best_sum = 0
    
    for trial in range(20):
        centers = create_pattern(n, trial)
        centers = hill_climb(centers)
        centers = fine_tune(centers)
        centers = inflate_radii(centers)
        radii = compute_max_radii(centers)
        total = np.sum(radii)
        if total > best_sum:
            best_sum = total
            best_centers = centers.copy()
    
    # Final polish with extra fine-tuning and inflation
    best_centers = fine_tune(best_centers)
    best_centers = inflate_radii(best_centers)
    radii = compute_max_radii(best_centers)
    return best_centers, radii, np.sum(radii)


def inflate_radii(centers):
    """Greedy radius inflation: try to grow each circle while maintaining constraints."""
    n = centers.shape[0]
    radii = compute_max_radii(centers)
    border_dist = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    for iteration in range(100):
        improved = False
        for i in np.argsort(-radii):  # Start with largest circles
            # Try to inflate this circle
            max_r = border_dist[i]
            for j in range(n):
                if j != i:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
        if not improved:
            break
    return centers


def create_pattern(n, trial):
    """Create diverse initial configurations: 5 pattern types with variations."""
    centers = np.zeros((n, 2))
    pattern_type = trial % 5
    np.random.seed(42 + trial * 7)
    
    # 4 corners - optimal placement for largest circles
    rc = 0.11 + 0.008 * (trial % 6)
    centers[0] = [rc, rc]
    centers[1] = [1 - rc, rc]
    centers[2] = [rc, 1 - rc]
    centers[3] = [1 - rc, 1 - rc]
    
    if pattern_type == 0:
        # Corner-shell with hexagonal interior
        e = 0.07 + 0.008 * ((trial // 5) % 3)
        for i, (x, y) in enumerate([(0.30, e), (0.70, e), (0.30, 1-e), (0.70, 1-e),
                                    (e, 0.30), (e, 0.70), (1-e, 0.30), (1-e, 0.70)]):
            centers[4 + i] = [x, y]
        idx = 12
        for cnt, y in zip([5, 4, 5], [0.25, 0.50, 0.75]):
            xs = [0.18, 0.34, 0.50, 0.66, 0.82] if cnt == 5 else [0.26, 0.42, 0.58, 0.74]
            for x in xs:
                centers[idx] = [x, y]
                idx += 1
    
    elif pattern_type == 1:
        # Dense edge packing for larger radii
        e = 0.065
        for i, (x, y) in enumerate([(0.25, e), (0.50, e), (0.75, e), (0.25, 1-e),
                                    (0.50, 1-e), (0.75, 1-e), (e, 0.50), (1-e, 0.50)]):
            centers[4 + i] = [x, y]
        idx = 12
        for cnt, y in zip([4, 3, 4, 3], [0.22, 0.42, 0.58, 0.78]):
            xs = np.linspace(0.20, 0.80, cnt)
            for x in xs:
                centers[idx] = [x, y]
                idx += 1
    
    elif pattern_type == 2:
        # Staggered row pattern
        e = 0.07
        for i, (x, y) in enumerate([(0.28, e), (0.72, e), (0.28, 1-e), (0.72, 1-e),
                                    (e, 0.28), (e, 0.72), (1-e, 0.28), (1-e, 0.72)]):
            centers[4 + i] = [x, y]
        idx = 12
        for row, y in enumerate([0.23, 0.39, 0.61, 0.77]):
            cnt = 4 if row % 2 == 0 else 3
            offset = 0 if row % 2 == 0 else 0.12
            xs = [0.18 + offset + j * 0.22 for j in range(cnt)]
            for x in xs:
                centers[idx] = [x, y]
                idx += 1
    
    elif pattern_type == 3:
        # Concentric rings around center
        for i in range(8):
            angle = np.pi/8 + i * np.pi/4
            r = 0.32
            centers[4 + i] = [0.5 + r*np.cos(angle), 0.5 + r*np.sin(angle)]
        idx = 12
        for i in range(6):
            angle = i * np.pi/3
            r = 0.18
            centers[idx] = [0.5 + r*np.cos(angle), 0.5 + r*np.sin(angle)]
            idx += 1
        for i in range(8):
            angle = i * np.pi/4
            r = 0.25
            centers[idx] = [0.5 + r*np.cos(angle), 0.5 + r*np.sin(angle)]
            idx += 1
    
    else:
        # Compact grid with perturbations
        e = 0.07 + 0.004 * trial
        for i, (x, y) in enumerate([(0.30, e), (0.70, e), (0.30, 1-e), (0.70, 1-e),
                                    (e, 0.30), (e, 0.70), (1-e, 0.30), (1-e, 0.70)]):
            centers[4 + i] = [x, y]
        idx = 12
        positions = [(0.22, 0.22), (0.50, 0.22), (0.78, 0.22),
                     (0.36, 0.38), (0.64, 0.38),
                     (0.22, 0.50), (0.50, 0.50), (0.78, 0.50),
                     (0.36, 0.62), (0.64, 0.62),
                     (0.22, 0.78), (0.50, 0.78), (0.78, 0.78)]
        for x, y in positions:
            centers[idx] = [x, y]
            idx += 1
    
    centers += np.random.uniform(-0.012, 0.012, (n, 2))
    return np.clip(centers, 0.02, 0.98)


def hill_climb(centers):
    """Multi-scale hill-climbing with 12 directions per step."""
    n = centers.shape[0]
    best = centers.copy()
    best_sum = np.sum(compute_max_radii(best))
    
    for step in [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4,
                          np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8]
                for angle in angles:
                    dx, dy = step * np.cos(angle), step * np.sin(angle)
                    new_c = best.copy()
                    new_c[i] = [np.clip(best[i, 0] + dx, 0.02, 0.98),
                               np.clip(best[i, 1] + dy, 0.02, 0.98)]
                    new_sum = np.sum(compute_max_radii(new_c))
                    if new_sum > best_sum + 1e-9:
                        best = new_c
                        best_sum = new_sum
                        improved = True
    return best


def fine_tune(centers):
    """Fine-grained optimization with very small steps and multiple passes."""
    n = centers.shape[0]
    best = centers.copy()
    best_sum = np.sum(compute_max_radii(best))
    
    for step in [0.0008, 0.0004, 0.0002, 0.0001]:
        for _ in range(3):
            improved = False
            for i in range(n):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = best.copy()
                        new_c[i] = [np.clip(best[i, 0] + dx, 0.02, 0.98),
                                   np.clip(best[i, 1] + dy, 0.02, 0.98)]
                        new_sum = np.sum(compute_max_radii(new_c))
                        if new_sum > best_sum + 1e-10:
                            best = new_c
                            best_sum = new_sum
                            improved = True
            if not improved:
                break
    return best


def compute_max_radii(centers):
    """Shrink-grow iteration for maximum radii with improved convergence."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dists[i, j] = dists[j, i] = np.linalg.norm(centers[i] - centers[j])
    
    for _ in range(600):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if radii[i] + radii[j] > dists[i, j] + 1e-12:
                    scale = dists[i, j] / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        for i in range(n):
            border = min(centers[i, 0], centers[i, 1], 1-centers[i, 0], 1-centers[i, 1])
            limits = [dists[i, j] - radii[j] for j in range(n) if j != i]
            max_r = min(border, min(limits)) if limits else border
            if max_r > radii[i] + 1e-12:
                radii[i] = max_r
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
