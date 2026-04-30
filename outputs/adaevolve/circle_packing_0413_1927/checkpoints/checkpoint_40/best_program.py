# EVOLVE-BLOCK-START
"""Hexagonal lattice packing with spacing parameter optimization for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Hexagonal lattice: circles placed in staggered rows with offset.
    Single spacing parameter s determines all positions.
    Radii computed as max possible given wall and neighbor constraints.
    Grid search over s finds optimal configuration.
    """
    patterns = [[5,4,5,4,5,3], [4,5,4,5,4,4], [5,6,5,6,4], [4,4,5,4,4,5]]
    best_sum, best_c, best_r = 0, None, None
    
    for pat in patterns:
        for s in np.linspace(0.10, 0.25, 200):
            h = s * np.sqrt(3) / 2  # Vertical spacing for hexagonal
            y0 = (1 - (len(pat)-1) * h) / 2
            c = []
            for ri, cnt in enumerate(pat):
                y = y0 + ri * h
                x0 = (1 - (cnt-1) * s) / 2
                off = s/2 if ri % 2 else 0  # Offset alternate rows
                c.extend([[x0 + ci*s + off, y] for ci in range(cnt)])
            c = np.array(c)
            
            # Max radii from wall constraints
            r = np.array([min(x, y, 1-x, 1-y) for x, y in c])
            
            # Resolve overlaps iteratively
            for _ in range(50):
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        d = np.linalg.norm(c[i] - c[j])
                        if r[i] + r[j] > d > 1e-10:
                            sc = d / (r[i] + r[j])
                            r[i], r[j] = r[i]*sc, r[j]*sc
            
            total = np.sum(r)
            if total > best_sum:
                best_sum, best_c, best_r = total, c.copy(), r.copy()
    
    return best_c, best_r, best_sum
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