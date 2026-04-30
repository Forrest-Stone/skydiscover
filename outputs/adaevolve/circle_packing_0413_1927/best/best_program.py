# EVOLVE-BLOCK-START
"""Extended hexagonal packing search for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Enhanced hexagonal packing with:
    1. Extended pattern set (10 patterns vs 4)
    2. Wider multiplier range (0.80-1.45 vs 0.90-1.25)
    3. More convergence iterations (100 vs 50)
    4. Finer granularity search (150 vs 100 steps)
    """
    n = 26
    patterns = [[5,6,5,6,4], [4,5,4,5,4,4], [5,4,5,4,5,3], [6,5,6,5,4],
                [4,5,5,5,4,3], [3,5,5,5,5,3], [4,4,5,5,4,4], [5,5,5,5,3,3],
                [4,6,6,6,4], [3,4,5,6,5,3]]
    best = (0, None, None)
    
    for pat in patterns:
        rows, mx = len(pat), max(pat)
        r_base = min(1.0/(2*mx), 1.0/(2 + (rows-1)*np.sqrt(3)))
        
        for m in np.linspace(0.80, 1.45, 150):
            r_eq, h = r_base * m, r_base * m * np.sqrt(3)
            c = []
            for ri, cnt in enumerate(pat):
                y = r_eq if ri == 0 else (1-r_eq if ri == rows-1 else r_eq + ri*h)
                x0 = r_eq if cnt == mx else 2*r_eq
                x_step = (1 - 2*x0) / max(cnt-1, 1)
                c.extend([[x0 + ci*x_step, y] for ci in range(cnt)])
            
            if len(c) != n: continue
            c = np.array(c)
            r = np.minimum.reduce([c[:,0], c[:,1], 1-c[:,0], 1-c[:,1]])
            
            for _ in range(100):
                for i in range(n):
                    mr = min(c[i,0], c[i,1], 1-c[i,0], 1-c[i,1])
                    for j in range(n):
                        if i != j: mr = min(mr, max(0, np.linalg.norm(c[i]-c[j]) - r[j]))
                    r[i] = mr
            
            if np.sum(r) > best[0]:
                best = (np.sum(r), c.copy(), r.copy())
    
    return best[1], best[2], best[0]
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