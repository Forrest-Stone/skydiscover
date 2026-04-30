# EVOLVE-BLOCK-START
"""Hexagonal packing with wall-touching boundaries for n=26 circles"""
import numpy as np

def construct_packing():
    """
    Analytical hexagonal packing where:
    1. First/last rows touch walls (maximizing boundary circle radii)
    2. Row height = r * sqrt(3) for proper hexagonal geometry
    3. Analytical bounds guide the radius search
    4. Boundary circles expand to walls while interior circles constrained by neighbors
    """
    n = 26
    patterns = [[5,6,5,6,4], [4,5,4,5,4,4], [5,4,5,4,5,3], [6,5,6,5,4]]
    best_sum, best_c, best_r = 0, None, None
    
    for pat in patterns:
        rows, max_cnt = len(pat), max(pat)
        # Analytical bounds for equal circles
        r_h = 1.0 / (2 * max_cnt)  # Horizontal fit
        r_v = 1.0 / (2 + (rows-1) * np.sqrt(3))  # Vertical fit
        r_base = min(r_h, r_v)
        
        for mult in np.linspace(0.90, 1.25, 100):
            r_eq = r_base * mult
            h = r_eq * np.sqrt(3)  # Correct row height for hexagonal packing
            
            c = []
            for ri, cnt in enumerate(pat):
                # First row touches bottom wall, last row touches top wall
                if ri == 0:
                    y = r_eq
                elif ri == rows - 1:
                    y = 1 - r_eq
                else:
                    y = r_eq + ri * h
                
                # Horizontal positioning: max-count rows have edge circles touch walls
                if cnt == max_cnt:
                    x0 = r_eq
                    x_step = (1 - 2*r_eq) / max(cnt-1, 1)
                else:
                    x0 = 2*r_eq  # Offset for shorter rows
                    x_step = (1 - 4*r_eq) / max(cnt-1, 1)
                
                for ci in range(cnt):
                    x = x0 + ci * x_step
                    c.append([x, y])
            
            if len(c) != n: continue
            c = np.array(c)
            
            # Wall constraints
            r = np.minimum.reduce([c[:,0], c[:,1], 1-c[:,0], 1-c[:,1]])
            
            # Resolve overlaps iteratively
            for _ in range(50):
                for i in range(n):
                    max_r = min(c[i,0], c[i,1], 1-c[i,0], 1-c[i,1])
                    for j in range(n):
                        if i != j:
                            d = np.linalg.norm(c[i] - c[j])
                            max_r = min(max_r, max(0, d - r[j]))
                    r[i] = max_r
            
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