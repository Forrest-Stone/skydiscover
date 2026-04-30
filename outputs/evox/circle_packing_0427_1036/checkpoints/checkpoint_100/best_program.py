# EVOLVE-BLOCK-START
"""Multi-start SLSQP with greedy radius expansion for n=26 circle packing"""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    """
    Enhanced multi-start optimization with position refinement.
    Uses diverse initial patterns, SLSQP optimization, then
    alternates position refinement with radius expansion.
    """
    n = 26
    best_sum, best_centers, best_radii = 0, None, None
    
    for x0 in _generate_initial_guesses():
        # Phase 1: SLSQP optimization
        result = minimize(
            lambda v: -np.sum(v[2::3]), x0, method='SLSQP',
            bounds=[(0,1), (0,1), (1e-6, 0.5)] * n,
            constraints=_build_constraints(n),
            options={'maxiter': 3000, 'ftol': 1e-14}
        )
        centers = np.array([[result.x[i*3], result.x[i*3+1]] for i in range(n)])
        radii = np.array([max(result.x[i*3+2], 1e-8) for i in range(n)])
        
        # Phase 2: Enforce validity and expand
        radii = _enforce_validity(centers, radii)
        radii = _expand_radii(centers, radii)
        
        # Phase 3: Position refinement with radius expansion
        centers, radii = _refine_positions(centers, radii)
        
        if np.sum(radii) > best_sum:
            best_sum, best_centers, best_radii = np.sum(radii), centers.copy(), radii.copy()
    
    return best_centers, best_radii, best_sum


def _build_constraints(n):
    """Build boundary and no-overlap constraints."""
    cons = []
    for i in range(n):
        idx = i * 3
        cons.extend([
            {'type': 'ineq', 'fun': lambda v, i=idx: v[i] - v[i+2]},
            {'type': 'ineq', 'fun': lambda v, i=idx: v[i+1] - v[i+2]},
            {'type': 'ineq', 'fun': lambda v, i=idx: 1 - v[i] - v[i+2]},
            {'type': 'ineq', 'fun': lambda v, i=idx: 1 - v[i+1] - v[i+2]}
        ])
    for i in range(n):
        for j in range(i+1, n):
            ii, jj = i*3, j*3
            cons.append({'type': 'ineq', 'fun': lambda v, a=ii, b=jj: 
                np.sqrt((v[a]-v[b])**2 + (v[a+1]-v[b+1])**2) - v[a+2] - v[b+2]})
    return cons


def _generate_initial_guesses():
    """Generate diverse initial configurations with multiple geometric patterns."""
    guesses = []
    n = 26
    
    # Pattern 1: Corner-heavy with varied sizes
    for cr in [0.125, 0.12, 0.115, 0.11]:
        for er in [0.07, 0.065, 0.075, 0.06]:
            for ir in [0.065, 0.07, 0.055, 0.06]:
                v = []
                for cx, cy in [(cr, cr), (1-cr, cr), (cr, 1-cr), (1-cr, 1-cr)]:
                    v.extend([cx, cy, cr])
                for x in [0.35, 0.50, 0.65]:
                    v.extend([x, er, er])
                for x in [0.35, 0.50, 0.65]:
                    v.extend([x, 1-er, er])
                v.extend([er, 0.50, er])
                v.extend([1-er, 0.50, er])
                for x in [0.22, 0.42, 0.58, 0.78]:
                    v.extend([x, 0.25, ir])
                for x in [0.32, 0.50, 0.68]:
                    v.extend([x, 0.42, ir])
                for x in [0.22, 0.42, 0.58, 0.78]:
                    v.extend([x, 0.58, ir])
                for x in [0.32, 0.50, 0.68]:
                    v.extend([x, 0.75, ir])
                guesses.append(np.array(v))
    
    # Pattern 2: Hexagonal grid pattern
    for base_r in [0.085, 0.09, 0.095]:
        v = []
        pts = []
        spacing = 2.2 * base_r
        for row in range(6):
            y = 0.1 + row * spacing * 0.866
            if y > 0.95:
                continue
            cols = 5 if row % 2 == 0 else 4
            x_start = 0.1 if row % 2 == 0 else 0.1 + spacing/2
            for col in range(cols):
                x = x_start + col * spacing
                if x < 0.95:
                    pts.append((x, y))
        pts = pts[:n]
        for x, y in pts:
            v.extend([x, y, base_r])
        while len(v) < 3*n:
            v.extend([0.5, 0.5, 0.01])
        guesses.append(np.array(v[:3*n]))
    
    # Pattern 3: Dense center with smaller edges
    for center_r in [0.10, 0.095]:
        v = []
        # Larger central cluster
        for dx, dy in [(0,0), (0.2,0), (-0.2,0), (0,0.2), (0,-0.2),
                       (0.15,0.15), (-0.15,0.15), (0.15,-0.15), (-0.15,-0.15)]:
            v.extend([0.5+dx, 0.5+dy, center_r])
        # Smaller corner/edge circles
        for cx, cy in [(0.1,0.1), (0.9,0.1), (0.1,0.9), (0.9,0.9)]:
            v.extend([cx, cy, 0.08])
        for x in [0.3, 0.5, 0.7]:
            v.extend([x, 0.05, 0.05])
            v.extend([x, 0.95, 0.05])
        for y in [0.3, 0.5, 0.7]:
            v.extend([0.05, y, 0.05])
            v.extend([0.95, y, 0.05])
        guesses.append(np.array(v[:3*n]))
    
    return guesses[:25]


def _enforce_validity(centers, radii):
    """Shrink radii to satisfy all constraints."""
    n = len(radii)
    radii = radii.copy()
    for i in range(n):
        radii[i] = min(radii[i], centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d + 1e-12:
                    s = (d - 1e-9) / (radii[i] + radii[j])
                    radii[i] *= s
                    radii[j] *= s
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 1e-8)


def _expand_radii(centers, radii):
    """Priority-based radius expansion - expand circle with most room first."""
    n = len(radii)
    radii = radii.copy()
    
    for iteration in range(3000):
        # Find circle with maximum potential gain
        best_gain = 0
        best_i = -1
        best_max_r = radii.copy()
        
        for i in range(n):
            max_r = min(centers[i,0], centers[i,1], 1-centers[i,0], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i]-centers[j])
                    max_r = min(max_r, d - radii[j])
            gain = max_r - radii[i]
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_max_r[i] = max_r
        
        if best_gain < 1e-12:
            break
        
        radii[best_i] = best_max_r[best_i]
    
    return radii


def _refine_positions(centers, radii):
    """Adaptive position refinement with decreasing step sizes and 16 directions."""
    n = len(radii)
    centers = centers.copy()
    radii = radii.copy()
    
    # Multi-scale refinement: start with larger steps, progressively reduce
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        for iteration in range(100):
            improved = False
            for i in range(n):
                best_local_r = radii[i]
                best_pos = centers[i].copy()
                
                # Check 16 directions uniformly
                for k in range(16):
                    angle = 2 * np.pi * k / 16
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    new_x = centers[i, 0] + dx
                    new_y = centers[i, 1] + dy
                    
                    if new_x < 0.005 or new_x > 0.995 or new_y < 0.005 or new_y > 0.995:
                        continue
                    
                    # Calculate max radius at new position
                    max_r = min(new_x, new_y, 1-new_x, 1-new_y)
                    for j in range(n):
                        if i != j:
                            d = np.sqrt((new_x - centers[j,0])**2 + (new_y - centers[j,1])**2)
                            max_r = min(max_r, d - radii[j])
                    
                    if max_r > best_local_r + 1e-12:
                        best_local_r = max_r
                        best_pos = np.array([new_x, new_y])
                        improved = True
                
                if best_local_r > radii[i] + 1e-12:
                    centers[i] = best_pos
                    radii[i] = best_local_r
            
            if not improved:
                break
    
    # Final aggressive radius expansion
    radii = _expand_radii(centers, radii)
    
    return centers, radii


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
