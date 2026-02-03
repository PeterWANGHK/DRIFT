"""
DRIFT: Dynamic Risk Inference via Field Transport
=================================================
REFACTORED visualization matching reference GIF style:

Key Fixes:
1. Local advection: Each vehicle's risk advects at that vehicle's ABSOLUTE speed
2. Background blend: Regions with weak influence move with average group speed
3. Smooth diffusion: Risk spreads to larger, smoother shapes (source_sigma_scale=1.4)
4. Occlusion risk: Smoothly wraps around truck-trailer body + shadow
5. Merge pressure: Scales with remaining ramp distance (pressure ∝ 1/remaining_dist)
6. No free-floating risk: Risk stays tied to traffic sources
7. Fixed plot range: Traffic stays within frame, movement shown via position changes

Style: Matches reference GIF with step counter, jet colormap, speed labels
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio")


# =============================================================================
# Configuration
# =============================================================================

class DRIFTConfig:
    """Configuration for DRIFT visualization."""
    # Fixed plot domain (matches reference GIF)
    x_min, x_max = 25, 235
    y_min, y_max = -2, 14
    nx, ny = 280, 50
    
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # PDE parameters
    D0 = 2.5              # Base diffusion
    D_occ = 6.0           # Enhanced diffusion around occlusion
    lambda_decay = 0.55  # Slow decay
    
    # Source spread (enlarged for smoother fields)
    source_sigma_scale = 1.4
    sigma_x_base = 10.0
    sigma_y_base = 2.0
    
    # Lane geometry (4 lanes like reference)
    lane_width = 3.5
    lane_centers = [2, 5.5, 9, 12.5]
    
    # Merge zone
    merge_x_start = 160
    merge_x_end = 210


cfg = DRIFTConfig()


# =============================================================================
# Vehicle Functions
# =============================================================================

def create_vehicle(vid, x, y, vx, vy, vclass='car', length=None, width=None):
    """Create vehicle dictionary."""
    if length is None:
        length = 10.0 if vclass == 'truck_trailer' else 4.5
    if width is None:
        width = 2.5 if 'truck' in vclass else 2.0
    heading = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0
    return {
        'id': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy,
        'heading': heading, 'class': vclass, 'length': length, 'width': width
    }


def move_vehicle(v, dt, ax_accel=0, ay_accel=0):
    """Move vehicle, keeping within plot bounds."""
    if v is None:
        return None
    
    new_x = v['x'] + v['vx'] * dt + 0.5 * ax_accel * dt**2
    new_y = v['y'] + v['vy'] * dt + 0.5 * ay_accel * dt**2
    new_vx = max(15, v['vx'] + ax_accel * dt)
    new_vy = v['vy'] + ay_accel * dt
    
    # Clamp y to road bounds
    new_y = np.clip(new_y, cfg.lane_centers[0] - 1, cfg.lane_centers[-1] + 1)
    
    # FIXED PLOT RANGE: Wrap vehicles that exit right to left side
    if new_x > cfg.x_max - 5:
        new_x = cfg.x_min + 10 + np.random.uniform(0, 20)
    if new_x < cfg.x_min + 5:
        new_x = cfg.x_min + 10
    
    return create_vehicle(v['id'], new_x, new_y, new_vx, new_vy,
                         v['class'], v['length'], v['width'])


# =============================================================================
# Risk Source Computation with Local Advection Velocities
# =============================================================================

def compute_vehicle_sources_with_velocity(vehicles, X, Y):
    """
    Compute vehicle risk sources and LOCAL velocity field.
    Each vehicle's risk advects at that vehicle's ABSOLUTE speed.
    """
    R = np.zeros_like(X)
    vx_weighted = np.zeros_like(X)
    vy_weighted = np.zeros_like(Y)
    weight_sum = np.ones_like(X) * 1e-8
    
    # Enlarged source spread
    sigma_x = cfg.sigma_x_base * cfg.source_sigma_scale
    sigma_y = cfg.sigma_y_base * cfg.source_sigma_scale
    
    for v in vehicles:
        if v is None:
            continue
        
        # Vehicle class weight
        weight = 1.3 if 'truck' in v['class'] else 1.0
        
        # Speed-based intensity (faster vehicles = slightly higher risk)
        speed = np.sqrt(v['vx']**2 + v['vy']**2)
        intensity = 0.5 + 0.015 * speed
        
        # Anisotropic Gaussian centered on vehicle (ENLARGED for smoother fields)
        h = v['heading']
        cos_h, sin_h = np.cos(h), np.sin(h)
        dX = X - v['x']
        dY = Y - v['y']
        dX_rot = cos_h * dX + sin_h * dY
        dY_rot = -sin_h * dX + cos_h * dY
        
        # Scale by vehicle size
        sigma_par = sigma_x * (1 + 0.02 * v['length'])
        sigma_perp = sigma_y * (1 + 0.05 * v['width'])
        
        gaussian = np.exp(-0.5 * (dX_rot**2 / sigma_par**2 + dY_rot**2 / sigma_perp**2))
        
        # Accumulate risk
        source = weight * intensity * gaussian
        R += source
        
        # Weight velocity by source strength for LOCAL ADVECTION
        # Use ABSOLUTE velocity (not relative!) so each source advects at its own speed
        vx_weighted += source * v['vx']
        vy_weighted += source * v['vy']
        weight_sum += source
    
    # Normalize to get local velocity field
    vx_local = vx_weighted / weight_sum
    vy_local = vy_weighted / weight_sum
    
    # Background blend: regions with weak influence use average group speed
    avg_vx = np.mean([v['vx'] for v in vehicles if v is not None])
    blend_factor = np.clip(weight_sum / 0.3, 0, 1)  # Blend threshold
    vx_local = blend_factor * vx_local + (1 - blend_factor) * avg_vx
    vy_local = blend_factor * vy_local
    
    # CLIP velocity to prevent abrupt out-of-traffic motion
    max_speed = max([v['vx'] for v in vehicles if v is not None]) * 1.2
    vx_local = np.clip(vx_local, 0, max_speed)
    
    return R, vx_local, vy_local


def compute_occlusion_risk(ego, truck, X, Y):
    """
    Compute smooth occlusion risk that WRAPS AROUND the truck-trailer.
    Includes:
    1. Soft blob around truck body (proximity uncertainty)
    2. Shadow cone behind truck (sensor blind zone)
    3. Edge emergence zones (cut-in areas)
    """
    R_occ = np.zeros_like(X)
    occ_mask = np.zeros_like(X, dtype=bool)
    
    if truck is None:
        return R_occ, occ_mask, None
    
    # Vector from ego to truck
    dx_ego = truck['x'] - ego['x']
    if dx_ego < 3:
        return R_occ, occ_mask, None
    
    L, W = truck['length'], truck['width']
    tx, ty = truck['x'], truck['y']
    
    # 1. SOFT BLOB around truck body (uncertainty near large vehicle)
    dist_to_truck = np.sqrt((X - tx)**2 + (Y - ty)**2)
    truck_blob = 0.35 * np.exp(-dist_to_truck**2 / (L * 1.5)**2)
    
    # 2. SHADOW CONE behind truck (from ego's perspective)
    behind_truck = X > tx + L/4
    dist_from_back = X - (tx + L/2)
    dist_from_back = np.maximum(dist_from_back, 0)
    
    # Shadow width expands with distance
    shadow_half_width = W/2 + dist_from_back * 0.08
    lateral_dist = np.abs(Y - ty)
    in_shadow = behind_truck & (lateral_dist < shadow_half_width) & (dist_from_back < 50)
    
    occ_mask = in_shadow
    
    # Shadow risk: concentrated at lane centers
    shadow_dist = np.sqrt(dist_from_back**2 + (Y - ty)**2)
    p_dist = np.exp(-shadow_dist / 25)
    
    # Lane prior
    p_lane = np.zeros_like(X)
    for lane_y in cfg.lane_centers:
        p_lane += np.exp(-0.5 * ((Y - lane_y) / 1.2)**2)
    p_lane = p_lane / (p_lane.max() + 1e-6)
    
    shadow_risk = in_shadow.astype(float) * 0.5 * p_dist * (0.5 + 0.5 * p_lane)
    
    # 3. EDGE EMERGENCE zones (vehicles cut in from truck edges)
    # Top edge
    edge_top = 0.25 * np.exp(-((Y - (ty + W/2 + 1))**2 / 3 + (X - tx - 8)**2 / 80))
    # Bottom edge  
    edge_bot = 0.25 * np.exp(-((Y - (ty - W/2 - 1))**2 / 3 + (X - tx - 8)**2 / 80))
    edge_risk = (edge_top + edge_bot) * (X > tx)
    
    # Combine all occlusion components (SMOOTH wrapping)
    R_occ = truck_blob + shadow_risk + edge_risk
    R_occ = gaussian_filter(R_occ, sigma=1.5)  # Smooth transitions
    
    # Create shadow polygon for visualization
    shadow_polygon = None
    if dx_ego > 5:
        corners = np.array([
            [tx + L/2, ty - W/2],
            [tx + L/2 + 45, ty - W/2 - 4],
            [tx + L/2 + 45, ty + W/2 + 4],
            [tx + L/2, ty + W/2]
        ])
        shadow_polygon = corners
    
    return R_occ, occ_mask, shadow_polygon


def compute_merge_pressure(vehicles, X, Y):
    """
    Compute merge pressure that scales with REMAINING RAMP DISTANCE.
    Pressure = k / remaining_distance (increases as vehicle approaches gore)
    """
    R_merge = np.zeros_like(X)
    
    for v in vehicles:
        if v is None:
            continue
        
        # Check if in ramp lane and merge zone
        in_ramp = abs(v['y'] - cfg.lane_centers[-1]) < cfg.lane_width
        in_merge_zone = cfg.merge_x_start < v['x'] < cfg.merge_x_end
        
        if in_ramp and in_merge_zone:
            # REMAINING DISTANCE to gore point
            remaining_dist = max(cfg.merge_x_end - v['x'], 3)
            
            # Merge pressure INVERSELY PROPORTIONAL to remaining distance
            k_merge = 40.0
            pressure = k_merge / remaining_dist
            
            # Conflict zone extends ahead and toward mainline
            dX = X - v['x']
            dY = Y - v['y']
            
            # Risk ahead of vehicle
            ahead = np.exp(-0.5 * ((dX - 8)**2 / 120))
            
            # Lateral spread toward mainline
            target_y = cfg.lane_centers[-2]
            lateral_dir = (v['y'] - target_y) / 2
            lateral_spread = np.exp(-0.5 * ((dY + lateral_dir)**2 / 8))
            
            R_merge += pressure * 0.03 * ahead * lateral_spread
    
    # Background merge zone (topology-induced)
    s = np.clip((X - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start), 0, 1)
    ramp_up = s**2 * (3 - 2*s)
    
    merge_y = (cfg.lane_centers[-1] + cfg.lane_centers[-2]) / 2
    lateral_bg = np.exp(-0.5 * ((Y - merge_y)**2 / 8))
    
    R_merge += 0.12 * ramp_up * lateral_bg
    
    return R_merge


# =============================================================================
# PDE Solver with Local Advection
# =============================================================================

class DRIFTSolver:
    """
    Advection-Diffusion solver where risk advects at LOCAL vehicle speeds.
    ∂R/∂t = -∇·(v_local R) + ∇·(D∇R) + Q - λR
    """
    
    def __init__(self):
        self.R = np.zeros_like(cfg.X)
    
    def reset(self):
        self.R = np.zeros_like(cfg.X)
    
    def step(self, Q, D, vx, vy, dt=0.05):
        """
        Advance one timestep:
        - Source injection Q
        - Local advection at vehicle speeds (v_local)
        - Diffusion with spatially-varying D
        - Slow decay
        """
        R = self.R.copy()
        
        # 1. Source injection (tied to vehicles)
        R = R + dt * Q * 0.3
        
        # 2. Decay
        R = R * (1 - cfg.lambda_decay * dt)
        
        # 3. ADVECTION at local velocities: -∇·(vR)
        dR_dx = np.zeros_like(R)
        dR_dy = np.zeros_like(R)
        
        # Upwind scheme
        dR_dx[:, 1:-1] = np.where(vx[:, 1:-1] > 0,
                                   (R[:, 1:-1] - R[:, :-2]) / cfg.dx,
                                   (R[:, 2:] - R[:, 1:-1]) / cfg.dx)
        dR_dy[1:-1, :] = np.where(vy[1:-1, :] > 0,
                                   (R[1:-1, :] - R[:-2, :]) / cfg.dy,
                                   (R[2:, :] - R[1:-1, :]) / cfg.dy)
        
        # Scale advection to match vehicle speeds (risk moves WITH traffic)
        advection = vx * dR_dx + vy * dR_dy
        R = R - dt * advection * 0.08  # Reduced coefficient for stability
        
        # 4. DIFFUSION: ∇·(D∇R) - spreads risk to larger, smoother shape
        laplacian = np.zeros_like(R)
        laplacian[1:-1, 1:-1] = D[1:-1, 1:-1] * (
            (R[1:-1, 2:] - 2*R[1:-1, 1:-1] + R[1:-1, :-2]) / cfg.dx**2 +
            (R[2:, 1:-1] - 2*R[1:-1, 1:-1] + R[:-2, 1:-1]) / cfg.dy**2
        )
        R = R + dt * laplacian
        
        # Clamp and smooth
        R = np.clip(R, 0, 2.5)
        R = gaussian_filter(R, sigma=0.5)
        
        # Boundaries
        R[0, :] = R[1, :]
        R[-1, :] = R[-2, :]
        R[:, 0] = R[:, 1]
        R[:, -1] = R[:, -2]
        
        self.R = R
        return self.R.copy()


# =============================================================================
# Visualization (Reference GIF Style)
# =============================================================================

def draw_road(ax):
    """Draw road with lanes (reference style)."""
    road_bottom = cfg.lane_centers[0] - cfg.lane_width/2
    road_top = cfg.lane_centers[-1] + cfg.lane_width/2
    ax.fill_between([cfg.x_min, cfg.x_max], [road_bottom]*2, [road_top]*2,
                   color='#2a2a4a', alpha=0.3, zorder=0)
    
    for i, lane_y in enumerate(cfg.lane_centers):
        if i == 0:
            ax.axhline(y=road_bottom, color='white', linestyle='-', linewidth=1.5, alpha=0.8)
        if i == len(cfg.lane_centers) - 1:
            ax.axhline(y=road_top, color='white', linestyle='-', linewidth=1.5, alpha=0.8)
        else:
            ax.axhline(y=lane_y + cfg.lane_width/2, color='white', linestyle='--', 
                      linewidth=1, alpha=0.6)


def draw_vehicle(ax, v, show_speed=True):
    """Draw vehicle (reference style: orange/yellow rectangles with speed labels)."""
    if v is None:
        return
    
    x, y = v['x'], v['y']
    L, W = v['length'], v['width']
    
    # Vehicle color
    if 'truck' in v['class']:
        fc = 'orange'
        ec = 'darkorange'
    else:
        fc = '#FFD700'  # Gold/yellow like reference
        ec = '#DAA520'
    
    # Rectangle
    rect = mpatches.Rectangle((x - L/2, y - W/2), L, W,
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.5, zorder=10)
    ax.add_patch(rect)
    
    # Center dot
    ax.plot(x, y, 'b.', markersize=4, zorder=11)
    
    # Speed label (reference style: above vehicle)
    if show_speed:
        speed = np.sqrt(v['vx']**2 + v['vy']**2)
        ax.text(x, y - W/2 - 0.6, f'Speed: {speed:.2f} m/s',
               fontsize=6, ha='center', va='top', color='black',
               fontweight='bold')


def create_scenario():
    """Create traffic scenario matching reference GIF."""
    vehicles = [
        # Lane 1 (y ≈ 2) - fast lane
        create_vehicle(0, 80, cfg.lane_centers[0], 25.31, 0, 'car'),
        
        # Lane 2 (y ≈ 5.5) - includes truck
        #create_vehicle(1, 130, cfg.lane_centers[1], 22.41, 0, 'car'),
        #create_vehicle(2, 155, cfg.lane_centers[1], 23.59, 0, 'car'),
        #create_vehicle(3, 175, cfg.lane_centers[1], 22.75, 0, 'car'),
        
        # Truck-trailer
        #create_vehicle(4, 115, cfg.lane_centers[1], 20.0, 0, 'truck_trailer', 10, 2.5),
        
        # Lane 3 (y ≈ 9)
        #create_vehicle(5, 175, cfg.lane_centers[2], 22.36, 0, 'car', 5),
        #create_vehicle(6, 200, cfg.lane_centers[2], 23.26, 0, 'car'),
        
        # Lane 4 (y ≈ 12.5) - includes merging vehicle
        #create_vehicle(7, 175, cfg.lane_centers[3], 21.90, 0, 'car'),
        #create_vehicle(8, 205, cfg.lane_centers[3], 21.04, 0, 'car'),
    ]
    
    return vehicles


# =============================================================================
# Main Animation - ADVECTION Effect
# =============================================================================

def create_advection_gif(output_path='./output', n_frames=80, fps=10):
    """
    ADVECTION animation: Risk field moves WITH traffic flow.
    Each vehicle's risk advects at that vehicle's absolute speed.
    """
    print("\n" + "=" * 70)
    print("DRIFT: ADVECTION Effect Animation")
    print("Risk follows vehicle speeds locally, diffuses to smooth shape")
    print("=" * 70)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    vehicles = create_scenario()
    solver = DRIFTSolver()
    
    # Ego for occlusion reference
    ego = create_vehicle(-1, 60, cfg.lane_centers[1], 22, 0, 'car')
    
    dt = 0.12
    frames = []
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    print(f"Generating {n_frames} frames...")
    
    for step in range(n_frames):
        if step % 10 == 0:
            print(f"  Frame {step}/{n_frames}")
        
        ax.clear()
        
        # Find truck
        truck = next((v for v in vehicles if v and 'truck' in v['class']), None)
        
        # Compute vehicle sources with LOCAL velocity field
        R_veh, vx_local, vy_local = compute_vehicle_sources_with_velocity(vehicles, cfg.X, cfg.Y)
        
        # Compute occlusion risk (wraps around truck)
        R_occ, occ_mask, shadow_poly = compute_occlusion_risk(ego, truck, cfg.X, cfg.Y)
        
        # Compute merge pressure
        R_merge = compute_merge_pressure(vehicles, cfg.X, cfg.Y)
        
        # Total source
        Q_total = R_veh + R_occ + R_merge
        
        # Diffusion coefficient: base + enhanced around occlusion
        D = cfg.D0 * np.ones_like(cfg.X)
        if occ_mask.any():
            occ_influence = gaussian_filter(occ_mask.astype(float), sigma=4)
            D = D + cfg.D_occ * occ_influence
        
        # Evolve PDE with local advection
        for _ in range(4):
            solver.step(Q_total, D, vx_local, vy_local, dt=dt/4)
        
        # Blend with source to keep risk TIED TO TRAFFIC
        R_display = 0.55 * solver.R + 0.45 * gaussian_filter(Q_total, sigma=2.5)
        R_display = gaussian_filter(R_display, sigma=1.2)
        
        # Plot risk field
        pcm = ax.pcolormesh(cfg.X, cfg.Y, R_display, cmap='jet', shading='gouraud',
                           vmin=0, vmax=1.2)
        
        draw_road(ax)
        
        # Draw vehicles
        for v in vehicles:
            draw_vehicle(ax, v, show_speed=True)
        
        # Step counter (reference style)
        ax.text(cfg.x_min + 3, cfg.y_min + 0.3, f'Step: {step + 1}',
               fontsize=10, fontweight='bold', color='black',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Fixed axis
        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.y_min, cfg.y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x [m]', fontsize=9)
        ax.set_ylabel('y [m]', fontsize=9)
        
        # Colorbar
        if step == 0:
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Risk value', fontsize=9)
        
        fig.tight_layout()
        
        # Capture frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
        
        # Move vehicles (stay within plot bounds)
        vehicles = [move_vehicle(v, dt, ax_accel=0, ay_accel=0) for v in vehicles]
        ego = move_vehicle(ego, dt, ax_accel=0, ay_accel=0)
    
    plt.close()
    
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, 'drift_advection_effect.gif')
        print(f"\nSaving: {gif_path}")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"Saved: {gif_path}")
        return gif_path
    
    return None


# =============================================================================
# Main Animation - DIFFUSION Effect
# =============================================================================

def create_diffusion_gif(output_path='./output', n_frames=80, fps=10):
    """
    DIFFUSION animation: Risk spreads smoothly, enhanced in occlusion zone.
    Shows how risk diffuses from occlusion region outward.
    """
    print("\n" + "=" * 70)
    print("DRIFT: DIFFUSION Effect Animation")
    print("Risk spreads from occlusion with enhanced D")
    print("=" * 70)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    vehicles = create_scenario()
    solver = DRIFTSolver()
    
    ego = create_vehicle(-1, 60, cfg.lane_centers[1], 22, 0, 'car')
    
    dt = 0.12
    frames = []
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    print(f"Generating {n_frames} frames...")
    
    for step in range(n_frames):
        if step % 10 == 0:
            print(f"  Frame {step}/{n_frames}")
        
        ax.clear()
        
        truck = next((v for v in vehicles if v and 'truck' in v['class']), None)
        
        # Compute sources with local velocities
        R_veh, vx_local, vy_local = compute_vehicle_sources_with_velocity(vehicles, cfg.X, cfg.Y)
        R_occ, occ_mask, shadow_poly = compute_occlusion_risk(ego, truck, cfg.X, cfg.Y)
        R_merge = compute_merge_pressure(vehicles, cfg.X, cfg.Y)
        
        Q_total = R_veh + R_occ + R_merge
        
        # Diffusion coefficient with strong enhancement in occlusion
        D = cfg.D0 * np.ones_like(cfg.X)
        if occ_mask.any():
            occ_influence = gaussian_filter(occ_mask.astype(float), sigma=4)
            D = D + cfg.D_occ * 1.5 * occ_influence  # Enhanced for diffusion demo
        
        # Evolve
        for _ in range(4):
            solver.step(Q_total, D, vx_local, vy_local, dt=dt/4)
        
        R_display = 0.5 * solver.R + 0.5 * gaussian_filter(Q_total, sigma=2.5)
        R_display = gaussian_filter(R_display, sigma=1.2)
        
        # Plot
        pcm = ax.pcolormesh(cfg.X, cfg.Y, R_display, cmap='jet', shading='gouraud',
                           vmin=0, vmax=1.2)
        
        draw_road(ax)
        
        for v in vehicles:
            draw_vehicle(ax, v, show_speed=True)
        
        ax.text(cfg.x_min + 3, cfg.y_min + 0.3, f'Step: {step + 1}',
               fontsize=10, fontweight='bold', color='black',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.y_min, cfg.y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x [m]', fontsize=9)
        ax.set_ylabel('y [m]', fontsize=9)
        
        if step == 0:
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Risk value', fontsize=9)
        
        fig.tight_layout()
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
        
        vehicles = [move_vehicle(v, dt, ax_accel=0, ay_accel=0) for v in vehicles]
        ego = move_vehicle(ego, dt, ax_accel=0, ay_accel=0)
    
    plt.close()
    
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, 'drift_diffusion_effect.gif')
        print(f"\nSaving: {gif_path}")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"Saved: {gif_path}")
        return gif_path
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DRIFT: Risk Field Visualization')
    parser.add_argument('-o', '--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--frames', type=int, default=40, help='Number of frames')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    
    args = parser.parse_args()
    
    create_advection_gif(args.output, n_frames=args.frames, fps=args.fps)
    create_diffusion_gif(args.output, n_frames=args.frames, fps=args.fps)
    
    print("\n" + "=" * 70)
    print("Done!")
    print(f"Output saved to: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
