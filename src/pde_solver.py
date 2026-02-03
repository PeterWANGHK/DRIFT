"""
PDE Risk Field Solver
Advection-Diffusion-Telegrapher Equation for Traffic Risk Propagation

PDE: τ∂²R/∂t² + ∂R/∂t + ∇·(vR) = ∇·(D∇R) + Q(x,t) - λR

Based on Zhang et al. (2021) GVF formulation extended with:
- Occlusion-induced hazard sources
- Merge topology drift
- Telegrapher (damped wave) dynamics
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from config import Config as cfg


# ============================================================
# VEHICLE UTILITIES
# ============================================================

def create_vehicle(vid, x, y, vx, vy, vclass='car'):
    """
    Create a vehicle dictionary.
    
    Args:
        vid: Vehicle ID
        x, y: Position [m]
        vx, vy: Velocity [m/s]
        vclass: 'car' or 'truck'
    
    Returns:
        dict: Vehicle info
    """
    length = cfg.truck_length if vclass == 'truck' else cfg.car_length
    width = cfg.truck_width if vclass == 'truck' else cfg.car_width
    heading = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0
    
    return {
        'id': vid,
        'x': x, 'y': y,
        'vx': vx, 'vy': vy,
        'heading': heading,
        'class': vclass,
        'length': length,
        'width': width
    }


def move_vehicle(v, dt, ax=0, ay=0):
    """
    Move vehicle forward in time with optional acceleration.
    
    Args:
        v: Vehicle dict
        dt: Time step [s]
        ax, ay: Acceleration [m/s²]
    
    Returns:
        dict: Updated vehicle
    """
    if v is None:
        return None
    
    new_x = v['x'] + v['vx'] * dt + 0.5 * ax * dt**2
    new_y = v['y'] + v['vy'] * dt + 0.5 * ay * dt**2
    new_vx = v['vx'] + ax * dt
    new_vy = v['vy'] + ay * dt
    
    return create_vehicle(v['id'], new_x, new_y, new_vx, new_vy, v['class'])


# ============================================================
# SOURCE TERM Q(x,t)
# ============================================================

def compute_Q_vehicle(vehicles, ego, X, Y):
    """
    Compute vehicle-induced risk source using GVF-style Gaussian kernels.
    
    Q_veh(x,t) = Σ_i w_i(t) * N(x; x_i(t), Σ_i(t))
    
    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates
    
    Returns:
        ndarray: Vehicle source field Q_veh
    """
    Q = np.zeros_like(X)
    
    ego_vx, ego_vy = ego['vx'], ego['vy']
    
    for v in vehicles:
        if v is None or v['id'] == ego['id']:
            continue
        
        # Relative velocity (determines risk weight)
        rel_vx = v['vx'] - ego_vx
        rel_vy = v['vy'] - ego_vy
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
        
        # Distance to ego
        dx = v['x'] - ego['x']
        dy = v['y'] - ego['y']
        dist = np.sqrt(dx**2 + dy**2)
        
        # Risk weight: higher for closer vehicles with higher relative speed
        # w_i = ω_class * ω_TTC * ω_rel
        omega_class = 1.5 if v['class'] == 'truck' else 1.0
        omega_dist = np.exp(-dist / 50.0)
        omega_rel = 1.0 + rel_speed / 5.0
        
        weight = omega_class * omega_dist * omega_rel
        
        # Anisotropic Gaussian kernel (aligned with vehicle heading)
        heading = v['heading']
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        
        # Rotate coordinates to vehicle-aligned frame
        dX = X - v['x']
        dY = Y - v['y']
        dX_rot = cos_h * dX + sin_h * dY
        dY_rot = -sin_h * dX + cos_h * dY
        
        # Kernel parameters (inflate with relative speed)
        sigma_par = cfg.sigma_x * (1 + 0.05 * np.abs(rel_vx))
        sigma_perp = cfg.sigma_y
        
        # Gaussian kernel
        gaussian = np.exp(-0.5 * (dX_rot**2 / sigma_par**2 + dY_rot**2 / sigma_perp**2))
        
        Q += weight * gaussian
    
    return Q


def compute_Q_occlusion(vehicles, ego, X, Y):
    """
    Compute occlusion-induced latent hazard source.
    
    Q_occ(x,t) = 1_{x ∈ Ω_occ} * ω_sev * p_emerge(x,t)
    
    Args:
        vehicles: List of vehicle dicts (trucks create occlusion)
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates
    
    Returns:
        Q_occ: Occlusion source field
        occ_mask: Boolean mask of occluded regions
    """
    Q_occ = np.zeros_like(X)
    occ_mask = np.zeros_like(X, dtype=bool)
    
    for v in vehicles:
        if v is None or v['class'] != 'truck':
            continue
        
        # Direction from ego to truck
        to_truck = np.array([v['x'] - ego['x'], v['y'] - ego['y']])
        dist_to_truck = np.linalg.norm(to_truck)
        
        if dist_to_truck < 1:
            continue
        
        # Shadow region extends behind truck (from ego's perspective)
        dx = X - v['x']
        dy = Y - v['y']
        
        # Points behind the truck (dot product with direction to truck > 0)
        behind = (dx * to_truck[0] + dy * to_truck[1]) > 0
        
        # Within shadow cone (lateral spread increases with distance)
        dist_from_truck = np.sqrt(dx**2 + dy**2)
        heading = v['heading']
        lateral_dist = np.abs(-np.sin(heading) * dx + np.cos(heading) * dy)
        
        shadow_width = v['width'] + dist_from_truck * 0.15
        in_shadow = behind & (lateral_dist < shadow_width) & (dist_from_truck < 50)
        
        occ_mask |= in_shadow
        
        # Occlusion risk: decays with distance from occluding vehicle
        p_emerge = np.exp(-dist_from_truck / 30)
        Q_occ += in_shadow.astype(float) * 0.5 * p_emerge
    
    return Q_occ, occ_mask


def compute_Q_merge(vehicles, ego, X, Y):
    """
    Compute merge-zone conflict source, gated by vehicle density.

    Q_merge = ρ_vehicles(x) * [k_merge * ρ_merge(x)]

    Only injects risk where vehicles are actually present.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        ndarray: Merge zone source field (gated by vehicle density)
    """
    # Merge zone geometry (base risk pattern)
    s = np.clip((X - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start), 0, 1)
    ramp = 3 * s**2 - 2 * s**3  # Smooth onset g(s)

    # Lateral: concentrated between ramp and mainline
    y_center = cfg.merge_y_ramp / 2
    lateral = np.exp(-0.5 * ((Y - y_center)**2 / 16))

    # Gore point intensity
    gore = np.exp(-((X - cfg.merge_x_end)**2 + (Y - 4)**2) / 100)

    Q_merge_base = 0.3 * ramp * lateral + 0.5 * gore

    # NEW: Compute vehicle density field (only inject risk where vehicles present)
    rho_vehicles = np.zeros_like(X)
    for v in vehicles:
        if v is None or v['id'] == ego['id']:
            continue
        # Gaussian density around each vehicle (30m x 4m influence region)
        dist_sq = ((X - v['x'])**2 / 900 + (Y - v['y'])**2 / 16)
        rho_vehicles += np.exp(-0.5 * dist_sq)

    # Normalize and gate
    rho_norm = np.clip(rho_vehicles, 0, 1.0)

    # Only inject merge risk where vehicles are present
    Q_merge = Q_merge_base * rho_norm

    return Q_merge


def compute_Q_merge_decomposed(vehicles, ego, X, Y):
    """
    Decompose merge risk into ambient (topology) and vehicle-induced components.

    This maintains unified PDE formulation while allowing different decay rates:
    - Ambient: persistent structural hazard (slower decay, reaches equilibrium)
    - Vehicle: dynamic conflict risk (faster decay via activity-based mechanism)

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        Q_merge_ambient: Small persistent risk from merge topology (always present)
        Q_merge_vehicle: Vehicle-gated risk (decays when vehicles leave)
    """
    # ===== SPATIAL GEOMETRY (shared by both components) =====
    # Smooth onset function g(s) = 3s² - 2s³
    s = np.clip((X - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start), 0, 1)
    ramp = 3 * s**2 - 2 * s**3

    # Lateral: concentrated between ramp and mainline
    y_center = cfg.merge_y_ramp / 2
    lateral = np.exp(-0.5 * ((Y - y_center)**2 / 16))

    # Gore point intensity (conflict apex)
    gore = np.exp(-((X - cfg.merge_x_end)**2 + (Y - 4)**2) / 100)

    # Base spatial pattern
    merge_pattern = 0.3 * ramp * lateral + 0.5 * gore

    # ===== AMBIENT TOPOLOGY RISK (PERSISTENT) =====
    # Small constant risk indicating structural merge hazard
    Q_merge_ambient = cfg.k_merge_ambient * merge_pattern

    # ===== VEHICLE-INDUCED RISK (GATED, DECAYS) =====
    # Compute vehicle density field (only inject where vehicles present)
    rho_vehicles = np.zeros_like(X)
    for v in vehicles:
        if v is None or v['id'] == ego['id']:
            continue
        # Gaussian density around each vehicle (30m x 4m influence region)
        dist_sq = ((X - v['x'])**2 / 900 + (Y - v['y'])**2 / 16)
        rho_vehicles += np.exp(-0.5 * dist_sq)

    # Normalize and gate
    rho_norm = np.clip(rho_vehicles, 0, 1.0)

    # Vehicle-induced merge risk (only where vehicles present)
    Q_merge_vehicle = cfg.k_merge_vehicle * merge_pattern * rho_norm

    return Q_merge_ambient, Q_merge_vehicle


def compute_total_Q(vehicles, ego, X, Y):
    """
    Compute total source term Q = Q_veh + Q_occ + Q_merge.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict
        X, Y: Meshgrid coordinates

    Returns:
        Q_total: Total source field
        Q_veh: Vehicle source
        Q_occ: Occlusion source
        occ_mask: Occlusion mask
    """
    Q_veh = compute_Q_vehicle(vehicles, ego, X, Y)
    Q_occ, occ_mask = compute_Q_occlusion(vehicles, ego, X, Y)
    Q_merge = compute_Q_merge(vehicles, ego, X, Y)  # Now uses vehicle density

    Q_total = Q_veh + Q_occ + Q_merge

    return Q_total, Q_veh, Q_occ, occ_mask


# ============================================================
# VELOCITY FIELD v_eff = v_flow + v_topo
# ============================================================

def compute_velocity_field(vehicles, ego, X, Y):
    """
    Compute effective velocity field for advection.

    v_eff = v_flow (GVF) + v_topo (merge drift)

    For WORLD-FIXED frame: Uses absolute velocities so risk propagates with traffic.

    Args:
        vehicles: List of vehicle dicts
        ego: Ego vehicle dict (included for compatibility, not used for world frame)
        X, Y: Meshgrid coordinates

    Returns:
        vx, vy: Effective velocity components
        vx_flow, vy_flow: GVF flow field
        vx_topo, vy_topo: Topology drift field
    """
    # ---- GVF-style flow field ----
    vx_flow = np.zeros_like(X)
    vy_flow = np.zeros_like(Y)
    total_weight = np.zeros_like(X) + 1e-6

    for v in vehicles:
        if v is None:
            continue

        # ABSOLUTE velocity (for world-fixed frame)
        abs_vx = v['vx']
        abs_vy = v['vy']

        # Gaussian influence kernel
        dx = X - v['x']
        dy = Y - v['y']
        dist_sq = dx**2 / 400 + dy**2 / 9  # Anisotropic
        weight = np.exp(-0.5 * dist_sq)

        vx_flow += weight * abs_vx
        vy_flow += weight * abs_vy
        total_weight += weight

    vx_flow = vx_flow / total_weight
    vy_flow = vy_flow / total_weight
    
    # ---- Topology drift (merge zone) ----
    # Smooth onset function
    s = np.clip((X - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start), 0, 1)
    urgency = s**2 * (3 - 2*s)
    
    # Only apply in merge region
    in_merge = (Y > 2) & (X > cfg.merge_x_start) & (X < cfg.merge_x_end)
    
    vx_topo = np.zeros_like(X)
    vy_topo = -2.0 * urgency * in_merge.astype(float)  # Drift toward mainline
    
    # Combine
    vx = vx_flow + vx_topo
    vy = vy_flow + vy_topo
    
    return vx, vy, vx_flow, vy_flow, vx_topo, vy_topo


def compute_diffusion_field(occ_mask, X, Y):
    """
    Compute spatially-varying diffusion coefficient.
    
    D(x,t) = D_0 + D_occ * 1_{x ∈ Ω_occ}
    
    Higher diffusion in occluded regions (more uncertainty).
    
    Args:
        occ_mask: Boolean mask of occluded regions
        X, Y: Meshgrid coordinates
    
    Returns:
        D: Diffusion coefficient field
    """
    D = cfg.D0 * np.ones_like(X)
    D[occ_mask] += cfg.D_occ
    D = gaussian_filter(D, sigma=1.0)  # Smooth transitions
    return D


# ============================================================
# PDE SOLVER
# ============================================================

class PDESolver:
    """
    Solver for the advection-diffusion-telegrapher PDE.

    τ∂²R/∂t² + ∂R/∂t + ∇·(vR) = ∇·(D∇R) + Q - λR

    Attributes:
        R: Current risk field
        R_t: Time derivative (for telegrapher term)
        Q_history: Source activity tracker (for activity-based decay)
        Q_ambient: Topology baseline tracker (doesn't contribute to activity)
    """

    def __init__(self):
        self.X, self.Y = cfg.X, cfg.Y
        self.dx, self.dy = cfg.dx, cfg.dy
        self.R = np.zeros_like(self.X)
        self.R_t = np.zeros_like(self.X)

        # NEW: Source activity tracker (for intelligent decay)
        self.Q_history = np.zeros_like(self.X)
        self.Q_ambient = np.zeros_like(self.X)  # Topology baseline (doesn't decay aggressively)
    
    def reset(self):
        """Reset risk field to zero."""
        self.R = np.zeros_like(self.X)
        self.R_t = np.zeros_like(self.X)
        self.Q_history = np.zeros_like(self.X)
        self.Q_ambient = np.zeros_like(self.X)
    
    def step(self, Q, D, vx, vy, dt=0.05, tau=None, Q_ambient=None):
        """
        Advance PDE one time step.

        Uses operator splitting with fixes:
        1. Source + Multi-Level Decay (λ = λ₀ + |v|/L_decay + λ_activity)
        2. Proper Variable Diffusion (∇·(D∇R) with gradient term)
        3. Conservative Advection (flux form ∇·(vR))
        4. Telegrapher damping (consistent update)
        5. Activity-based decay (clears stale risk, preserves topology)

        Args:
            Q: Source field (total, including ambient + vehicle-induced)
            D: Diffusion coefficient field
            vx, vy: Velocity field components
            dt: Time step [s]
            tau: Telegrapher inertia (None uses config value)
            Q_ambient: Ambient topology risk (optional, for activity tracking)

        Returns:
            R: Updated risk field
        """
        if tau is None:
            tau = cfg.tau

        R = self.R.copy()
        R_t = self.R_t.copy()

        # Update ambient topology baseline (if provided)
        if Q_ambient is not None:
            self.Q_ambient = Q_ambient

        # ===================================================================
        # STEP 1: Source + Multi-Level Decay
        # ===================================================================

        # (A) Base flow-dependent decay: λ_base = λ₀ + |v|/L_decay
        speed = np.sqrt(vx**2 + vy**2)
        lambda_field = cfg.lambda_decay + speed / cfg.L_decay

        # (B) Sponge layer at downstream boundary (right edge)
        if hasattr(cfg, 'sponge_length') and cfg.sponge_length > 0:
            x_max = self.X[:, -1].mean()  # Right boundary
            x_sponge_start = x_max - cfg.sponge_length
            w_sponge = np.clip((self.X - x_sponge_start) / cfg.sponge_length, 0, 1)
            lambda_field = lambda_field + cfg.lambda_sponge * w_sponge**2

        # (C) Activity-based decay boost (clears stale vehicle risk, preserves topology)
        if hasattr(cfg, 'tau_source_decay') and cfg.tau_source_decay > 0:
            # Separate vehicle-induced sources from ambient topology
            Q_dynamic = Q - self.Q_ambient  # Vehicle-induced sources only

            # Update activity tracker with exponential memory
            # ∂Q_history/∂t = (Q_dynamic - Q_history) / τ_memory
            tau_src = cfg.tau_source_decay
            self.Q_history = self.Q_history + dt * (Q_dynamic - self.Q_history / tau_src)

            # Compute activity-normalized field (0 = no activity, 1 = high activity)
            Q_max = self.Q_history.max()
            if Q_max > 1e-6:
                activity_norm = np.clip(self.Q_history / Q_max, 0, 1)
            else:
                activity_norm = np.zeros_like(self.Q_history)

            # Decay boost in low-activity regions (clears stale vehicle risk)
            # λ_activity = λ_boost * (1 - activity_norm)
            if hasattr(cfg, 'lambda_activity_boost'):
                lambda_boost = cfg.lambda_activity_boost * (1.0 - activity_norm)
                lambda_field = lambda_field + lambda_boost

        # Apply source and decay
        R = R + dt * (Q - lambda_field * R)

        # ===================================================================
        # STEP 2: Proper Variable Diffusion ∇·(D∇R) = D∇²R + ∇D·∇R
        # ===================================================================
        # Compute gradients of R (central differences)
        dR_dx = np.zeros_like(R)
        dR_dy = np.zeros_like(R)
        dR_dx[:, 1:-1] = (R[:, 2:] - R[:, :-2]) / (2 * self.dx)
        dR_dy[1:-1, :] = (R[2:, :] - R[:-2, :]) / (2 * self.dy)

        # Compute gradients of D
        dD_dx = np.zeros_like(D)
        dD_dy = np.zeros_like(D)
        dD_dx[:, 1:-1] = (D[:, 2:] - D[:, :-2]) / (2 * self.dx)
        dD_dy[1:-1, :] = (D[2:, :] - D[:-2, :]) / (2 * self.dy)

        # Laplacian of R
        laplacian = np.zeros_like(R)
        laplacian[1:-1, 1:-1] = (
            (R[1:-1, 2:] - 2*R[1:-1, 1:-1] + R[1:-1, :-2]) / self.dx**2 +
            (R[2:, 1:-1] - 2*R[1:-1, 1:-1] + R[:-2, 1:-1]) / self.dy**2
        )

        # Full variable diffusion: ∇·(D∇R) = D∇²R + ∇D·∇R
        diffusion_term = D * laplacian + dD_dx * dR_dx + dD_dy * dR_dy

        # Apply
        R = R + dt * diffusion_term

        # ===================================================================
        # STEP 3: Conservative Advection -∇·(vR) using flux form
        # ===================================================================
        # Compute fluxes: F_x = vx * R, F_y = vy * R
        F_x = vx * R
        F_y = vy * R

        # Divergence of flux using upwind scheme
        div_flux = np.zeros_like(R)

        # ∂F_x/∂x (upwind)
        div_flux[:, 1:-1] += np.where(
            vx[:, 1:-1] > 0,
            (F_x[:, 1:-1] - F_x[:, :-2]) / self.dx,
            (F_x[:, 2:] - F_x[:, 1:-1]) / self.dx
        )

        # ∂F_y/∂y (upwind)
        div_flux[1:-1, :] += np.where(
            vy[1:-1, :] > 0,
            (F_y[1:-1, :] - F_y[:-2, :]) / self.dy,
            (F_y[2:, :] - F_y[1:-1, :]) / self.dy
        )

        # Update R
        R = R - dt * div_flux

        # ===================================================================
        # STEP 4: Telegrapher term (wave-like propagation with inertia)
        # ===================================================================
        if tau > 0:
            # Telegrapher equation as first-order system:
            # ∂R/∂t = R_t  (field evolves with velocity R_t)
            # ∂R_t/∂t = (RHS - R_t)/τ  (velocity relaxes to RHS with time constant τ)

            # Compute full RHS using updated R from steps 1-3
            full_rhs = Q - lambda_field * R + diffusion_term - div_flux

            # Steps 1-3 gave us: R_direct = R_old + dt * full_rhs
            # But with telegrapher, we need: R = R_old + dt * R_t_new

            # Store the "instantaneous" update for comparison
            R_instantaneous = R.copy()

            # Update telegrapher velocity toward full_rhs
            R_t = R_t + (dt / tau) * (full_rhs - R_t)

            # Apply telegrapher: replace instantaneous update with damped evolution
            # R should evolve according to R_t, not instantaneously
            R = self.R + dt * R_t

        # ===================================================================
        # Post-processing
        # ===================================================================
        # Enforce non-negativity
        R = np.clip(R, 0, 10)

        # Boundary conditions (zero flux)
        R[0, :] = R[1, :]
        R[-1, :] = R[-2, :]
        R[:, 0] = R[:, 1]
        R[:, -1] = R[:, -2]

        # Optional smoothing (disabled by default)
        if hasattr(cfg, 'post_smooth_sigma') and cfg.post_smooth_sigma > 0:
            R = gaussian_filter(R, sigma=cfg.post_smooth_sigma)

        self.R = R
        self.R_t = R_t if tau > 0 else np.zeros_like(R)

        return self.R.copy()
