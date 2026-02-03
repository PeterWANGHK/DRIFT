"""
DRIFT: Dynamic Risk Inference via Field Transport
-------------------------------------------------
UPDATED VERSION with:
1. Safety Controller - Distance thresholds prevent unrealistic collisions
2. Hidden Merge Scenario - Ego in adjacent lane, merge car hidden by truck
3. Lane-Change Parallel Risk - Elevated risk when vehicles become parallel

Scenario: Ego in fast lane passes truck while hidden merge vehicle enters mainline.
Creates realistic safety-critical situation with occlusion-induced risk.
"""

import os
import copy
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy.random as npr
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from config import Config as cfg
from pde_solver import PDESolver, compute_velocity_field, compute_Q_merge_decomposed

# HJ Reachability for principled occlusion risk
try:
    from hj_reachability import HJOcclusionRisk, HJConfig, compute_occlusion_risk_hj
    HAS_HJ = True
except ImportError:
    HAS_HJ = False
    print("Note: hj_reachability not available, using heuristic occlusion risk")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio")


# =============================================================================
# Global HJ Solver (precomputed once at module load)
# =============================================================================

HJ_SOLVER = None

def get_hj_solver():
    """Get or create the global HJ solver."""
    global HJ_SOLVER
    if HJ_SOLVER is None and HAS_HJ:
        print("Initializing HJ Reachability solver (one-time computation)...")
        HJ_SOLVER = HJOcclusionRisk(precompute=True)
    return HJ_SOLVER


# =============================================================================
# Safety Controller (prevents unrealistic collisions)
# =============================================================================

class SafetyThresholds:
    """Distance and safety thresholds for realistic vehicle behavior."""
    min_gap_highway = 25.0       # At highway speeds [m]
    min_gap_urban = 12.0         # At urban speeds [m]
    min_gap_critical = 5.0       # Absolute minimum [m]
    time_headway_safe = 2.0      # 2-second rule [s]
    time_headway_alert = 1.5     # Alert [s]
    ttc_warning = 4.0            # Warning [s]
    ttc_critical = 2.5           # Critical [s]
    ttc_emergency = 1.5          # Emergency [s]
    comfortable_decel = -2.5     # [m/s²]
    firm_decel = -4.5            # [m/s²]
    emergency_decel = -7.5       # [m/s²]


class SafetyController:
    """Controller that enforces safety constraints on vehicle motion."""
    
    def __init__(self, thresholds=None):
        self.th = thresholds or SafetyThresholds()
    
    def compute_ttc(self, gap, rel_speed):
        """Compute Time-To-Collision."""
        if rel_speed <= 0.1:
            return float('inf')
        return gap / rel_speed
    
    def compute_safe_gap(self, speed):
        """Compute safe following gap based on speed."""
        return max(speed * self.th.time_headway_safe, self.th.min_gap_highway)
    
    def get_safe_acceleration(self, ego, lead, desired_accel):
        """Compute safe acceleration considering vehicle ahead."""
        if lead is None:
            return desired_accel
        
        gap = lead["x"] - ego["x"] - lead["length"]/2 - ego["length"]/2
        rel_speed = ego["vx"] - lead["vx"]
        safe_gap = self.compute_safe_gap(ego["vx"])
        ttc = self.compute_ttc(gap, rel_speed)
        
        if ttc < self.th.ttc_emergency or gap < self.th.min_gap_critical:
            return self.th.emergency_decel
        if ttc < self.th.ttc_critical or gap < self.th.min_gap_urban:
            return self.th.firm_decel
        if gap < safe_gap * 0.8:
            return self.th.comfortable_decel
        if gap < safe_gap:
            return min(desired_accel, 0.0)
        return desired_accel
    
    def find_lead_vehicle(self, ego, vehicles, lane_tolerance=None):
        """Find the lead vehicle in ego's lane."""
        if lane_tolerance is None:
            lane_tolerance = cfg.lane_width / 2
        
        lead = None
        min_gap = float('inf')
        
        for v in vehicles:
            if v is None or v["id"] == ego["id"]:
                continue
            if abs(v["y"] - ego["y"]) > lane_tolerance:
                continue
            gap = v["x"] - ego["x"]
            if gap > 0 and gap < min_gap:
                min_gap = gap
                lead = v
        return lead


# =============================================================================
# Visualization colors
# =============================================================================

BG_DARK = "#0D1117"
BG_PANEL = "#161B22"
SHADOW_COLOR = "#4A4A4A"
ROAD_COLOR = "#1F2937"


def get_grid():
    """Use the global grid from config."""
    return cfg.X, cfg.Y


def build_diffusion_field(occ_mask, base_scale=0.10, occ_scale=1.8, skirt_gain=0.6, 
                          skirt_sigma=3.0, smooth_sigma=0.65):
    """Build spatially varying diffusion coefficient."""
    D = cfg.D0 * base_scale * np.ones_like(cfg.X)
    if occ_mask is not None and occ_mask.any():
        core = gaussian_filter(occ_mask.astype(float), sigma=1.2)
        skirt = gaussian_filter(occ_mask.astype(float), sigma=skirt_sigma)
        D += cfg.D_occ * occ_scale * core
        D += cfg.D_occ * skirt_gain * skirt
    return gaussian_filter(D, sigma=smooth_sigma)


def step_with_cfl(solver, Q, D, vx, vy, dt_frame, min_substeps=3, cfl=0.9):
    """
    Advance the PDE while respecting the Courant limit for advection.
    Falls back to additional sub-steps when vehicle speeds are high so
    the field moves smoothly with the traffic instead of lagging.
    """
    max_vx = float(np.max(np.abs(vx)))
    max_vy = float(np.max(np.abs(vy)))

    # If essentially static, a single step is fine
    if max_vx < 1e-8 and max_vy < 1e-8:
        solver.step(Q, D, vx, vy, dt=dt_frame)
        return

    cfl_x = max_vx * dt_frame / cfg.dx
    cfl_y = max_vy * dt_frame / cfg.dy
    required = max(cfl_x, cfl_y) / cfl

    substeps = max(min_substeps, int(np.ceil(required)))
    sub_dt = dt_frame / substeps

    for _ in range(substeps):
        solver.step(Q, D, vx, vy, dt=sub_dt)


# =============================================================================
# Vehicle utilities
# =============================================================================

def create_vehicle(vid, x, y, vx, vy, vclass="car", length=None, width=None, **kwargs):
    """Create vehicle dictionary."""
    if length is None:
        if vclass == "truck":
            length = cfg.truck_length
        elif "truck_trailer" in vclass or "trailer" in vclass:
            length = cfg.truck_length * 1.5
        else:
            length = cfg.car_length
    if width is None:
        width = cfg.truck_width if "truck" in vclass else cfg.car_width
    heading = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0
    
    v = {
        "id": vid,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "heading": heading,
        "class": vclass,
        "length": length,
        "width": width,
    }
    v.update(kwargs)
    return v


def move_vehicle(v, dt, ax=0, ay=0):
    """Move vehicle forward in time; clamp to roadway bounds."""
    if v is None:
        return None
    new_x = v["x"] + v["vx"] * dt + 0.5 * ax * dt**2
    new_y = v["y"] + v["vy"] * dt + 0.5 * ay * dt**2
    new_vx = v["vx"] + ax * dt
    new_vy = v["vy"] + ay * dt

    new_y = np.clip(new_y, cfg.lane_centers[0] - 1, cfg.lane_centers[-1] + 1)

    if new_x > cfg.x_max - 2:
        new_x = cfg.x_min + 5
    if new_x < cfg.x_min + 1:
        new_x = cfg.x_min + 5

    return create_vehicle(v["id"], new_x, new_y, new_vx, new_vy, v["class"],
                         v["length"], v["width"], **{k: v[k] for k in v if k not in
                         ["id", "x", "y", "vx", "vy", "heading", "class", "length", "width"]})


def transform_to_ego_frame(vehicles, ego):
    """
    Transform vehicle positions to ego-relative coordinates.

    In ego-relative frame:
    - Ego is at origin (x=0, y=0)
    - Grid coordinates X, Y represent positions relative to ego
    - Velocities remain absolute (world frame)

    This fixes the frame mismatch issue: grid is ego-relative but we need
    vehicle positions in the same frame for correct risk field computation.

    Args:
        vehicles: List of vehicle dicts (world coordinates)
        ego: Ego vehicle dict (world coordinates)

    Returns:
        vehicles_ego: List of vehicle dicts (ego-relative coordinates)
    """
    if ego is None:
        return vehicles

    vehicles_ego = []
    for v in vehicles:
        if v is None:
            vehicles_ego.append(None)
            continue

        # Transform to ego-relative coordinates
        v_rel = v.copy()
        v_rel['x'] = v['x'] - ego['x']  # Relative x position
        v_rel['y'] = v['y'] - ego['y']  # Relative y position
        # Velocities remain in world frame (absolute)
        # This is correct because compute_velocity_field uses relative velocities internally

        vehicles_ego.append(v_rel)

    return vehicles_ego


# =============================================================================
# Hidden Merge Scenario
# =============================================================================

def create_hidden_merge_scenario():
    """
    Create the "Hidden Merge" scenario.
    
    Setup:
    - Ego in Lane 0 (fast lane), accelerating to pass truck
    - Truck in Lane 1 (middle lane), slower - OCCLUDER
    - Merging vehicle on ramp, hidden by truck
    - Blocker in Lane 2 forces merge vehicle toward Lane 1
    """
    vehicles = [
        # EGO: Fast lane (Lane 0), accelerating
        create_vehicle(0, -20.0, cfg.lane_centers[0], 24.0, 0, "car", name="EGO", target_speed=30.0, ax=1.0),
        
        # TRUCK: Middle lane (Lane 1), slower - OCCLUDER
        create_vehicle(1, 10.0, cfg.lane_centers[1], 18.0, 0, "truck_trailer",
                      name="TRUCK", target_speed=18.0),
        
        # MERGING VEHICLE: On ramp, HIDDEN by truck
        create_vehicle(2, cfg.merge_x_start + 8, cfg.lane_centers[-1], 20.0, -0.6, "car", name="MERGE", target_lane=cfg.lane_centers[1], is_merging=True),
        
        # BLOCKER: Lane 2, blocks merge vehicle's first choice
        #create_vehicle(3, cfg.merge_x_start + 30, cfg.lane_centers[2], 19.0, 0, "car", name="BLOCKER", target_speed=19.0),
        
        # TRAFFIC: Additional vehicle in Lane 0 ahead (gives ego something to track)
        #create_vehicle(4, 50.0, cfg.lane_centers[0], 27.0, 0, "car", name="TRAFFIC", target_speed=27.0),
    ]
    
    return vehicles


def update_hidden_merge_vehicles(vehicles, dt, t, safety_ctrl):
    """Update vehicles for hidden merge scenario with safety constraints."""
    updated = []

    # Safe access with bounds checking
    ego = vehicles[0] if len(vehicles) > 0 else None
    truck = vehicles[1] if len(vehicles) > 1 else None
    merge_veh = vehicles[2] if len(vehicles) > 2 else None
    blocker = vehicles[3] if len(vehicles) > 3 else None

    # Early return if essential vehicles don't exist
    if ego is None:
        return vehicles
    
    for v in vehicles:
        if v is None:
            updated.append(None)
            continue
        
        new_v = v.copy()
        
        if v["id"] == 0:  # EGO
            lead = safety_ctrl.find_lead_vehicle(ego, vehicles)
            desired_ax = v.get("ax", 1.0)
            if v["vx"] >= v.get("target_speed", 30.0):
                desired_ax = 0.0
            safe_ax = safety_ctrl.get_safe_acceleration(v, lead, desired_ax)
            
            new_v["vx"] = max(10.0, v["vx"] + safe_ax * dt)
            new_v["x"] = v["x"] + v["vx"] * dt + 0.5 * safe_ax * dt**2
            new_v["ax_actual"] = safe_ax
        
        elif v["id"] == 1:  # TRUCK
            new_v["x"] = v["x"] + v["vx"] * dt
        
        elif v["id"] == 2:  # MERGE VEHICLE
            target_lane_y = v.get("target_lane", cfg.lane_centers[1])
            in_merge = cfg.merge_x_start < v["x"] < cfg.merge_x_end + 25
            
            if in_merge:
                if blocker and abs(blocker["x"] - v["x"]) < 25:
                    target_lane_y = cfg.lane_centers[1]
                
                dy = target_lane_y - v["y"]
                if abs(dy) > 0.3:
                    new_v["vy"] = np.sign(dy) * min(1.0, abs(dy) / dt * 0.25)
                else:
                    new_v["vy"] = 0
                    new_v["y"] = target_lane_y
            else:
                new_v["vy"] = 0
            
            new_v["x"] = v["x"] + v["vx"] * dt
            new_v["y"] = v["y"] + new_v["vy"] * dt
            new_v["y"] = np.clip(new_v["y"], cfg.lane_centers[0] - 1, cfg.lane_centers[-1] + 1)
            
            if new_v["vx"] != 0 or new_v["vy"] != 0:
                new_v["heading"] = np.arctan2(new_v["vy"], new_v["vx"])
        
        else:  # Other vehicles
            new_v["x"] = v["x"] + v["vx"] * dt
            new_v["y"] = v["y"] + v.get("vy", 0) * dt
        
        updated.append(new_v)
    
    return updated


def is_vehicle_occluded(ego, target, occluder):
    """Check if target vehicle is occluded from ego's view by occluder."""
    if target is None or occluder is None:
        return False
    
    dx_occ = occluder["x"] - ego["x"]
    dy_occ = occluder["y"] - ego["y"]
    dx_tgt = target["x"] - ego["x"]
    dy_tgt = target["y"] - ego["y"]
    
    if dx_occ < -5 or dx_tgt < dx_occ - occluder["length"]/2:
        return False
    
    occ_half_w = occluder["width"] / 2 + 2.0
    
    if dy_occ > 0:
        return dy_tgt > dy_occ - occ_half_w and dx_tgt > dx_occ
    else:
        return dy_tgt < dy_occ + occ_half_w and dx_tgt > dx_occ


# =============================================================================
# Occlusion geometry
# =============================================================================

def compute_shadow(ego, truck, X, Y):
    """Compute occlusion shadow polygon and mask."""
    if truck is None or ego is None:
        return None, np.zeros_like(X, dtype=bool)

    dx = truck["x"] - ego["x"]
    dy = truck["y"] - ego["y"]
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 3 or dx < -5:
        return None, np.zeros_like(X, dtype=bool)

    L, W = truck["length"], truck["width"]
    corners_local = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])

    h = truck["heading"]
    cos_h, sin_h = np.cos(h), np.sin(h)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (rot @ corners_local.T).T + np.array([truck["x"], truck["y"]])

    angles = np.arctan2(corners[:, 1] - ego["y"], corners[:, 0] - ego["x"])
    left_idx, right_idx = np.argmax(angles), np.argmin(angles)
    left_corner, right_corner = corners[left_idx], corners[right_idx]

    shadow_length = 45
    left_dir = (left_corner - np.array([ego["x"], ego["y"]])) / (np.linalg.norm(left_corner - np.array([ego["x"], ego["y"]])) + 1e-6)
    right_dir = (right_corner - np.array([ego["x"], ego["y"]])) / (np.linalg.norm(right_corner - np.array([ego["x"], ego["y"]])) + 1e-6)

    left_far = left_corner + left_dir * shadow_length
    right_far = right_corner + right_dir * shadow_length

    shadow_polygon = np.array([left_corner, left_far, right_far, right_corner])

    path = MplPath(shadow_polygon)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(points).reshape(X.shape)

    return shadow_polygon, mask


# =============================================================================
# Source terms (become Q for PDE)
# =============================================================================

def compute_risk_from_vehicles(vehicles, X, Y):
    """Risk Gaussian centered on each vehicle."""
    R = np.zeros_like(X)
    for v in vehicles:
        if v is None:
            continue
        weight = 1.5 if "truck" in v["class"] else 1.0
        h = v["heading"]
        cos_h, sin_h = np.cos(h), np.sin(h)
        dX = X - v["x"]
        dY = Y - v["y"]
        dX_rot = cos_h * dX + sin_h * dY
        dY_rot = -sin_h * dX + cos_h * dY
        sigma_x = cfg.sigma_x * (1 + v["length"] / 12)
        sigma_y = cfg.sigma_y * 0.6 * (1 + v["width"] / 6)
        gaussian = np.exp(-0.5 * (dX_rot**2 / sigma_x**2 + dY_rot**2 / sigma_y**2))
        R += weight * gaussian
    return R


def _smoothstep(s):
    s_clamped = np.clip(s, 0, 1)
    return 3 * s_clamped**2 - 2 * s_clamped**3


def compute_merge_risk(vehicles, X, Y, merging_id=None):
    """Merge conflict source localized to ramp/mainline interface."""
    R_merge = np.zeros_like(X)
    merge_len = cfg.merge_x_end - cfg.merge_x_start
    decay_len = 40.0

    for v in vehicles:
        if v is None:
            continue

        is_merge_car = merging_id is not None and v.get("id") == merging_id
        in_ramp_band = abs(v["y"] - cfg.lane_centers[-1]) < cfg.lane_width
        in_target_band = abs(v["y"] - cfg.lane_centers[-2]) < cfg.lane_width
        in_merge_x = (cfg.merge_x_start - 5) < v["x"] < (cfg.merge_x_end + decay_len)
        if not ((in_ramp_band or (is_merge_car and in_target_band)) and in_merge_x):
            continue

        dX = X - v["x"]
        dY = Y - v["y"]

        ahead = np.exp(-0.5 * ((dX - 8) ** 2 / 140))

        target_y = cfg.lane_centers[-2]
        mix = np.clip((v["x"] - (cfg.merge_x_end - 8)) / 12.0, 0, 1)
        aim_y = (1 - mix) * v["y"] + mix * target_y
        lateral = np.exp(-0.5 * ((Y - aim_y) ** 2 / 14))

        progress = (v["x"] - cfg.merge_x_start) / merge_len
        progress = np.clip(progress, 0, 1.2)
        rise = _smoothstep(progress)

        if merging_id is not None and v.get("id") == merging_id:
            if v["x"] <= cfg.merge_x_end:
                decay = 1.0
            else:
                over = v["x"] - cfg.merge_x_end
                decay = np.exp(-over / decay_len)
            gain = 1.6 * rise * decay
        else:
            gain = 0.6 * (0.4 + 0.6 * rise)

        handoff = 1.0
        if v["x"] > cfg.merge_x_end:
            over = v["x"] - cfg.merge_x_end
            handoff = np.exp(-over / (decay_len * 0.6))

        R_merge += gain * ahead * lateral * handoff

    if R_merge.max() > 0:
        R_merge = gaussian_filter(R_merge, sigma=1.6)
    return R_merge


def compute_occlusion_risk(shadow_mask, X, Y, truck, ego=None, merge_vehicle=None, enable_gating=True):
    """
    Risk in occlusion zone - uses HJ reachability when available.

    HJ-based risk provides formal safety guarantees:
    - V(z,T) ≤ 0 means collision reachable within horizon T
    - Depends on ego speed and braking capability
    - Sharp "unsafe boundary" rather than smooth blob

    Falls back to heuristic if HJ not available.

    NEW: Strong gating - only inject source if vehicle actually in/near shadow.
         Diffusion (D_occ) still operates to spread existing risk from nearby vehicles.

    Args:
        shadow_mask: Boolean mask of occluded region
        X, Y: Meshgrid coordinates
        truck: Truck/occluder vehicle dict
        ego: Ego vehicle dict
        merge_vehicle: Merging vehicle dict (for gating)
        enable_gating: If True, gates source by vehicle presence

    Returns:
        R_occ: Occlusion risk source (strongly gated if enabled)
    """
    R_occ = np.zeros_like(X)
    if shadow_mask is None or not shadow_mask.any() or truck is None:
        return R_occ

    # ===== STRONG GATING: Only inject if vehicle in/near shadow =====
    gate = 1.0  # Default: full injection

    if enable_gating and merge_vehicle is not None:
        merge_x, merge_y = merge_vehicle['x'], merge_vehicle['y']

        # Check if merge vehicle is in shadow region
        # Find nearest grid indices
        iy = np.argmin(np.abs(Y[:, 0] - merge_y))
        ix = np.argmin(np.abs(X[0, :] - merge_x))

        # Bounds check
        iy = np.clip(iy, 0, shadow_mask.shape[0] - 1)
        ix = np.clip(ix, 0, shadow_mask.shape[1] - 1)

        merge_in_shadow = shadow_mask[iy, ix]

        # Distance to nearest shadow point
        if shadow_mask.any():
            shadow_y, shadow_x = np.where(shadow_mask)
            shadow_points_x = X[shadow_y, shadow_x]
            shadow_points_y = Y[shadow_y, shadow_x]

            dists = np.sqrt(
                (shadow_points_x - merge_x)**2 +
                (shadow_points_y - merge_y)**2
            )
            min_dist_to_shadow = dists.min() if len(dists) > 0 else np.inf
        else:
            min_dist_to_shadow = np.inf

        # Gating function: 1.0 in shadow, exponential decay outside
        gate_distance = 20.0  # [m] - gate width

        if merge_in_shadow:
            gate = 1.0  # Full injection
        else:
            # Exponential decay with distance from shadow
            gate = np.exp(-min_dist_to_shadow / gate_distance)

        # Strong threshold: effectively zero beyond gate distance
        if gate < 0.05:
            gate = 0.0  # No source injection if vehicle far from shadow

    elif enable_gating:
        # No merge vehicle info - minimal ambient occlusion risk
        gate = cfg.k_occ_ambient  # Small baseline (e.g., 0.05)

    # If gate is zero, return immediately (no source injection)
    if gate < 1e-6:
        return R_occ

    # ===== COMPUTE OCCLUSION RISK (with gate multiplier) =====

    # Try HJ-based risk first (principled, guarantee-backed)
    if HAS_HJ and ego is not None:
        hj_solver = get_hj_solver()
        if hj_solver is not None:
            R_occ = hj_solver.compute_risk(
                ego, X, Y, shadow_mask,
                lane_centers=cfg.lane_centers
            )

            # Distance modulation (closer to truck = higher uncertainty)
            dist = np.sqrt((X - truck['x'])**2 + (Y - truck['y'])**2)
            dist_factor = 0.4 + 0.6 * np.exp(-dist / 35)
            R_occ = R_occ * dist_factor

            # Soft boundary
            shadow_soft = gaussian_filter(shadow_mask.astype(float), sigma=1.0)
            R_occ = R_occ * shadow_soft

            # Apply gate
            return R_occ * gate

    # Fallback: heuristic occlusion risk
    dist = np.sqrt((X - truck["x"]) ** 2 + (Y - truck["y"]) ** 2)

    p_lane = np.zeros_like(X)
    for lane_y in cfg.lane_centers:
        p_lane += np.exp(-0.5 * ((Y - lane_y) / 1.0) ** 2)
    p_lane = p_lane / (p_lane.max() + 1e-6)

    p_dist = np.exp(-((dist - 18) ** 2) / 220)

    core = shadow_mask.astype(float) * 1.1 * p_lane * p_dist
    halo = gaussian_filter(shadow_mask.astype(float), sigma=2.5) * 0.6 * p_lane

    R_occ = (core + halo) * gate  # Apply gate
    return R_occ


def compute_lc_parallel_risk(ego, vehicles, X, Y):
    """
    Compute lane-change collision risk when vehicles are parallel.
    HIGH risk when ego and another vehicle (especially merge car) are:
    - In adjacent lanes
    - Longitudinally parallel (overlapping x-positions)
    """
    R_lc = np.zeros_like(X)
    
    for v in vehicles:
        if v is None or v["id"] == ego["id"]:
            continue
        
        dy = abs(v["y"] - ego["y"])
        dx = v["x"] - ego["x"]
        
        # Adjacent lanes: between 0.5 and 2 lane widths apart
        is_adjacent = cfg.lane_width * 0.4 < dy < cfg.lane_width * 2.2
        
        # Parallel: overlapping in x
        overlap_threshold = max(ego["length"], v["length"]) * 1.8
        is_parallel = abs(dx) < overlap_threshold
        
        # Approaching parallel
        rel_vx = v["vx"] - ego["vx"]
        will_be_parallel = (dx * rel_vx < 0 and abs(dx) < overlap_threshold * 2.5)
        
        if is_adjacent and (is_parallel or will_be_parallel):
            proximity = 1.0 - abs(dx) / (overlap_threshold * 2)
            proximity = np.clip(proximity, 0, 1)
            
            is_merging = v.get("is_merging", False)
            merge_factor = 1.8 if is_merging else 1.0
            
            vy_toward_ego = -np.sign(v["y"] - ego["y"]) * v.get("vy", 0)
            lateral_factor = 1.0 + 0.6 * max(0, vy_toward_ego)
            
            intensity = 1.3 * proximity * merge_factor * lateral_factor
            
            mid_x = (ego["x"] + v["x"]) / 2
            mid_y = (ego["y"] + v["y"]) / 2
            
            sigma_x = max(abs(dx) / 2 + 4, 6)
            sigma_y = max(dy / 2 + 1, 2.5)
            
            dist_sq = (X - mid_x)**2 / sigma_x**2 + (Y - mid_y)**2 / sigma_y**2
            R_lc += intensity * np.exp(-0.5 * dist_sq)
            
            ahead_x = max(ego["x"], v["x"]) + 10
            dist_ahead = (X - ahead_x)**2 / 120 + (Y - mid_y)**2 / 15
            R_lc += 0.5 * intensity * np.exp(-0.5 * dist_ahead)
    
    if R_lc.max() > 0:
        R_lc = gaussian_filter(R_lc, sigma=1.8)
    
    return R_lc


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_road(ax):
    ax.fill_between([cfg.x_min, cfg.x_max], 
                   [cfg.lane_centers[0] - cfg.lane_width / 2] * 2, 
                   [cfg.lane_centers[-1] + cfg.lane_width / 2] * 2, 
                   color=ROAD_COLOR, alpha=0.3, zorder=0)
    for i, lane_y in enumerate(cfg.lane_centers):
        if i == 0:
            ax.axhline(y=lane_y - cfg.lane_width / 2, color="white", linestyle="-", linewidth=2, alpha=0.8)
        if i == len(cfg.lane_centers) - 1:
            ax.axhline(y=lane_y + cfg.lane_width / 2, color="white", linestyle="-", linewidth=2, alpha=0.8)
        elif i < len(cfg.lane_centers) - 1:
            ax.axhline(y=lane_y + cfg.lane_width / 2, color="white", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(x=cfg.merge_x_start, color="yellow", linestyle=":", lw=1.5, alpha=0.5)
    ax.axvline(x=cfg.merge_x_end, color="red", linestyle=":", lw=2, alpha=0.7)


def draw_shadow(ax, shadow_polygon, alpha=0.45):
    if shadow_polygon is None:
        return
    patch = plt.Polygon(shadow_polygon, facecolor=SHADOW_COLOR, alpha=alpha, 
                       edgecolor="red", linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(patch)


def draw_vehicle(ax, v, is_ego=False, is_merging=False, is_occluded=False):
    if v is None:
        return
    x, y = v["x"], v["y"]
    L, W = v["length"], v["width"]
    
    if "truck" in v["class"]:
        gap = 0.8
        cab_len = min(4.5, max(3.0, 0.32 * L))
        trailer_len = max(0.6 * L, L - cab_len - gap)
        trailer = mpatches.Rectangle((x - L / 2, y - W / 2), trailer_len, W, 
                                     facecolor="orange", edgecolor="black", linewidth=1.5, zorder=10)
        ax.add_patch(trailer)
        cab = mpatches.Rectangle((x - L / 2 + trailer_len + gap, y - W / 2), cab_len, W, 
                                 facecolor="darkblue", edgecolor="black", linewidth=1.5, zorder=11)
        ax.add_patch(cab)
    else:
        if is_ego:
            color = "#2ECC71"
            ec = "yellow"
            lw = 2.5
        elif is_merging:
            color = "#E74C3C" if is_occluded else "#F39C12"
            ec = "red" if is_occluded else "white"
            lw = 2.0
        else:
            color = "#3498DB"
            ec = "white"
            lw = 1.0
        rect = mpatches.FancyBboxPatch((x - L / 2, y - W / 2), L, W, 
                                       boxstyle="round,pad=0.1", facecolor=color, 
                                       edgecolor=ec, linewidth=lw, zorder=10)
        ax.add_patch(rect)
    
    speed = np.sqrt(v["vx"] ** 2 + v["vy"] ** 2)
    if is_ego:
        label = "EGO"
    elif is_merging:
        label = "MERGE\n(HIDDEN)" if is_occluded else "MERGE"
    else:
        label = v.get("name", f"{speed:.0f} m/s")
    
    ax.text(x, y - W / 2 - 1.4, label, ha="center", va="top", fontsize=7, 
           color="yellow" if is_ego else ("red" if is_occluded else "white"), 
           fontweight="bold", 
           bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7))


def enforce_min_gap(vehicles, min_gap=10.0):
    """Push following vehicles back to maintain a longitudinal gap."""
    lanes = cfg.lane_centers
    half = cfg.lane_width / 2
    for lane_y in lanes:
        lane_veh = [v for v in vehicles if v and abs(v["y"] - lane_y) <= half]
        lane_veh.sort(key=lambda v: v["x"], reverse=True)
        for ahead, behind in zip(lane_veh, lane_veh[1:]):
            if behind["x"] > ahead["x"] - min_gap:
                new_x = ahead["x"] - min_gap
                behind["x"] = new_x
                behind["vx"] = min(behind["vx"], ahead["vx"])


# =============================================================================
# Animations powered by PDE solver
# =============================================================================

def create_advection_animation(output_path="./output", n_frames=70, fps=8):
    """
    Advection + diffusion using the PDE solver.
    Uses Hidden Merge scenario with safety controller.
    """
    print("\n" + "=" * 70)
    print("DRIFT: ADVECTION (PDE-driven) - Hidden Merge Scenario")
    print("Ego in fast lane, merge car hidden by truck")
    print("=" * 70)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create hidden merge scenario
    vehicles = create_hidden_merge_scenario()
    safety_ctrl = SafetyController()
    merging_id = 2  # Merge vehicle ID

    solver = PDESolver()
    dt_frame = 0.12
    substeps = 3
    frames = []

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(BG_DARK)

    X, Y = get_grid()

    # Warm start (in world frame)
    ego = vehicles[0]
    truck = vehicles[1]
    shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)
    Q0 = (compute_risk_from_vehicles(vehicles, X, Y) +
          compute_merge_risk(vehicles, X, Y, merging_id=merging_id) +
          compute_occlusion_risk(shadow_mask, X, Y, truck, ego=ego))
    solver.R = gaussian_filter(Q0, sigma=0.8) + 0.4 * gaussian_filter(shadow_mask.astype(float), sigma=2.0)

    print(f"Generating {n_frames} frames...")

    for step in range(n_frames):
        if step % 10 == 0:
            print(f"  Frame {step}/{n_frames}")
        
        fig.clf()
        
        ego = vehicles[0]
        truck = vehicles[1]
        merging = vehicles[merging_id] if merging_id < len(vehicles) else None

        # Check if merge vehicle is occluded
        merge_occluded = is_vehicle_occluded(ego, merging, truck)

        # Compute risk sources (in world frame)
        R_veh = compute_risk_from_vehicles(vehicles, X, Y)

        # NEW: Decomposed merge risk (ambient + vehicle-induced)
        Q_merge_ambient, Q_merge_vehicle = compute_Q_merge_decomposed(vehicles, ego, X, Y)

        shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)

        # NEW: Strongly gated occlusion risk (only if vehicle in shadow)
        R_occ = compute_occlusion_risk(shadow_mask, X, Y, truck, ego=ego,
                                       merge_vehicle=merging, enable_gating=True)

        R_lc = compute_lc_parallel_risk(ego, vehicles[1:], X, Y)

        # Combine sources - LC risk important when merge car visible and parallel
        # Note: Merge is now split into ambient (persistent) + vehicle (decays)
        lc_weight = 0.3 if merge_occluded else 0.9
        Q_total = (Q_merge_ambient +           # Persistent topology risk
                   0.9 * Q_merge_vehicle +     # Vehicle-induced merge risk (decays)
                   gaussian_filter(R_veh + 1.05 * R_occ, sigma=0.20) +  # Vehicle + occlusion
                   lc_weight * R_lc)           # Lane-change risk

        # Build diffusion field (enhanced in shadow ALWAYS - allows spreading without injection)
        D = build_diffusion_field(shadow_mask, base_scale=0.10, occ_scale=1.4, skirt_gain=0.5, smooth_sigma=0.6)

        # Velocity field uses absolute velocities (world frame)
        vx, vy, *_ = compute_velocity_field(vehicles, ego, X, Y)
        vy = vy * 0.5

        # Evolve PDE with ambient topology baseline for activity-based decay
        sub_dt = dt_frame / substeps
        for _ in range(substeps):
            solver.step(Q_total, D, vx, vy, dt=sub_dt, Q_ambient=Q_merge_ambient)

        R_display = gaussian_filter(solver.R, sigma=0.40)

        # Create figure with two panels
        gs = matplotlib.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.15)
        
        # Main panel: Risk field
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor(BG_PANEL)

        vmax = 2.2
        pcm = ax.pcolormesh(X, Y, R_display, cmap="jet", shading="gouraud", vmin=0, vmax=vmax)

        max_val = R_display.max()
        lower = 0.25
        upper = min(max_val, vmax) * 0.9
        if max_val > lower and upper > lower:
            levels = np.linspace(lower, upper, 5)
            ax.contour(X, Y, R_display, levels=levels, colors="white", linewidths=1, alpha=0.6)

        draw_road(ax)
        draw_shadow(ax, shadow_poly, alpha=0.4)

        # Draw vehicles in world frame
        for v in vehicles:
            if v is None:
                continue
            is_merge = v["id"] == merging_id
            is_occ = is_merge and merge_occluded
            draw_vehicle(ax, v, is_ego=(v["id"] == 0), is_merging=is_merge, is_occluded=is_occ)

        skip = 10
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], vx[::skip, ::skip], vy[::skip, ::skip], 
                 color="cyan", scale=250, alpha=0.35, width=0.003, zorder=2)

        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.y_min, cfg.y_max)
        ax.set_aspect("equal")
        ax.set_title(f"ADVECTION (PDE) - Hidden Merge | t = {step*dt_frame:.2f}s | Step {step+1}", 
                    fontsize=13, fontweight="bold", color="white")
        ax.set_xlabel("x [m]", color="white", fontsize=11)
        ax.set_ylabel("y [m]", color="white", fontsize=11)
        ax.tick_params(colors="white")

        # Status info
        occ_status = "HIDDEN" if merge_occluded else "VISIBLE"
        occ_color = "red" if merge_occluded else "lime"
        ax.text(cfg.x_min + 3, cfg.y_max - 1.2, 
               f"Merge vehicle: {occ_status}", fontsize=10, color=occ_color, fontweight="bold",
               bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

        # Ego speed and safety info
        ego_speed = ego["vx"]
        ax_actual = ego.get("ax_actual", 0)
        ax.text(cfg.x_min + 3, cfg.y_max - 3.0,
               f"Ego: {ego_speed:.1f} m/s | accel: {ax_actual:.1f} m/s²", 
               fontsize=9, color="white",
               bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Risk Value", color="white", fontsize=10)
        cbar.ax.tick_params(colors="white")

        # Info panel
        ax_info = fig.add_subplot(gs[1])
        ax_info.set_facecolor(BG_PANEL)
        ax_info.axis('off')
        
        # Build merge vehicle info with None check
        if merging is not None:
            merge_status = "HIDDEN by truck!" if merge_occluded else "VISIBLE"
            merge_color = "red" if merge_occluded else "yellow"
            merge_info = f"""Merge ({merge_color}):
  {merge_status}
  Position: ({merging["x"]:.0f}, {merging["y"]:.1f})
  Speed: {merging["vx"]:.1f} m/s"""
        else:
            merge_info = "Merge: N/A"

        info_text = f"""Hidden Merge Scenario
━━━━━━━━━━━━━━━━━━━━
Time: {step*dt_frame:.1f}s

Ego (green):
  Lane 0 (fast)
  Speed: {ego["vx"]:.1f} m/s
  Accel: {ax_actual:.1f} m/s²

Truck (orange):
  Lane 1 (middle)
  Speed: {truck["vx"]:.1f} m/s
  OCCLUDER

{merge_info}

LC Risk: {"HIGH" if R_lc.max() > 0.5 else "LOW"}
━━━━━━━━━━━━━━━━━━━━
Safety Controller: ACTIVE
HJ Reachability: {"ON" if HAS_HJ else "OFF"}
"""
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=9, color="white", verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.9))

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())

        # Update vehicles with safety controller
        vehicles = update_hidden_merge_vehicles(vehicles, dt_frame, step * dt_frame, safety_ctrl)
        enforce_min_gap(vehicles, min_gap=12.0)

    plt.close()
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, "drift_advection_pde.gif")
        print(f"\nSaving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"Saved: {gif_path}")
        return gif_path
    return None


def create_diffusion_animation(output_path="./output", n_frames=50, fps=8):
    """
    Diffusion-focused animation driven by PDE solver.
    Uses the same Hidden Merge scenario but focuses on diffusion visualization.
    """
    print("\n" + "=" * 70)
    print("DRIFT: DIFFUSION (PDE-driven) - Occlusion Spreading")
    print("=" * 70)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vehicles = create_hidden_merge_scenario()
    safety_ctrl = SafetyController()

    ego = vehicles[0]
    truck = vehicles[1]

    X, Y = get_grid()

    # Compute shadow in world frame
    shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)

    solver = PDESolver()
    dt_frame = 0.08
    substeps = 4
    frames = []

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(BG_DARK)

    # Seed the field (in world frame)
    R_veh0 = compute_risk_from_vehicles(vehicles, X, Y)
    R_occ0 = compute_occlusion_risk(shadow_mask, X, Y, truck, ego=ego,
                                    merge_vehicle=vehicles[2], enable_gating=True)
    Q_merge_ambient0, Q_merge_vehicle0 = compute_Q_merge_decomposed(vehicles, ego, X, Y)
    solver.R = gaussian_filter(2.0 * R_occ0 + 0.2 * R_veh0 + 0.8 * (Q_merge_ambient0 + Q_merge_vehicle0), sigma=0.65)

    print(f"Generating {n_frames} frames...")

    for step in range(n_frames):
        if step % 10 == 0:
            print(f"  Frame {step}/{n_frames}")
        
        fig.clf()

        ego = vehicles[0]
        truck = vehicles[1]
        merging = vehicles[2] if len(vehicles) > 2 else None
        merge_occluded = is_vehicle_occluded(ego, merging, truck)

        # Compute risk sources in world frame WITH DECOMPOSITION
        R_veh = compute_risk_from_vehicles(vehicles, X, Y)

        # NEW: Decomposed merge risk (ambient + vehicle-induced)
        Q_merge_ambient, Q_merge_vehicle = compute_Q_merge_decomposed(vehicles, ego, X, Y)

        # NEW: Strongly gated occlusion risk
        R_occ = compute_occlusion_risk(shadow_mask, X, Y, truck, ego=ego,
                                       merge_vehicle=merging, enable_gating=True)

        R_lc = compute_lc_parallel_risk(ego, vehicles[1:], X, Y)

        Q_total = (1.6 * R_occ +
                   Q_merge_ambient +           # Persistent topology risk
                   1.05 * Q_merge_vehicle +    # Vehicle-induced merge risk (decays)
                   0.2 * gaussian_filter(R_veh, sigma=0.25) +
                   0.5 * R_lc)

        D = build_diffusion_field(shadow_mask, base_scale=0.10, occ_scale=1.8, skirt_gain=0.6, smooth_sigma=0.6)

        # Velocity field uses absolute velocities (world frame)
        vx, vy, *_ = compute_velocity_field(vehicles, ego, X, Y)

        sub_dt = dt_frame / substeps
        for _ in range(substeps):
            solver.step(Q_total, D, vx, vy, dt=sub_dt, Q_ambient=Q_merge_ambient)

        R_vis = gaussian_filter(solver.R, sigma=0.35)
        vmax = 1.2

        gs = matplotlib.gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25, height_ratios=[1, 1.2])

        # Panel 1: D field
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(BG_PANEL)
        pcm1 = ax1.pcolormesh(X, Y, D, cmap="Greens", shading="gouraud", vmin=cfg.D0, vmax=cfg.D0 + cfg.D_occ)
        ax1.contour(X, Y, shadow_mask.astype(float), levels=[0.5], colors="red", linewidths=2, linestyles="--")
        draw_road(ax1)
        # Draw vehicles in world frame
        for v in vehicles:
            draw_vehicle(ax1, v, is_ego=(v["id"] == 0), is_merging=(v["id"] == 2),
                        is_occluded=(v["id"] == 2 and merge_occluded))
        ax1.set_xlim(cfg.x_min, cfg.x_max)
        ax1.set_ylim(cfg.y_min, cfg.y_max)
        ax1.set_aspect("equal")
        ax1.set_title("Diffusion Coefficient D(x)", fontsize=11, fontweight="bold", color="white")
        ax1.tick_params(colors="white")
        cbar1 = plt.colorbar(pcm1, ax=ax1, shrink=0.7)
        cbar1.set_label("D [m²/s]", color="white")
        cbar1.ax.tick_params(colors="white")

        # Panel 2: Shadow + LC risk
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(BG_PANEL)
        
        # Show LC risk overlay
        lc_cmap = LinearSegmentedColormap.from_list("lc", [(0, "#161B22"), (0.3, "#4A2D4A"), (0.7, "#9A4A4A"), (1, "#FF4444")])
        ax2.pcolormesh(X, Y, R_lc, cmap=lc_cmap, shading="gouraud", alpha=0.8, vmin=0, vmax=1.5)
        
        draw_road(ax2)
        draw_shadow(ax2, shadow_poly, alpha=0.45)
        # Draw vehicles in world frame
        for v in vehicles:
            draw_vehicle(ax2, v, is_ego=(v["id"] == 0), is_merging=(v["id"] == 2),
                        is_occluded=(v["id"] == 2 and merge_occluded))
        ax2.set_xlim(cfg.x_min, cfg.x_max)
        ax2.set_ylim(cfg.y_min, cfg.y_max)
        ax2.set_aspect("equal")
        ax2.set_title("Lane-Change Parallel Risk", fontsize=11, fontweight="bold", color="white")
        ax2.tick_params(colors="white")

        # Panel 3: Risk field (PDE)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor(BG_PANEL)
        pcm3 = ax3.pcolormesh(X, Y, R_vis, cmap="jet", shading="gouraud", vmin=0, vmax=vmax)
        max_val = R_vis.max()
        lower = 0.15
        upper = min(max_val, vmax) * 0.9
        if max_val > lower and upper > lower:
            levels = np.linspace(lower, upper, 6)
            ax3.contour(X, Y, R_vis, levels=levels, colors="white", linewidths=0.8, alpha=0.6)
        ax3.contour(X, Y, shadow_mask.astype(float), levels=[0.25], colors="lime", linewidths=2.2, linestyles="--", alpha=0.9)
        draw_road(ax3)
        draw_shadow(ax3, shadow_poly, alpha=0.35)
        # Draw vehicles in world frame
        for v in vehicles:
            draw_vehicle(ax3, v, is_ego=(v["id"] == 0), is_merging=(v["id"] == 2),
                        is_occluded=(v["id"] == 2 and merge_occluded))
        ax3.set_xlim(cfg.x_min, cfg.x_max)
        ax3.set_ylim(cfg.y_min, cfg.y_max)
        ax3.set_aspect("equal")
        
        occ_status = "HIDDEN" if merge_occluded else "VISIBLE"
        ax3.set_title(f"Risk via PDE Diffusion | t = {step*dt_frame:.2f}s | Merge: {occ_status}", 
                     fontsize=12, fontweight="bold", color="white")
        ax3.set_xlabel("x [m]", color="white", fontsize=11)
        ax3.set_ylabel("y [m]", color="white", fontsize=11)
        ax3.tick_params(colors="white")
        cbar3 = plt.colorbar(pcm3, ax=ax3, shrink=0.65, pad=0.02)
        cbar3.set_label("Risk (Uncertainty)", color="white", fontsize=10)
        cbar3.ax.tick_params(colors="white")

        fig.suptitle("DRIFT: PDE-driven Diffusion - Hidden Merge Scenario", 
                    fontsize=14, fontweight="bold", color="white", y=0.98)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())

        # Update vehicles (slower for diffusion demo)
        vehicles = update_hidden_merge_vehicles(vehicles, dt_frame * 0.5, step * dt_frame, safety_ctrl)
        
        # Recompute shadow as vehicles move
        shadow_poly, shadow_mask = compute_shadow(vehicles[0], vehicles[1], X, Y)

    plt.close()
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, "drift_diffusion_pde.gif")
        print(f"\nSaving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"Saved: {gif_path}")
        return gif_path
    return None


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DRIFT: PDE-based Advection/Diffusion with Hidden Merge")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--frames", type=int, default=70, help="Number of frames")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")

    args = parser.parse_args()

    create_advection_animation(args.output, n_frames=args.frames, fps=args.fps)
    create_diffusion_animation(args.output, n_frames=args.frames, fps=args.fps)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Output saved to: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
