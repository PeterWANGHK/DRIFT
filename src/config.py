"""
PDE Risk Field Configuration
Advection-Diffusion-Telegrapher Model for Traffic Risk Propagation
"""
import numpy as np


class Config:
    """Configuration for the PDE risk field solver."""
    
    # ============ Spatial Domain ============
    # NOTE: Grid is in WORLD-FIXED coordinates
    # Vehicles travel through the static grid (e.g., x=0 to x=100)
    x_min, x_max = -30, 200   # Longitudinal range [m] in world frame
    y_min, y_max = -15, 15    # Lateral range [m] in world frame
    nx, ny = 150, 70          # Grid resolution
    
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # ============ PDE Parameters ============
    # Diffusion (reduced for numerical stability)
    D0 = 1.0              # Base diffusion coefficient [m²/s]
    D_occ = 3.0           # Additional diffusion in occluded regions [m²/s]

    # Decay (optimized for smooth temporal clearance)
    lambda_decay = 0.15   # Base decay rate [1/s] (3× faster than before)
    L_decay = 30.0        # Distance half-life [m] - tighter spatial decay

    # Activity-based decay boost (clears stale vehicle risk, preserves topology)
    lambda_activity_boost = 0.4   # Additional decay in low-activity regions [1/s]
    tau_source_decay = 2.5        # Source memory time constant [s]

    # Sponge layer (absorbing boundary to prevent reflections)
    sponge_length = 15.0  # [m] - width of absorbing boundary layer
    lambda_sponge = 1.5   # [1/s] - more aggressive absorption at boundaries

    # Telegrapher (wave-like propagation)
    # Set tau=0 to disable (recommended for testing)
    # Set tau=0.2-0.5 to enable smooth wave-like propagation with inertia
    tau = 0.0             # Inertia/reaction time [s] (0 = disabled)

    # Post-processing
    post_smooth_sigma = 0.0   # Smoothing sigma (0.0 = disabled; use 0.1-0.2 for viz only)
    
    # ============ Source Parameters ============
    # GVF-style Gaussian kernel
    sigma_x = 12.0        # Longitudinal spread [m]
    sigma_y = 3.0         # Lateral spread [m]

    # Merge risk decomposition (ambient topology + vehicle-induced)
    k_merge_ambient = 0.08    # Ambient topology risk coefficient (persistent)
    k_merge_vehicle = 0.6     # Vehicle-induced merge risk coefficient (decays)

    # Occlusion risk gating
    k_occ_ambient = 0.05      # Ambient occlusion risk when no vehicle in shadow (minimal)
    
    # ============ Merge Zone ============
    merge_x_start = 30.0  # Merge zone start [m]
    merge_x_end = 70.0    # Merge zone end (gore point) [m]
    merge_y_ramp = 6.0    # Ramp lane y-position [m]
    
    # ============ Lane Geometry ============
    lane_width = 4.0
    lane_centers = [-4.0, 0.0, 4.0, 8.0]  # Lane center y-positions
    
    # ============ Vehicle Dimensions ============
    car_length = 5.0
    car_width = 2.0
    truck_length = 12.0
    truck_width = 2.5
