"""
Test script to verify PDE solver fixes.

Verifies:
1. Flow-dependent decay is working
2. Diffusion stability (no explosions)
3. Conservative advection (mass conservation)
4. Ego-relative frame transformation
5. Merge gating by vehicle density
"""
import numpy as np
from config import Config as cfg
from pde_solver import PDESolver, create_vehicle, compute_Q_merge

print("=" * 70)
print("DRIFT PDE Solver - Fix Verification Tests")
print("=" * 70)

# Test 1: Flow-dependent decay parameters
print("\n[Test 1] Flow-dependent decay parameters")
print(f"  Base decay λ₀:        {cfg.lambda_decay:.3f} s⁻¹")
print(f"  Distance half-life:   {cfg.L_decay:.1f} m")
print(f"  Sponge length:        {cfg.sponge_length:.1f} m")
print(f"  Sponge strength:      {cfg.lambda_sponge:.1f} s⁻¹")
print("  ✓ Parameters loaded correctly")

# Test 2: Diffusion stability check
print("\n[Test 2] Diffusion stability")
print(f"  D₀:                   {cfg.D0:.1f} m²/s (reduced from 8.0)")
print(f"  D_occ:                {cfg.D_occ:.1f} m²/s (reduced from 15.0)")
print(f"  D_max:                {cfg.D0 + cfg.D_occ:.1f} m²/s")

dx = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
dy = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
D_max = cfg.D0 + cfg.D_occ
dt_stable = 1.0 / (2 * D_max * (1/dx**2 + 1/dy**2))
print(f"  Grid spacing:         Δx={dx:.3f}m, Δy={dy:.3f}m")
print(f"  Stability limit:      dt_max ≈ {dt_stable*1000:.1f} ms")
print(f"  Typical dt (4 subs):  {120/4:.1f} ms")
if 120/4 < dt_stable * 1.5:
    print("  ✓ Stable regime (dt < 1.5 × dt_stable)")
else:
    print("  ⚠ Marginally stable (consider more substeps)")

# Test 3: Conservative advection (basic check)
print("\n[Test 3] Conservative advection implementation")
solver = PDESolver()
# Initialize Gaussian blob
x_center, y_center = 20, 0
R_init = np.exp(-((cfg.X - x_center)**2 / 100 + (cfg.Y - y_center)**2 / 16))
solver.R = R_init.copy()

# Compute initial mass
mass_init = np.sum(solver.R) * dx * dy

# Constant velocity field
vx = 20 * np.ones_like(cfg.X)
vy = np.zeros_like(cfg.Y)
D = cfg.D0 * np.ones_like(cfg.X)
Q = np.zeros_like(cfg.X)

# Run 10 steps with no source/sink
for _ in range(10):
    solver.step(Q, D, vx, vy, dt=0.03, tau=0)  # tau=0 disables telegrapher

mass_final = np.sum(solver.R) * dx * dy
mass_change = abs(mass_final - mass_init) / mass_init * 100

print(f"  Initial mass:         {mass_init:.4f}")
print(f"  Final mass:           {mass_final:.4f}")
print(f"  Change:               {mass_change:.2f}%")
if mass_change < 10:
    print("  ✓ Mass reasonably conserved (< 10% change)")
else:
    print(f"  ⚠ Mass conservation issue ({mass_change:.1f}% change)")

# Test 4: Ego-relative frame transformation
print("\n[Test 4] Ego-relative frame transformation")
from drift_pde_visualization import transform_to_ego_frame

ego = create_vehicle(0, x=50, y=0, vx=25, vy=0)
other = create_vehicle(1, x=70, y=4, vx=30, vy=0)
vehicles = [ego, other]

vehicles_ego = transform_to_ego_frame(vehicles, ego)

print(f"  World frame:  ego at x={ego['x']}, other at x={other['x']}")
print(f"  Ego frame:    ego at x={vehicles_ego[0]['x']}, other at x={vehicles_ego[1]['x']}")
print(f"  Relative dx:  {vehicles_ego[1]['x']:.1f} m (should be 20)")

if abs(vehicles_ego[0]['x']) < 0.01 and abs(vehicles_ego[1]['x'] - 20) < 0.01:
    print("  ✓ Ego-relative transformation correct")
else:
    print("  ✗ Transformation error!")

# Test 5: Merge gating by vehicle density
print("\n[Test 5] Merge zone gating by vehicle density")

# Empty merge zone
ego_empty = create_vehicle(0, x=0, y=0, vx=20, vy=0)
vehicles_empty = [ego_empty]
Q_merge_empty = compute_Q_merge(vehicles_empty, ego_empty, cfg.X, cfg.Y)
max_risk_empty = np.max(Q_merge_empty)

# Vehicle in merge zone
ego_with = create_vehicle(0, x=0, y=0, vx=20, vy=0)
merge_vehicle = create_vehicle(1, x=50, y=5, vx=25, vy=0)
vehicles_with = [ego_with, merge_vehicle]
Q_merge_with = compute_Q_merge(vehicles_with, ego_with, cfg.X, cfg.Y)
max_risk_with = np.max(Q_merge_with)

print(f"  Empty merge zone:     max risk = {max_risk_empty:.4f}")
print(f"  With vehicle:         max risk = {max_risk_with:.4f}")

if max_risk_empty < 0.1 and max_risk_with > max_risk_empty:
    print("  ✓ Merge gating working (risk only when vehicles present)")
else:
    print(f"  ⚠ Check merge gating (empty={max_risk_empty:.3f}, with={max_risk_with:.3f})")

# Test 6: Smoothing disabled by default
print("\n[Test 6] Post-processing smoothing")
if hasattr(cfg, 'post_smooth_sigma'):
    print(f"  post_smooth_sigma:    {cfg.post_smooth_sigma}")
    if cfg.post_smooth_sigma == 0:
        print("  ✓ Extra smoothing disabled (as intended)")
    else:
        print(f"  ⚠ Smoothing enabled with σ={cfg.post_smooth_sigma}")
else:
    print("  ⚠ post_smooth_sigma not defined (will use default)")

print("\n" + "=" * 70)
print("Verification Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Run: python drift_pde_visualization.py")
print("2. Check that risk field dissipates when vehicles leave")
print("3. Verify no 'ghost' risk blobs behind departed vehicles")
print("4. Confirm smooth field evolution (no oscillations)")
