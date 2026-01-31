# DRIFT: Dynamic Risk Inference via Field Transport for highway interactive driving scenarios
This repository is the official implementation of DRIFT.

## Key Features

- **Unified PDE Framework**: Single equation captures advection, diffusion, and wave-like propagation
- **GVF-based Vehicle Interaction**: Anisotropic Gaussian kernels model relative motion risk
- **Occlusion Reasoning**: Shadow regions behind trucks inject latent hazard
- **Merge Topology**: Road geometry creates conflict zones with elevated risk
- **Interpretable Sources**: Clear decomposition into Q_veh, Q_occ, Q_merge

## Mathematical Model

The risk field R(x,t) evolves according to:

```
τ ∂²R/∂t² + ∂R/∂t + ∇·(v_eff R) = ∇·(D∇R) + Q(x,t) - λR
```

where:
- **τ∂²R/∂t²**: Telegrapher inertia (wave-like, finite propagation speed)
- **∂R/∂t**: First-order relaxation
- **∇·(v_eff R)**: Advection by flow + topology drift
- **∇·(D∇R)**: Diffusion (enhanced in occlusion)
- **Q(x,t)**: Source = Q_veh + Q_occ + Q_merge
- **-λR**: Decay/forgetting

<p align="center">
  <img src="output/drift_source_components.png" alt="Source Decomposition" width="900">
</p>

## Source Terms

| Source | Description |
|--------|-------------|
| Q_veh | Vehicle-induced risk using GVF-style anisotropic Gaussians weighted by TTC, relative speed, and vehicle class |
| Q_occ | Occlusion hazard in sensor shadow behind large vehicles; higher at lane centers and truck edges where cut-ins emerge |
| Q_merge | Merge conflict pressure; intensifies toward gore point with topology-driven drift |
