# DRIFT: Dynamic Risk Inference via Field Transport for highway interactive driving scenarios
This repository is the official implementation of DRIFT.

## Key Features

- **Unified PDE Framework**: Single equation captures advection, diffusion, and wave-like propagation
- **GVF-based Vehicle Interaction**: Anisotropic Gaussian kernels model relative motion risk
- **Occlusion Reasoning**: Shadow regions behind trucks inject latent hazard
- **Merge Topology**: Road geometry creates conflict zones with elevated risk
- **Interpretable Sources**: Clear decomposition into Q_veh, Q_occ, Q_merge

## Source Terms

| Source | Description |
|--------|-------------|
| Q_veh | Vehicle-induced risk using GVF-style anisotropic Gaussians weighted by TTC, relative speed, and vehicle class |
| Q_occ | Occlusion hazard in sensor shadow behind large vehicles; higher at lane centers and truck edges where cut-ins emerge |
| Q_merge | Merge conflict pressure; intensifies toward gore point with topology-driven drift |

## Dataset and Data Engineering
see another repo: https://github.com/PeterWANGHK/Benchmark-RiskField.git

## Demonstration example:

### single vehicle field propagation (with truck occlusion effects)
![Individual effect](/assets/drift_singlecar.gif)

### Group field propagation:
![Group effect](/assets/drift_advection_effect.gif)

### Group field propagation with occlusion-aware and merging pressure:
![Group effect_occlusion_merging](/assets/drift_advection_effect_v2.gif)
