# DRIFT: Dynamic Risk Inference via Field Transport for highway interactive driving scenarios
This repository is the official implementation of DRIFT.

DRIFT is a unified field-theoretic framework for modeling and propagating traffic risk in complex driving scenarios. We formulate risk as a spatiotemporal field R(x,t) governed by an advection-diffusion-telegrapher partial differential equation (PDE), where risk is injected through three physically-motivated source terms: 
- vehicle-induced risk modeled via anisotropic Gaussian kernels derived from Gaussian Vector Fields (GVF), capturing relative motion and interaction intensity;
- occlusion-induced latent hazards representing unobserved threats in sensor shadow regions behind large vehicles such as truck-trailers; and
- merge pressure encoding topological conflict zones where lane-change maneuvers create elevated collision potential. 
The risk field propagates through space via advection along the traffic flow and topology-induced drift, spreads through diffusion representing spatial uncertainty, and exhibits wave-like dynamics through the telegrapher term that enforces finite propagation speed. DRIFT provides a principled, interpretable approach to risk assessment that naturally handles occlusion reasoning, multi-vehicle interactions, and road topology—enabling safer motion planning for autonomous vehicles in challenging scenarios such as highway merging with limited visibility.
