## Physics-informed Advection-Diffusion for merging traffic with occlusion uncertainty
### the data-driven version is under construction, stay tuned!
This folder contains the sample codes for the DRIFT demonstration with architected physics-based risk field refractor with vehicle-coupled advection and boundary constraints

Features: 
- Local advection
- Smooth diffusion
- Occlusion handling
- Merging pressure
- Fixed plot range

### Usage with default frame rate and step
```shell
python drift_effects_correct.py
```
### Usage with adjustable settings for traffic scenario visualization:
```bash
# Generate GIF animation
python drift_visualization.py --output ./output --frames 60 --fps 6

# Options
#   -o, --output    Output directory (default: ./output)
#   --frames        Number of animation frames (default: 80)
#   --fps           Frames per second (default: 8)
```
### demonstrations of the advection-diffusion effects with merging topology and occlusion using the empirical PDE parameters (pending fine-tuning):
![effect](/assets/drift_advection_correct.gif)
### demonstration of advection-diffusion effects with occlusion-aware risk modeling in occluded zone (view with relative (x, t))
![effect](/assets/drift_advection_pde.gif)
### demonstration of advection-diffusion effects with occlusion-aware risk modeling in occluded zone (view with absolute (x, t))
![effect](/assets/drift_diffusion_pde.gif)
