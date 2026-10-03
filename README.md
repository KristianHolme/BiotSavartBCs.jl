# BiotSavartBCs

[![Build Status](https://github.com/WaterLily-jl/BiotSavartBCs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/WaterLily-jl/BiotSavartBCs.jl/actions/workflows/CI.yml?query=branch%3Amain)

![disk](docs/disk_high_re.png)

This repository defines an extension to the [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) flow solver, adding "external flow" boundary conditions based on the [Biot-Savart equation](https://en.wikipedia.org/wiki/Biot%E2%80%93Savart_law#Aerodynamics_applications). This equation is used to update the velocity *on the boundaries* of the simulation domain based on the vorticity *within* the domain. The resulting boundary conditions are an excellent model for external flow, allowing you to use very small domains around any immersed bodies. [See the paper for the detailed methodology and validation.](https://arxiv.org/abs/2404.09034)

### WaterLily.jl Simulations with BiotSavartBCs.jl

Install the package from the Julia package manager
```julia
julia> ]
pkg> add BiotSavartBCs
```
Using these new boundary conditions within a `WaterLily` simulation is really straightforward; this requires changing only two (😱) lines of code. The first one is obviously

```julia
using WaterLily,BiotSavartBCs
```
The second line that you have to modify creates the Biot-Savart Simulation structure
```julia
sim = BiotSimulation((4L,2L,2L), Ut, L; body=AutoBody(sdf,map), ν=U*L/Re, T, mem=CUDA.CuArray)
```
You can update and plot the simulation structure exactly the same at with a standard WaterLily `Simulation`
```julia
sim_step!(sim, t_end; remeasure::Bool)
```
There are numerous examples in the `examples` folder of this repository that show how to use these new boundary conditions in practice.

### Method

This package takes a practical approach to avoid the two fundamental issues with applying the Biot-Savart equation to set the boundary conditions of a projection-based Navier-Stokes solver: 
 1. A naive weighted sum over the $N_s$ vorticity sources at every cell for all $N_t$ targets at the domain cell faces would make the boundary condition update take $O(N_s N_t)$ operations, making it *orders of magnitude slower* than the rest of the solver. We accelerate the BC update by clustering the vorticity sources using a tree method (oct-tree in 3D and quad-tree in 2D). This reuses the pooling method in WaterLily's Multigrid pressure solver and reduces the cost to $O(\log(N_s) N_t)$. We can further accelerate the BC update by also clustering the target faces - making this an $O(N_t)$ Fast Multi*level* Method FMℓM - a variant of the classic [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method). Finally, we parallelize over all the targets using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) which works on the GPU or multi-threaded CPU.
 2. The pressure projection step depends sensitively on the boundaries conditions, but these *cannot be set* since the unknown pressure generates vorticity on immersed bodies. We solve this problem using a matrix partition method, similar to the approach used for partitioned Fluid-Structure-Interaction (FSI) methods. In practise we see the Multigrid pressure solver actually converges *faster* with `BiotSavartBcs` than with reflection BCs.

The resulting simulation update is very fast, especially with large 3D grids on the GPU - exactly where the ability to use a snug domain is the most important. See the paper for detailed methods, examples, and computational benchmarks. 

### Mixed domain boundary conditions

You can turn off the Biot-Savart update to a domain face by passing the face index to the optional `nonbiotfaces` keyword argument (-3 is the negative z domain face, 2 is the positive y face, etc). In this case, the normal velocity at this face remains zero. Using this we can, for example, model a square plate abutting two slip-walls using:
```julia
function sym_square(N;Re=5e2,mem=Array,U=1,T=Float32,thk=2,L=T(N/2))
    body = AutoBody() do (x,y,z),t
        hypot(x-L,y-min(y,L-thk),z-min(z,L-thk))-thk
    end
    BiotSimulation((2N,N,N), (U,0,0),L;ν=U*2L/Re,body,mem,T,nonbiotfaces=(-2,-3))
end
sim_slip_walls = sym_square(96,mem=CuArray);
sim_step!(sim_slip_walls,2,remeasure=false) # or whatever
```

If we instead want to model a square plate (of twice the width) in an unbounded domain using a symmetric flow condition on the y & z planes, then we _also_ need to add the influence of the images of the vortices to the Biot-Savart boundaries. This is done by overwritting the `symmetry` function before running the simulation:
```julia
import BiotSavartBCs: interaction,symmetry,image
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₂,sgn₂ = image(T,size(ω),-2)  # image target and sign in y
    T₃,sgn₃ = image(T,size(ω),-3)  # image target and sign in z
    T₂₃,_   = image(T₃,size(ω),-2) # image of image!
    # Add up the four contributions
    return interaction(ω,T,args...)+sgn₃*interaction(ω,T₃,args...)+
     sgn₂*(interaction(ω,T₂,args...)+sgn₃*interaction(ω,T₂₃,args...))
end
sim_sym_walls = sym_square(96,mem=CuArray); # no difference!
sim_step!(sim_sym_walls,2,remeasure=false) # BiotBCs now see reflected domain
```

#### Periodic boundary conditions

A 3D simulation can be periodic in one direction, such as the span of a cylinder, by passing `perdir`:
```julia
sim = BiotSimulation((4D,2D,D),(1,0,0),D;body,ν=D/Re,perdir=(3,))
```
The periodic faces use standard periodic conditions and the remaining faces use Biot-Savart conditions which include the periodic images of the vorticity: the nearest image on either side is included in the FMM sum, and the rest are added in closed form at the coarsest level. Since the periodic vorticity induces a nearly 2D velocity field, the FMM uses the 2D-sized near-field window in-plane. This requires `fmm=true` and can't be combined with the symmetry hook above.

### Gallery

Here are a few renderings of the cool things you can do with [`WaterLily.jl`](https://github.com/WaterLily-jl/WaterLily.jl) and these new Biot-Savart BCs

#### Flow behind a square plate at Re=125,000
[![square1](https://img.youtube.com/vi/CNQqI5rRdug/0.jpg)](https://www.youtube.com/shorts/CNQqI5rRdug)

[![square2](https://img.youtube.com/vi/tbf06uhnAEQ/0.jpg)](https://www.youtube.com/shorts/tbf06uhnAEQ)

#### Flow behind a Doritos at Re=25,000
[![doritos](https://img.youtube.com/vi/spFlx2YW0pg/0.jpg)](https://www.youtube.com/shorts/spFlx2YW0pg)

## Reproducing the results

The scripts used to produce the results presented [here](https://arxiv.org/abs/2404.09034) are in the `examples` folder, which has its own `Project.toml` environment (Julia `v1.11` or later). Clone the repository and instantiate that environment
```bash
git clone https://github.com/WaterLily-jl/BiotSavartBCs.jl
cd BiotSavartBCs.jl
julia --project=examples -e "using Pkg; Pkg.instantiate()"
```
and then run any of the example scripts within it, e.g.
```bash
julia --project=examples examples/ImpulsiveCircle.jl
```
The examples are set up to run with `mem=CUDA.CuArray` and require an NVIDIA GPU; use `mem=Array` to run them (slowly) on the CPU. The `examples` environment uses the `BiotSavartBCs` source in this repository, so the scripts always run against the checked-out version of the package.

The plots in the paper are written directly to `tex/fig` by these scripts:

| Script | Paper figures |
|---|---|
| `ImpulsiveCircle.jl` | `ImpCircle_Cd.png`, `ImpCircle_4_vort.png` |
| `DiskTests.jl` | `Disk_force_comparison.png`, `Disk_force_comparison_methods.png` |
| `Sphere.jl` | `drag.png` (and prints the mean drag values quoted in the text) |
| `AirfoilWake.jl` | `CL_mean_deflected_wake.png`, `poincare_deflected_wake.png` |
| `LambDipoleTests.jl` | `lamb_dipole_error_dists.png`, `lamb_dipole_speedup.png` |
| `HillVortexTests.jl` | `Hill_error_dists.png`, `Hill_speedup_dists.png` |

The remaining figures are hand-made from the output of these simulations and cannot be regenerated by a script:
- `flow_disk.svg`: composite of the vorticity snapshots saved by `DiskTests.jl`.
- `vortex_square_plate.svg` and `vortex_square_plate_oblique.svg`: ParaView renders of the λ₂ field written by `DiskTestsHighRe.jl` (whose disk geometry is a square plate), composited with the experimental images of Higuchi et al. (1996).
- `Deflected_wake_snap*.png`: frames of the vorticity animations saved by `AirfoilWake.jl`.
- `sphere3_zoom.png`: a Makie volume rendering of the `Sphere.jl` flow at $tU/R=53$.
- `domain.svg`, `multilevel_domain.svg` and `multilevel_domain_sym.svg`: schematic diagrams drawn in Inkscape.

### Citing

We simply ask you to cite the references below in any publication in which you have made use of the `BiotSavartBCs` project. If you are using other `WaterLily` packages, please cite them as indicated in their repositories.

```bibtex
@article{weymouth2024biot,
    title={Using Biot-Savart boundary conditions for unbounded external flow on Eulerian meshes},
    author={Gabriel D. Weymouth and Marin Lauber},
    year={2024},
    eprint={2404.09034},
    archivePrefix={arXiv},
    primaryClass={physics.flu-dyn}
}
```
