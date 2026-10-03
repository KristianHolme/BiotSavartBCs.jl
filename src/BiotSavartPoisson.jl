"""
    BiotSavartPoisson <: WaterLily.AbstractPoisson

Custom Poisson type for Biot-Savart boundary conditions. Wraps a `MultiLevelPoisson`
(which holds all pressure system state) and layers Biot-Savart state on top.

Fields:
- `ml`   : wrapped standard pressure solver
- `ω`    : multi-level vorticity (top level aliases `flow.f`)
- `tar`  : domain boundary target index arrays per multigrid level
- `ftar` : flattened target list for kernel dispatch
- `p`    : pressure solution accumulator
- `fmm`  : use Fast Multi-level Method (`true`) or tree-sum (`false`)
"""
struct BiotSavartPoisson{T,S,V} <: AbstractPoisson{T,S,V}
    ml   :: MultiLevelPoisson{T,S,V}
    ω    :: NTuple
    tar  :: NTuple
    ftar :: AbstractVector
    p    :: AbstractArray
    fmm  :: Bool
    function BiotSavartPoisson(flow; nonbiotfaces=(), fmm=true, mem=Array)
        flow.exitBC && throw(ArgumentError("exitBC=true is ignored when using Biot-Savart BCs"))
        perdir = flow.perdir; isempty(perdir) || (fmm && ndims(flow.p)==3 && length(perdir)==1) ||
            throw(ArgumentError("periodic Biot-Savart BCs require fmm=true, 3D and a single periodic direction"))
        ml = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ; perdir)
        ω  = MLArray(flow.f,perdir...)   # top level aliases flow.f — no copy
        tar  = mem.(collect_targets(ω, (nonbiotfaces...,perdir...,(-).(perdir)...)))
        ftar = flatten_targets(tar)
        p   = copy(flow.p)
        new{eltype(flow.p),typeof(flow.p),typeof(flow.μ₀)}(ml,ω,tar,ftar,p,fmm)
    end
end
WaterLily.update!(b::BiotSavartPoisson) = WaterLily.update!(b.ml)
import WaterLily: div,diagonal,δv,perBC!

"""
    mom_project!(a::AbstractFlow, b::BiotSavartPoisson, w::Int, t, tol=2e-3, itmx=32)

Custom project method for Biot-Savart BCs, applying biot_BC! to update the boundary velocity and residual at each iteration.
Note: a.p is used as the incremental pressure solution for each V-cycle, while b.p accumulates the total pressure solution.
"""
function WaterLily.mom_project!(a::AbstractFlow{N}, b::BiotSavartPoisson, w::Int, t, tol=2e-3,itmx=32) where N
    dt = a.Δt[end]/w; a.p .*= dt  # Scale p *= Δt/w
    U = BCTuple(a.uBC,t,N)        # BC tuple for current time step
    b.p .= 0; project_update!(a,b)                              # Project out initial μ₀∇p
    fill_ω!(b.ω,a.u,a.perdir); biotBC!(a.u,U,b.ω,b.tar,b.ftar;fmm=b.fmm,a.perdir) # Apply domain BCs with fresh ω

    # Set residual
    top = b.ml.levels[1]; top.r .= 0
    Dp = diagonal(top)  # Poisson matrix diagonal
    @inside top.r[I] = ifelse(top.iD[I]==0,0,div(I,a.u)-δv(Dp,I)*div(I,a.V)) # flow divergence, not BDIM-ϵ stretching
    fix_resid!(top.r,a.u,b.tar[1]) # only fix on the boundaries

    r₁tol = WaterLily.l1n_tol(top, tol)
    nᵖ,nᵇ,r₁ = 0,0,WaterLily.L₁(top)
    @log ", $nᵖ, $(WaterLily.L∞(top)), $r₁, $nᵇ\n"
    while nᵖ<itmx
        # V-cycle with fixed BCs until the residual drops >10x
        rtol = max(r₁tol,0.1r₁)
        while nᵖ<itmx
            WaterLily.Vcycle!(b.ml); WaterLily.smooth!(top)
            r₁ = WaterLily.L₁(top); nᵖ+=1
            r₁<rtol && break
        end
        # Update the BCs with Biot-Savart (which requires updating u,p,ω) and repeat until convergence
        project_update!(a,b) # Update u,p
        fill_ω!(b.ω,a.u,a.perdir); biotBC_r!(top.r,a.u,U,b.ω,b.tar,b.ftar;fmm=b.fmm,a.perdir) # Update BC+residual
        r₁ = WaterLily.L₁(top); r∞ = WaterLily.L∞(top); nᵇ+=1
        @log ", $nᵖ, $r∞, $r₁, $nᵇ\n"
        (r₁<r₁tol && r∞<tol) && break
    end
    push!(b.ml.n,nᵖ)
    pflowBC!(a.u)     # Update ghost BCs (domain is already correct)
    perBC!(a.u,a.perdir)
    a.p .= b.p ./ dt  # rescale pressure solution and copy to Flow
end
BCTuple(f::Function,t::T,N) where T = ntuple(i->f(i,zero(SVector{N,T}),t),N)
BCTuple(f::Tuple,t,N) = f

# Apply u-=μ₀∇p & accumulate p
function project_update!(a::AbstractFlow, b::BiotSavartPoisson)
    perBC!(a.p,a.perdir)
    @loop a.u[Ii] -= a.μ₀[Ii]*∂(last(Ii),front(Ii),a.p) over Ii ∈ inside_u(a.u)
    perBC!(a.u,a.perdir) # periodic ghosts for fill_ω!
    b.p .+= a.p; fill!(a.p,0) # accumulate total pressure solution, reset increment
end
