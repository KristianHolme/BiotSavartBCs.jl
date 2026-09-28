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
struct BiotSavartPoisson{T,S,V,W<:Tuple,R<:Tuple,F<:AbstractVector} <: AbstractPoisson{T,S,V}
    ml   :: MultiLevelPoisson{T,S,V}
    ω    :: W
    tar  :: R
    ftar :: F
    p    :: S
    fmm  :: Bool
    function BiotSavartPoisson(flow; nonbiotfaces=(), fmm=true, mem=Array)
        ml = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ; perdir=flow.perdir)
        ω  = MLArray(flow.f)   # top level aliases flow.f — no copy
        tar  = mem.(collect_targets(ω, nonbiotfaces))
        ftar = flatten_targets(tar)
        p   = copy(flow.p)
        new{eltype(flow.p),typeof(flow.p),typeof(flow.μ₀),typeof(ω),typeof(tar),typeof(ftar)}(ml,ω,tar,ftar,p,fmm)
    end
end
WaterLily.update!(b::BiotSavartPoisson) = WaterLily.update!(b.ml)

"""
    mom_project!(a::AbstractFlow, b::BiotSavartPoisson, w, t, tol=2e-3, itmx=32)

Custom project method for Biot-Savart BCs. Solves for pressure with a multigrid V-cycle, applying biot_BC! to update the boundary velocity and residual at each iteration.
Converged when `max|r| < tol` and `mean|r| < tol/10` after a BC update, the same grid-independent criterion as `WaterLily.solver!(::MultiLevelPoisson)`.
Note: a.p is used as the incremental pressure solution for each V-cycle, while b.p accumulates the total pressure solution.
"""
function WaterLily.mom_project!(a::AbstractFlow{N,T}, b::BiotSavartPoisson, w, t, tol=2e-3,itmx=32) where {N,T}
    dt = T(w)*a.Δt[end]; a.p .*= dt  # Scale p *= w*Δt
    U = BCTuple(a.uBC,t,N)        # BC tuple for current time step
    b.p .= 0; project_update!(a,b)                              # Project out initial μ₀∇p
    fill_ω!(b.ω,a.u); biotBC!(a.u,U,b.ω,b.tar,b.ftar;fmm=b.fmm) # Apply domain BCs with fresh ω

    # Set residual
    top = b.ml.levels[1]; top.r .= 0
    @inside top.r[I] = ifelse(top.iD[I]==0,0,WaterLily.div(I,a.u))
    fix_resid!(top.r,a.u,b.tar[1]) # only fix on the boundaries

    r₁tol = WaterLily.l1n_tol(top,tol)
    nᵖ,nᵇ,r∞ = 0,0,WaterLily.L∞(top)
    @log ", $nᵖ, $r∞, $(WaterLily.L₁(top)), $nᵇ\n"
    while nᵖ<itmx
        # V-cycle with fixed BCs until the max residual drops >10x
        rtol = max(tol,0.1r∞)
        while nᵖ<itmx
            WaterLily.Vcycle!(b.ml); WaterLily.smooth!(top)
            r∞ = WaterLily.L∞(top); nᵖ+=1
            r∞<rtol && break
        end
        # Update the BCs with Biot-Savart (which requires updating u,p,ω) and repeat until convergence
        project_update!(a,b) # Update u,p
        fill_ω!(b.ω,a.u); biotBC_r!(top.r,a.u,U,b.ω,b.tar,b.ftar;fmm=b.fmm) # Update BC+residual
        r∞,r₁ = WaterLily.L∞(top),WaterLily.L₁(top); nᵇ+=1
        @log ", $nᵖ, $r∞, $r₁, $nᵇ\n"
        (r∞<tol && r₁<r₁tol) && break
    end
    push!(b.ml.n,nᵖ)
    pflowBC!(a.u)     # Update ghost BCs (domain is already correct)
    a.p .= b.p ./ dt  # rescale pressure solution and copy to Flow
end
BCTuple(f::Function,t::T,N) where T = ntuple(i->f(i,zero(SVector{N,T}),t),N)
BCTuple(f::Tuple,t,N) = f

# Apply u-=μ₀∇p & accumulate p
function project_update!(a::AbstractFlow, b::BiotSavartPoisson)
    @loop a.u[Ii] -= a.μ₀[Ii]*∂(last(Ii),front(Ii),a.p) over Ii ∈ inside_u(a.u)
    b.p .+= a.p; fill!(a.p,0) # accumulate total pressure solution, reset increment
end
