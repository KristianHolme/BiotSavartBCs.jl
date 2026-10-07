# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
fill_ω!(ml::Tuple,u,perdir=()) = (ω=first(ml); fill!(ω,zero(eltype(ω))); fill_ω!(ω,u,perdir); restrict!(ml))
fill_ω!(ω::AbstractArray,u,perdir::Tuple=()) = fill_ω!(ω,u,sources(size_u(ω)[1],perdir...))
fill_ω!(ω::AbstractArray{<:Any,4},u,R::CartesianIndices,op=(ω,c)->c) = @loop (ω[I,1] = op(ω[I,1],centered_curl(1,I,u)); ω[I,2] = op(ω[I,2],centered_curl(2,I,u)); ω[I,3] = op(ω[I,3],centered_curl(3,I,u))) over I ∈ R
fill_ω!(ω::AbstractArray{<:Any,3},u,R::CartesianIndices,op=(ω,c)->c) = @loop (ω[I,1] = op(ω[I,1],centered_curl(3,I,u)); ω[I,2] = zero(eltype(ω))) over I ∈ R
Base.@propagate_inbounds centered_curl(i,I,u) = (j=i%3+1; k=(i+1)%3+1; ∂(k,j,I,u)-∂(j,k,I,u))

# Replace ω (holding the previous ∇×u) by Δω=∇×u-ω in the level-1 box `B`, and restrict Δω
Δω!(ml::Tuple,u,B) = (fill_ω!(first(ml),u,B,(ω,c)->c-ω); restrict!(ml,B))

# Box of the level-1 cells whose ω can change under u-=μ₀∇p. The centred curl of a staggered
# gradient is zero, so ω only changes where the curl stencil (±1 cell) reads a face with μ₀≠1.
# The normal faces on the domain boundary are not read. Periodic directions are kept whole.
using Atomix
function ωbox(μ₀,perdir=())
    N,n = size_u(μ₀)
    b = similar(μ₀,Int32,2n); copyto!(b,[fill(typemax(Int32),n);fill(typemin(Int32),n)])
    @loop _ωbox!(b,μ₀,I,perdir) over I ∈ inside(N)
    b = Array(b); lo,hi = b[1:n],b[n+1:2n]
    R = sources(N,perdir...)
    any(lo .> hi) && return R # no body
    inR(CartesianIndex(ntuple(d->d∈perdir ? 1 : lo[d]-1,n)...):CartesianIndex(ntuple(d->d∈perdir ? N[d] : hi[d]+1,n)...),R)
end
@inline function _ωbox!(b,μ₀,I::CartesianIndex{n},perdir) where n
    hit = false
    for i ∈ 1:n
        hit |= μ₀[I,i]≠1 && (I.I[i]>2 || i∈perdir)
    end
    hit && for d ∈ 1:n # check before the atomics to avoid contention
        I.I[d]<b[d] && Atomix.@atomic b[d] min Int32(I.I[d])
        I.I[d]>b[n+d] && Atomix.@atomic b[n+d] max Int32(I.I[d])
    end
end

# Incompressible & irrotational ghosts
function pflowBC!(u)
    N,n = size_u(u)
    @inline edge(I,j,val) = 2<I.I[j]<N[j] ? val : zero(val)
    for i ∈ 1:n # we know this is slow on GPUs!!
        for j ∈ 1:n # Tangential direction ghosts, curl=0
            j==i && continue
            @loop u[I,j] = u[I+δ(i,I),j] - edge(I,j,∂(j,CartesianIndex(I+δ(i,I),i),u)) over I ∈ slice_u(N,i,j,1)
            @loop u[I,j] = u[I-δ(i,I),j] + edge(I,j,∂(j,CartesianIndex(I,i),u)) over I ∈ slice_u(N,i,j,N[i])
        end # Normal direction ghosts, div=0
        @loop u[I,i] += WaterLily.div(I,u) over I ∈ WaterLily.slice(N.-1,1,i,2)
    end
end
slice_u(N::NTuple{n},i,j,s) where n = CartesianIndices(ntuple(k-> k==i ? (s:s) : k==j ? (2:N[k]) : (2:N[k]-1),n))

# Biot-Savart BCs
function biotBC!(u,U,ml,targets,flat_targets;fmm=true,perdir=(),symmetry=())
    fmm ? fmmBC!(ml,targets,flat_targets,perdir,symmetry) : treeBC!(ml,targets[1],perdir) # Fill ml[targets]=uᵥ
    @vecloop _biotBC!(u,U,ml[1],Ii) over Ii ∈ targets[1]           # Set u = uᵥ+U
end
@inline function _biotBC!(u,U,uᵥ,Ii)
    i,I = last(Ii),front(Ii); lower = I.I[i]==1
    u[I+(lower ? δ(i,I) : zero(I)),i] = U[i]+uᵥ[Ii]
end

# Biot-Savart BCs + residual update
function biotBC_r!(r,u,U,ml,targets,flat_targets,B...;fmm=true,perdir=(),symmetry=())
    fmm ? fmmBC!(ml,targets,flat_targets,perdir,symmetry,B...) : treeBC!(ml,targets[1],perdir) # Fill ml[targets]=uᵥ
    @vecloop _biotBC_r!(r,u,U,ml[1],Ii) over Ii ∈ targets[1]       # Update the u,r
    fix_resid!(r,u,targets[1])                                     # Fix u,r
end
@inline function _biotBC_r!(r,u,U,uᵥ,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    uₙ = U[i]+uᵥ[Ii]
    uI = lower ? Ii+δ(i,Ii) : Ii; uₙ⁰ = u[uI]; u[uI] = uₙ
    Atomix.@atomic r[I+(lower ? δ(i,I) : -δ(i,I))] += (uₙ-uₙ⁰)*(lower ? -1 : 1)
end

# Correct the global residual s.t. sum(r)=0
fix_resid!(r,u,targets,fix=sum(r)/length(targets)) = @vecloop _fix_resid!(r,u,fix,Ii) over Ii ∈ targets
@inline function _fix_resid!(r,u,fix,Ii)
    I,i = front(Ii),last(Ii); lower = I.I[i]==1
    u[I+ (lower ? δ(i,I) : zero(I)),i] += fix*(lower ? 1 : -1)
    Atomix.@atomic r[I+ (lower ? δ(i,I) : -δ(i,I))] -= fix
end
