# compute ω=∇×u excluding boundaries
import WaterLily: permute,∂
"""
    fill_ω!(ml,u,perdir=();fmm=false)

Fill the multi-level vorticity `ml` with ω=∇×u in the `sources`. With `fmm=true`, level 1 is only
filled within `layer` cells of the faces, where the FMM reads it, and keeps its old values elsewhere.
Level 2 is then summed from `u` directly.
"""
function fill_ω!(ml::Tuple,u,perdir=();fmm=false)
    ω,N = first(ml),size_u(first(ml))[1]
    (fmm && length(ml)>1) || return (fill!(ω,zero(eltype(ω))); fill_ω!(ω,u,perdir); restrict!(ml))
    fill_ω!(ω,u,perdir,layer(N,perdir...))
    curl_restrict!(ml[2],u,sources(N,perdir...)); restrict!(Base.tail(ml))
end
fill_ω!(ω::AbstractArray{<:Any,4},u,perdir=(),w=size(ω)) = @loop nearface(I,ω,w) && (ω[I,1] = centered_curl(1,I,u); ω[I,2] = centered_curl(2,I,u); ω[I,3] = centered_curl(3,I,u)) over I ∈ sources(size_u(ω)[1],perdir...)
fill_ω!(ω::AbstractArray{<:Any,3},u,perdir=(),w=size(ω)) = @loop nearface(I,ω,w) && (ω[I,1] = centered_curl(3,I,u); ω[I,2] = zero(eltype(ω))) over I ∈ sources(size_u(ω)[1],perdir...)
# I is within w[k] cells of a face normal to k
@inline nearface(I::CartesianIndex{n},ω,w) where n = any(ntuple(k->I.I[k]≤w[k] || I.I[k]>size(ω,k)-w[k],n))
Base.@propagate_inbounds centered_curl(i,I,u) = (j=i%3+1; k=(i+1)%3+1; ∂(k,j,I,u)-∂(j,k,I,u))

# Depth of the level-1 sources read by the FMM (`remaining` of the face targets) normal to each face
function layer(N::NTuple{n},d...) where n
    R,S = inside(N),sources(N,d...)
    T(k,s) = CartesianIndex(ntuple(j->j==k ? s : N[j]÷2,n))
    ntuple(k->k∈d ? 0 : max(last(inR(remaining(T(k,1),R,d...),S)).I[k],
                            N[k]+1-first(inR(remaining(T(k,N[k]),R,d...),S)).I[k]),n)
end

# Coarse-level vorticity a[I,i] = Σ ωᵢ over up(I) ∩ S, computed from u
curl_restrict!(a::AbstractArray{<:Any,4},u,S) = @loop (a[I,1] = curl_restrict(1,I,u,S); a[I,2] = curl_restrict(2,I,u,S); a[I,3] = curl_restrict(3,I,u,S)) over I ∈ inside(size_u(a)[1])
curl_restrict!(a::AbstractArray{<:Any,3},u,S) = @loop (a[I,1] = curl_restrict(3,I,u,S); a[I,2] = zero(eltype(a))) over I ∈ inside(size_u(a)[1])
@inline function curl_restrict(i,I,u,S)
    s = zero(eltype(u))
    for J ∈ up(I)
        J ∈ S && (s += @inbounds(centered_curl(i,J,u)))
    end; s
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

using Atomix
# Biot-Savart BCs + residual update
function biotBC_r!(r,u,U,ml,targets,flat_targets;fmm=true,perdir=(),symmetry=())
    fmm ? fmmBC!(ml,targets,flat_targets,perdir,symmetry) : treeBC!(ml,targets[1],perdir) # Fill ml[targets]=uᵥ
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
