# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3/π/4
@inline weighted(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = (-1)^i*@inbounds(ω[S,1]*r[i%2+1])/(r'*r)/π/2

# The kernel of `weighted` without its constant factor, which `scale` applies once per target
@inline kernel(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = (ir = 1/√(r'*r); permute((j,k)->@inbounds(ω[S,j]*r[k]),i)*(ir*ir*ir))
@inline kernel(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = @inbounds(ω[S,1]*r[i%2+1])/(r'*r)
scale(::Val{3},i) = 1f0/(4f0*Float32(π))
scale(::Val{2},i) = (isodd(i) ? -1f0 : 1f0)/(2f0*Float32(π))

"""
    Lane(p,P)

Lane `p` of the `P` lanes that share one target: it sums the source rows `p, p+P, …` of each
source box. `Lane(1,1)` sums them all.
"""
struct Lane; p::Int; P::Int; end

"""
    boxsum(f,R,H,lane)

Sum `f(S)` over the sources `S ∈ R` that are not in the hole `H`. Each line of `R` along the
first direction is split around `H`, so the inner loop has no branch.
"""
@fastmath @inline function boxsum(f::F,R::CartesianIndices{N},H::CartesianIndices{N},lane::Lane,s) where {F,N}
    x,hx = first(R.indices),first(H.indices)
    rows,hrows = CartesianIndices(Base.tail(R.indices)),CartesianIndices(Base.tail(H.indices))
    for j in lane.p:lane.P:length(rows)
        J = rows[j]
        if !isempty(hx) && J ∈ hrows
            for s1 in first(x):min(last(x),first(hx)-1); s += f(CartesianIndex(s1,J)); end
            for s1 in max(first(x),last(hx)+1):last(x); s += f(CartesianIndex(s1,J)); end
        else
            for s1 in x; s += f(CartesianIndex(s1,J)); end
        end
    end; s
end
@inline nohole(R::CartesianIndices{N}) where N = CartesianIndices(ntuple(_->1:0,N))

# Velocity induced at a target by the sources at one level
@inline induced(ω,Ti::CartesianIndex,l,depth,d::Int...) = induced(ω,Ti,l,depth,Lane(1,1),d...)
Base.@propagate_inbounds @fastmath function induced(ω,Ti::CartesianIndex{Np1},l,depth,lane::Lane,d...) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain,d...),close(T,domain,d...)
    l == depth && (Router = domain)
    # Top level: everything remaining in the source cells. Lower levels: Router minus Rinner
    R,H = l == 1 ? (inR(Router,sources(size_u(ω)[1],d...)),nohole(Router)) : (Router,Rinner)
    scale(Val(N),i)*boxsum(@inline(S->kernel(x-SVector{N,Float32}(S.I),S,i,ω)),R,H,lane,zero(eltype(ω)))
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

"""
    induced!(ml,flat_targets,perdir=(),symmetry=();idx=eachindex(flat_targets),out=nothing,P=lanes(ml,length(idx)),part=…)

Induced velocity on the targets `flat_targets[idx]`, including the images across the `symmetry`
faces. The sums go to the targets in `ml`, or to `out[idx]` when `out` is a vector over
`flat_targets`. `P` lanes share each target and write their partial sums to `part`, which are
then added in a fixed order.
"""
function induced!(ml,flat_targets,perdir=(),symmetry=();idx=eachindex(flat_targets),out=nothing,P=lanes(ml,length(idx)),part=similar(first(ml),length(idx)*P))
    n = length(idx); n == 0 && return
    if P == 1 # no partial sums to add
        isnothing(out) ? (@loop _induced1!(ml,flat_targets,idx,perdir,symmetry,k) over k ∈ 1:n) :
                         (@loop _induced1!(out,ml,flat_targets,idx,perdir,symmetry,k) over k ∈ 1:n)
        return
    end
    @loop _induced!(part,ml,flat_targets,idx,perdir,symmetry,Ij,n,P) over Ij ∈ CartesianIndices((n,P))
    if isnothing(out)
        @loop _gather!(ml,flat_targets,idx,part,k,n,P) over k ∈ 1:n
    else
        @loop _gather!(out,idx,part,k,n,P) over k ∈ 1:n
    end
end
# Lanes per target: many when few targets share the device, one on CPUs
lanes(ml,n) = onCPU(first(ml)) ? 1 : clamp(prevpow(2,max(1,32768÷max(n,1))),1,16)
partlength(nflat) = max(nflat,32768) # ≥ n*lanes(ml,n) for all n ≤ nflat
@inline function _induced!(part,ml,flat_targets,idx,perdir,symmetry,Ij,n,P)
    k,p = Ij.I
    part[k+(p-1)*n] = _sum(ml,flat_targets[idx[k]],perdir,symmetry,Lane(p,P))
end
@inline _sum(ml,(l,T),perdir,symmetry,lane) = isempty(perdir) ? images(ml[l],T,symmetry,l,length(ml),lane) : periodic(ml[l],T,l,length(ml),lane,perdir...)
@inline _induced1!(ml,flat_targets,idx,perdir,symmetry,k) = ((l,T) = flat_targets[idx[k]]; ml[l][T] = _sum(ml,(l,T),perdir,symmetry,Lane(1,1)))
@inline _induced1!(out::AbstractVector,ml,flat_targets,idx,perdir,symmetry,k) = (out[idx[k]] = _sum(ml,flat_targets[idx[k]],perdir,symmetry,Lane(1,1)))
@inline lanesum(part,k,n,P) = (s = zero(eltype(part)); for p in 1:P; s += part[k+(p-1)*n]; end; s)
@inline _gather!(ml,flat_targets,idx,part,k,n,P) = ((l,T) = flat_targets[idx[k]]; ml[l][T] = lanesum(part,k,n,P))
@inline _gather!(out::AbstractVector,idx,part,k,n,P) = (out[idx[k]] = lanesum(part,k,n,P))

# Symmetry planes on domain `faces`: sum the velocity induced at the target and all of its images
@inline images(ω,T,::Tuple{},args...) = induced(ω,T,args...)
@inline function images(ω,T,faces,args...)
    T′,sgn = image(T,size(ω),first(faces))
    images(ω,T,Base.tail(faces),args...)+sgn*images(ω,T′,Base.tail(faces),args...)
end

# Periodic in d: the domain and its nearest images at every level, then the images |n|≥2 at
# the coarsest level using Σ r/|r|³ ≈ ∫ r/|r|³ ds/L (the 2D kernel) minus the middle three periods
@inline periodic(ω,Ti,l,depth,d::Int) = periodic(ω,Ti,l,depth,Lane(1,1),d)
Base.@propagate_inbounds @fastmath function periodic(ω,Ti,l,depth,lane::Lane,d)
    L = size(ω,d)-2; Δ = L*δ(d,Ti)
    val = induced(ω,Ti-Δ,l,depth,lane,d)+induced(ω,Ti,l,depth,lane,d)+induced(ω,Ti+Δ,l,depth,lane,d)
    l < depth && return val
    i,T,e = last(Ti),front(Ti),SVector{3,Float32}(ntuple(k->k==d,3))
    x = shifted(T,i)+SVector{3,Float32}(T.I)
    R = inside(size_u(ω)[1])
    val + boxsum(@inline(S->far(ω,S,x,e,L,i,d)),R,nohole(R),lane,zero(eltype(ω)))
end
Base.@propagate_inbounds @fastmath function far(ω,S,x,e,L,i,d)
    r = x-SVector{3,Float32}(S.I); ρ = r-r[d]*e; ρ² = ρ'*ρ
    F(s) = (ρ*s/ρ²-e)/√(ρ²+s^2) # ∫ r/|r|³ ds
    R = (2ρ/ρ²+F(r[d]-1.5f0L)-F(r[d]+1.5f0L))/L
    permute((j,k)->ω[S,j]*R[k],i)/π/4
end

# Biot-Savart BC using FMM, summing only the targets in `idx`: the rest keep their `own` sums
function fmmBC!(ml,targets,flat_targets,perdir=(),symmetry=();own=nothing,idx=eachindex(flat_targets),kw...)
    induced!(ml,flat_targets,perdir,symmetry;idx,out=own,kw...)
    isnothing(own) || @loop _scatter!(ml,flat_targets,own,k) over k ∈ eachindex(flat_targets)
    project!(ml,targets)
end
@inline _scatter!(ml,flat_targets,own,k) = ((l,T) = flat_targets[k]; ml[l][T] = own[k])
