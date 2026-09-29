# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3/π/4
@inline weighted(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = (-1)^i*@inbounds(ω[S,1]*r[i%2+1])/(r'*r)/π/2

# Sum over sources at one interaction level
Base.@propagate_inbounds function interaction(ω,Ti::CartesianIndex{Np1},l,depth) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain),close(T,domain)
    l == depth && (Router = domain)
    # Top level: do everything remaining inside buff=2
    l == 1 && return boxsum(ω,x,i,inR(Router,inside(size_u(ω)[1],buff=2)),CartesianIndices(ntuple(_->1:0,N)))
    Rinner == Router && return zero(eltype(ω))
    boxsum(ω,x,i,Router,inR(Rinner,Router))
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Sum of `weighted` over the sources in R excluding the hole H ⊆ R. Lines along the first
# (contiguous) dimension are split around H so the inner loop has no branches and vectorizes.
@inline boxsum(ω,x,i,R,H) = i==1 ? boxsum(ω,x,Val(1),R,H) : i==2 ? boxsum(ω,x,Val(2),R,H) : boxsum(ω,x,Val(3),R,H)
@fastmath @inline function boxsum(ω,x::SVector{N},i::Val,R,H) where N
    s,hole = zero(eltype(ω)),!isempty(H)
    a,b = first(R.indices[1]),last(R.indices[1])
    ha,hb = first(H.indices[1]),last(H.indices[1])
    for J in CartesianIndices(Base.tail(R.indices))
        cut = hole && J ∈ CartesianIndices(Base.tail(H.indices))
        s += xline(ω,x,J,i,a,cut ? min(b,ha-1) : b)
        cut && (s += xline(ω,x,J,i,max(a,hb+1),b))
    end
    return s/(2(N-1)*Float32(π))
end
@fastmath @inline function xline(ω,x::SVector{3},J,::Val{i},a,b) where i
    j,k = i%3+1,(i+1)%3+1
    ry,rz,s = x[2]-J[1],x[3]-J[2],zero(eltype(ω))
    @inbounds @simd for xs in a:b
        r = SVector(x[1]-xs,ry,rz); r² = r'*r
        s += (ω[xs,J,j]*r[k]-ω[xs,J,k]*r[j])*inv(r²*√r²)
    end; s
end
@fastmath @inline function xline(ω,x::SVector{2},J,::Val{i},a,b) where i
    ry,s = x[2]-J[1],zero(eltype(ω))
    @inbounds @simd for xs in a:b
        r = SVector(x[1]-xs,ry)
        s += ω[xs,J,1]*r[i%2+1]*inv(r'*r)
    end; i==1 ? -s : s
end

# Interaction on targets
interaction!(ml,flat_targets) = @vecloop _interaction!(ml,lT) over lT ∈ flat_targets
@inline _interaction!(ml,lT) = ((l,T) = lT; ml[l][T] = symmetry(ml[l],T,l,length(ml)))
@inline symmetry(ω,T,args...) = interaction(ω,T,args...) # default is no applied symmetry

# Biot-Savart BC using FMM
fmmBC!(ml,targets,flat_targets) = (interaction!(ml,flat_targets);project!(ml,targets))
