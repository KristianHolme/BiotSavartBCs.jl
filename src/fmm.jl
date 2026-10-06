# inverse distance weighted source
using StaticArrays
@inline weighted(r::SVector{3,Float32},S::CartesianIndex{3},i,ω) = permute((j,k)->@inbounds(ω[S,j]*r[k]),i)/√(r'*r)^3/π/4
@inline weighted(r::SVector{2,Float32},S::CartesianIndex{2},i,ω) = (-1)^i*@inbounds(ω[S,1]*r[i%2+1])/(r'*r)/π/2

# Velocity induced at a target by the sources at one level
Base.@propagate_inbounds function induced(ω,Ti::CartesianIndex{Np1},l,depth,d...) where Np1
    i,T,N = last(Ti),front(Ti),Np1-1
    x = shifted(T,i)+SVector{N,Float32}(T.I)
    domain = inside(size_u(ω)[1])
    Router,Rinner = remaining(T,domain,d...),close(T,domain,d...)
    l == depth && (Router = domain)
    # Top level: do everything remaining in the source cells
    l == 1 && return boxsum(ω,x,i,inR(Router,sources(size_u(ω)[1],d...)),CartesianIndices(ntuple(_->1:0,N)))
    Rinner == Router && return zero(eltype(ω))
    boxsum(ω,x,i,Router,Rinner)
end
shifted(T::CartesianIndex{N},i) where N = SVector{N,Float32}(ntuple(j-> j==i ? (T.I[i]==1 ? 0.5 : -0.5) : 0,N))

# Sum of `weighted` over the sources in R excluding the hole H. Lines along the first
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

# Induced velocity on targets, including the images across the `symmetry` faces
induced!(ml,flat_targets,perdir=(),symmetry=()) = @loop _induced!(ml,lT,perdir,symmetry) over lT ∈ flat_targets
@inline _induced!(ml,lT,perdir,symmetry) = ((l,T) = lT; ml[l][T] = isempty(perdir) ? images(ml[l],T,symmetry,l,length(ml)) : periodic(ml[l],T,l,length(ml),perdir...))

# Symmetry planes on domain `faces`: sum the velocity induced at the target and all of its images
@inline images(ω,T,::Tuple{},args...) = induced(ω,T,args...)
@inline function images(ω,T,faces,args...)
    T′,sgn = image(T,size(ω),first(faces))
    images(ω,T,Base.tail(faces),args...)+sgn*images(ω,T′,Base.tail(faces),args...)
end

# Periodic in d: the domain and its nearest images at every level, then the images |n|≥2 at
# the coarsest level using Σ r/|r|³ ≈ ∫ r/|r|³ ds/L (the 2D kernel) minus the middle three periods
Base.@propagate_inbounds @fastmath function periodic(ω,Ti,l,depth,d)
    L = size(ω,d)-2; Δ = L*δ(d,Ti)
    val = induced(ω,Ti-Δ,l,depth,d)+induced(ω,Ti,l,depth,d)+induced(ω,Ti+Δ,l,depth,d)
    l < depth && return val
    i,T,e = last(Ti),front(Ti),SVector{3,Float32}(ntuple(k->k==d,3))
    x = shifted(T,i)+SVector{3,Float32}(T.I)
    for S in inside(size_u(ω)[1])
        r = x-SVector{3,Float32}(S.I); ρ = r-r[d]*e; ρ² = ρ'*ρ
        F(s) = (ρ*s/ρ²-e)/√(ρ²+s^2) # ∫ r/|r|³ ds
        R = (2ρ/ρ²+F(r[d]-1.5f0L)-F(r[d]+1.5f0L))/L
        val += permute((j,k)->ω[S,j]*R[k],i)/π/4
    end; val
end

# Biot-Savart BC using FMM
fmmBC!(ml,targets,flat_targets,perdir=(),symmetry=()) = (induced!(ml,flat_targets,perdir,symmetry);project!(ml,targets))
