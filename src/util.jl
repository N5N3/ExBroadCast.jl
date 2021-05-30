# check the backend
using ArrayInterface
const AnyGPU = ArrayInterface.GPU
gpu_copyto!(args...) = throw("You need using CUDA first!")
getdevice(x) = ArrayInterface.device(x)
getdevices(nt::NamedTuple) = nt |> values |> getdevices
getdevices(t::Tuple) = begin
    ds = getdevice.(t)
    d = first(ds)
    all(==(d), Base.tail(ds)) || throw("device error")
    d
end
# force inlined map that return nothing
@inline fmap(f::Op, t₁::Tuple{}) where {Op} = nothing
@inline fmap(f::Op, t₁::Tuple  ) where {Op} = begin
    f(t₁[1])
    fmap(f, Base.tail(t₁))
end

@inline fmap(f::Op, t₁::Tuple{}, t₂::Tuple{}) where {Op} = nothing
@inline fmap(f::Op, t₁::Tuple  , t₂::Tuple  ) where {Op} = begin
    f(t₁[1], t₂[1])
    fmap(f, Base.tail(t₁), Base.tail(t₂))
end


### import many funtions
## for Tuple
import Base.Broadcast: Style, _broadcast_getindex
## for copyto
import Base.Broadcast:  throwdm, AbstractArrayStyle, Unknown, combine_eltypes,
                        preprocess, Broadcasted, DefaultArrayStyle
const FilledBC = Broadcasted{<:AbstractArrayStyle{0}}
import Base.Broadcast:  ischunkedbroadcast, chunkedcopyto!, bitcache_size,
                        dumpbitcache, bitcache_chunks
## for materialize
import Base.Broadcast: materialize!, BroadcastStyle, combine_styles, instantiate

## TupleDummy
import Base: size, axes, setindex!, unalias, mightalias, unaliascopy, IndexStyle, parent
import Base.Broadcast: broadcast_unalias
const All_Linear = true
const Any_Cartesian = false
"""
    TupleDummy(arrays::Tuple)
Dummy structure for broadcast with multiple outputs.
A simplified SoA implementation.
**This is not part of the interface. Not exported.**
"""
struct TupleDummy{T,N,S<:Tuple,L,SS} <: AbstractArray{T,N}
    arrays::S
    ax::SS      # add this field to avoid repeated Base.OneTo()
end
function TupleDummy(arrays::Tuple{AbstractArray,AbstractArray,Vararg{AbstractArray}}) ## at least 2 outputs
    @assert !isempty(arrays)
    siz = size(first(arrays))
    ax = axes(first(arrays))
    for a in Base.tail(arrays)
        @assert size(a) == siz "Incompatible size"
        @assert axes(a) == ax "Incompatible axes"
    end
    LinearFLag = mapreduce((x) -> IndexStyle(x) === IndexLinear(), &, arrays)
    T = Tuple{map(eltype, arrays)...}
    S = Tuple{map(typeof, arrays)...}
    SS = typeof(ax)
    TupleDummy{T,length(siz),S,LinearFLag,SS}(arrays,ax)
end
parent(td::TupleDummy) = td.arrays
size(td::TupleDummy, args...) = size(td.arrays[1], args...)
axes(td::TupleDummy, args...) = td.ax #Fixed: default axes() doesn't support OffsetArray.

const LTD{T,N,S} = TupleDummy{T,N,S,All_Linear}
IndexStyle(td::LTD) = IndexLinear()
@inline setindex!(td::LTD, value::Tuple, ix::Int) =
    fmap(td.arrays, value) do a, v
        @inbounds setindex!(a, v, ix) #add @inbounds
    end

const CTD{T,N,S} = TupleDummy{T,N,S,Any_Cartesian}
IndexStyle(td::CTD) = IndexCartesian()
@inline setindex!(td::CTD{T,N}, value::Tuple, ixs::Vararg{Int,N}) where {T,N} =
    fmap(td.arrays, value) do a, v
        @inbounds setindex!(a, v, ixs...) #add @inbounds
    end

@inline unalias(dest::TupleDummy, A::AbstractRange) = A
@inline unalias(dest::TupleDummy, A::AbstractArray) =
    mapreduce(x -> mightalias(x, A), |, dest.arrays) ? unaliascopy(A) : A
@inline broadcast_unalias(dest::TupleDummy, src::AbstractArray) =
    mapreduce(x -> x === src, |, dest.arrays) ? src : unalias(dest, src)

getdevice(td::TupleDummy) = getdevices(td.arrays)

## toa_similar
_similar(bc, T) = similar(bc, T)
_similar(bc::Broadcasted{DefaultArrayStyle{N}}, ::Type{Bool}) where N =
    similar(Array{Bool}, axes(bc)) # use Array{Bool} by default, I give up the optimization for BitArray.
function toa_similar(bc::Broadcasted)
    ElType = combine_eltypes(bc.f, bc.args)
    ElType <: Tuple && Base.isconcretetype(ElType) ||
        error("Inlegal return type for @tab!")
    ax = axes(bc)
    dest = map(tuple(ElType.parameters...)) do T
        _similar(bc, T)
    end
    S = typeof(dest)
    SS = typeof(ax)
    dest, TupleDummy{ElType,length(ax),S,All_Linear,SS}(dest, ax)
end

## better each
export eachdim′, eachcol′, eachrow′, eachslice′
@inline unsafe_view(A, I...) = Base.unsafe_view(A, to_indices(A, I)...)
@inline eachcol′(A::AbstractVecOrMat) = (unsafe_view(A, :, i) for i in axes(A, 2))
@inline eachrow′(A::AbstractVecOrMat) = (unsafe_view(A, i, :) for i in axes(A, 1))
@inline function eachdim′(A::AbstractArray; dim::Val{D}) where {D}
    D <= ndims(A) ||
        throw(DimensionMismatch("A doesn't have $dim dimensions"))
    axes_all = ntuple(d -> d == D ? Ref(:) : axes(A, d), ndims(A))
    (unsafe_view(A, i...) for i in Iterators.product(axes_all...))
end
@inline function eachslice′(A::AbstractArray; dim::Val{D}) where {D}
    D <= ndims(A) ||
        throw(DimensionMismatch("A doesn't have $dim dimensions"))
    inds_before = ntuple(d -> (:), D - 1)
    inds_after = ntuple(d -> (:), ndims(A) - D)
    (unsafe_view(A, inds_before..., i, inds_after...) for i in axes(A, D))
end
