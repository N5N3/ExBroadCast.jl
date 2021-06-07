export Lazy, eachdim′, eachcol′, eachrow′, eachslice′
# getsame
@inline getsame(f::F, x) where {F} = f(x)
@inline getsame(f::F, x, y, zs...) where {F} = begin
    f(x) == getsame(f, y, zs...) || throw(ArgumentError("inputs with different $f"))
    f(x)
end

# check the backend
using ArrayInterface
const AnyGPU = ArrayInterface.GPU()
gpu_copyto!(args...) = throw("You need using CUDA first!")
device(::Type{T}) where {T} = ArrayInterface.device(T)
device(x) = typeof(x) |> device
@inline devices(::Type{<:NamedTuple{<:Any,T}}) where {T} = devices(T)
@inline devices(::Type{T}) where {T<:Tuple} = getsame(device, T.parameters...)

# force inlined map that return nothing
@inline fmap(f::F, t₁::Tuple{}) where {F} = nothing
@inline fmap(f::F, t₁::Tuple) where {F} = begin
    f(t₁[1])
    fmap(f, Base.tail(t₁))
end

@inline fmap(f::F, t₁::Tuple{}, t₂::Tuple{}) where {F} = nothing
@inline fmap(f::F, t₁::Tuple, t₂::Tuple) where {F} = begin
    f(t₁[1], t₂[1])
    fmap(f, Base.tail(t₁), Base.tail(t₂))
end


### import many funtions
import Base.Broadcast: throwdm, AbstractArrayStyle, Unknown, combine_eltypes,
    preprocess, Broadcasted, DefaultArrayStyle, ischunkedbroadcast, chunkedcopyto!, bitcache_size, 
    dumpbitcache, bitcache_chunks, materialize!, BroadcastStyle, combine_styles, instantiate, 
    broadcastable, broadcast_unalias, Style, _broadcast_getindex, broadcasted
import Base: size, axes, setindex!, unalias, mightalias, unaliascopy, IndexStyle, parent, 
    unsafe_setindex!, copyto!
const FilledBC = Broadcasted{<:AbstractArrayStyle{0}}
const AllLinear = true
const AnyCartesian = false
"""
    TupleDummy(arrays::Tuple)
Dummy structure for broadcast with multiple outputs.
A simplified SoA implementation.
**This is not part of the interface. Not exported.**
"""
struct TupleDummy{T,N,L,As,AXs} <: AbstractArray{T,N}
    arrays::As
    ax::AXs      # add this field to avoid repeated Base.OneTo()
    TupleDummy{T,N,L}(arrays::As, ax::AXs) where {T,N,L,As,AXs} =
        new{T,N,L,As,AXs}(arrays, ax)
end
function TupleDummy(arrays::Tuple{AbstractArray,AbstractArray,Vararg{AbstractArray}}) ## at least 2 outputs
    ax = getsame(axes, arrays...)
    LinearFLag = mapreduce((x) -> IndexStyle(x) isa IndexLinear, &, arrays)
    ElType = Tuple{eltype.(arrays)...}
    TupleDummy{ElType,length(ax),LinearFLag}(arrays, ax)
end
device(::Type{T}) where {T<:TupleDummy} = devices(T.parameters[4])
parent(td::TupleDummy) = td.arrays
size(td::TupleDummy, args...) = size(td.arrays[1], args...)
axes(td::TupleDummy) = td.ax


LTD{N} = TupleDummy{T,N,AllLinear} where {T}
Base.IndexStyle(::LTD) = IndexLinear()
@inline setindex!(td::LTD, value::Tuple, ix::Int) =
    fmap((a, v) -> unsafe_setindex!(a, v, ix) , td.arrays, value)
CTD{N} = TupleDummy{T,N,AnyCartesian} where {T}
Base.IndexStyle(::CTD) = IndexCartesian()
@inline setindex!(td::CTD{N}, value::Tuple, ixs::Vararg{Int,N}) where {N} =
    fmap((a, v) -> unsafe_setindex!(a, v, ixs...) , td.arrays, value)

@inline unalias(dest::TupleDummy, A::AbstractRange) = A
@inline unalias(dest::TupleDummy, A::AbstractArray) =
    mapreduce(x -> mightalias(x, A), |, dest.arrays) ? unaliascopy(A) : A
@inline broadcast_unalias(dest::TupleDummy, src::AbstractArray) =
    mapreduce(x -> x === src, |, dest.arrays) ? src : unalias(dest, src)

## toa_similar
_similar(bc, T) = similar(bc, T)
_similar(bc::Broadcasted{<:DefaultArrayStyle}, ::Type{Bool}) =
    similar(Array{Bool}, axes(bc))
function toa_similar(bc::Broadcasted)
    ElType = combine_eltypes(bc.f, bc.args)
    ElType <: Tuple{Any,Vararg{Any}} && Base.isconcretetype(ElType) ||
        throw(ArgumentError("$ElType is not a legal return type for @tab!"))
    dest = map(T -> _similar(bc, T), tuple(ElType.parameters...))
    TupleDummy{ElType,ndims(bc),AllLinear}(dest, axes(bc))
end

module LazyCollect
import Base: @propagate_inbounds, getindex
import Base.Broadcast: BroadcastStyle, broadcastable, extrude, newindex, broadcasted,
    instantiate, Broadcasted, DefaultArrayStyle
export Lazy

@inline ndims(x) = x isa Tuple ? 1 : Base.ndims(x)
@inline size(x) = x isa Tuple ? (length(x),) : Base.size(x)
# Like reshape, but allow tuple inputs.
struct FakeDim{N,S,T,D} <: AbstractArray{T,N}
    data::D
    FakeDim{N,S}(data::T) where {N,S,T} = new{N,S,eltype(data),T}(data)
end
FakeDim{AD}(data) where {AD} = AD == 0 ? data : FakeDim{AD + ndims(data),AD + 1}(data)
FakeDim{AD}(data::AbstractArray{T,0}) where {AD,T} = data
FakeDim{AD}(data::Ref{T}) where {AD,T} = data
FakeDim{AD}(data::Tuple{Any}) where {AD} = data
FakeDim{AD}(data::Number) where {AD} = data
FakeDim{AD}(id::FakeDim{N,S}) where {N,S,AD} = FakeDim{AD + N,AD + S}(id.data)
FakeDim{AD}(bc::Broadcasted) where {AD} = broadcasted(bc.f, FakeDim{AD}.(bc.args)...)

Base.axes(id::FakeDim{N,S}) where {N,S} =
    (ntuple(_ -> Base.OneTo(1), Val(S - 1))..., axes(id.data)...)
Base.size(id::FakeDim{N,S}) where {N,S} = (ntuple(_ -> 1, Val(S - 1))..., size(id.data)...)
@inline extrude(x::FakeDim) = x
# Specialize {N,N} to avoid allocation
@inline newindex(::FakeDim{N,N}, I::CartesianIndex) where {N} =
    CartesianIndex(ntuple(oneunit, Val(N - 1))..., I.I[N]...)
@propagate_inbounds getindex(id::FakeDim{N,N}, I::Vararg{Int,N}) where {N} = id.data[I[N]]
@inline newindex(::FakeDim{N,S}, I::CartesianIndex) where {N,S} =
    CartesianIndex(ntuple(oneunit, Val(S - 1))..., I.I[S:N]...)
@propagate_inbounds getindex(id::FakeDim{N,S}, I::Vararg{Int,N}) where {N,S} =
    id.data[I[S:N]...]
BroadcastStyle(::Type{ID}) where {ID<:FakeDim} =
    ID.parameters[4] <: Tuple ? DefaultArrayStyle{ID.parameters[1]}() :
    BroadcastStyle(ID.parameters[4])

# Lazy is used to wrap Generator/Productor to avoid collect before broadcast.
struct Lazy{P}
    ori::P
end
Lazy(l::Lazy) = l
@inline Base.collect(l::Lazy) = collect(l.ori)
@inline Base.iterate(l::Lazy, args...) = iterate(l.ori, args...)
@inline Base.simd_outer_range(l::Lazy) = Base.simd_outer_range(l.ori)
@inline Base.simd_inner_length(l::Lazy, args...) = Base.simd_inner_length(l.ori, args...)
@inline Base.simd_index(l::Lazy, args...) = Base.simd_index(l.ori, args...)

@inline broadcastable(l::Lazy) = lazycollect(l.ori)
@inline lazycollect(x) = broadcastable(x)
@inline lazycollect(g::Base.Generator) = broadcasted(g.f, Lazy(g.iter)) |> instantiate
@inline lazycollect(i::Iterators.ProductIterator) = begin
    iters = i.iterators
    broadcasted(tuple, fakedims(iters...)...) |> instantiate
end
@inline fakedims(x, args...) = fakedims(Val(0), x, args...)
@inline fakedims(::Val{N}, x, args...) where {N} =
    (FakeDim{N}(lazycollect(x)), fakedims(Val(N + ndims(x)), args...)...)
@inline fakedims(::Val{N}) where {N} = ()

end
using .LazyCollect
## better each
@inline unsafe_view(A, I...) = Base.unsafe_view(A, to_indices(A, I)...)
eachcol′(A::AbstractVecOrMat) = Lazy(unsafe_view(A, :, i) for i in axes(A, 2))
eachrow′(A::AbstractVecOrMat) = Lazy(unsafe_view(A, i, :) for i in axes(A, 1))
function eachdim′(A::AbstractArray; dim::Val{D}) where {D}
    D <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    axes_all = ntuple(d -> d == D ? Ref(:) : axes(A, d), ndims(A))
    Lazy(unsafe_view(A, i...) for i in Iterators.product(axes_all...))
end
function eachslice′(A::AbstractArray; dim::Val{D}) where {D}
    D <= ndims(A) || throw(DimensionMismatch("A doesn't have $dim dimensions"))
    inds_before = ntuple(d -> (:), D - 1)
    inds_after = ntuple(d -> (:), ndims(A) - D)
    Lazy(unsafe_view(A, inds_before..., i, inds_after...) for i in axes(A, D))
end
