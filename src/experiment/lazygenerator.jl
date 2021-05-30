module LazyGenerator
import ExBroadcast: AbstractScalar
import Base: @propagate_inbounds, getindex
import Base.Broadcast: BroadcastStyle, broadcastable, extrude, newindex, broadcasted
@inline ndims(x) = x isa Tuple ? 1 : Base.ndims(x)
@inline size(x) = x isa Tuple ? (length(x),) : Base.size(x)
struct IndDummy{N,S,T,D} <: AbstractArray{T,N}
    data::D
    IndDummy{N,S}(data::T) where {N,S,T} = new{N,S,eltype(data),T}(data)
end
IndDummy(data::AbstractScalar) = data
IndDummy{AD}(data) where AD = IndDummy{AD + ndims(data),AD + 1}(data)
IndDummy{AD}(id::IndDummy{N,S}) where {N,S,AD} = IndDummy{AD + N,AD + S}(id.data)

Base.axes(id::IndDummy{N,S}) where {N,S} = (ntuple(_ -> Base.OneTo(1), Val(S - 1))..., axes(id.data)...)
Base.size(id::IndDummy{N,S}) where {N,S} = (ntuple(_ -> 1, Val(S - 1))..., size(id.data)...)
@inline extrude(x::IndDummy) = x
@inline newindex(::IndDummy, I::CartesianIndex) = I
@propagate_inbounds getindex(id::IndDummy{N,N}, I::Vararg{Int, N}) where {N} = id.data[I[N]]
@propagate_inbounds getindex(id::IndDummy{N,S}, I::Vararg{Int, N}) where {N,S} = id.data[I[S:N]...]
BroadcastStyle(::Type{ID}) where ID <: IndDummy = BroadcastStyle(ID.parameters[4])

broadcastable(g::Base.Generator) = broadcasted(g.f, g.iter)
broadcastable(i::Iterators.ProductIterator) = begin
    iters = i.iterators
    N = length(iters) - 1
    dims = ntuple(i -> ndims(iters[i]), N) |> cumsum
    iters′ = ntuple(N) do i
        IndDummy{dims[i]}(broadcastable(iters[i+1]))
    end
    broadcasted(tuple, first(iters), iters′...)
end
end
using .LazyGenerator