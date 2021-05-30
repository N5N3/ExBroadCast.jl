## broadcast
tab_broadcast(f::Tf, As...) where {Tf} = tab_materialize(broadcasted(f, As...))

## materialize
@inline tab_materialize(x) = x
@inline tab_materialize(bc::Broadcasted) = tab_copy(instantiate(bc))

@inline function materialize!(
    dest::NTuple{N,AbstractArray},
    bc::Broadcasted{Style},
) where {N,Style}
    materialize!(TupleDummy(dest), bc)
    dest
end

## copyto
@inline tab_copy(bc::FilledBC) = copy(bc)
@inline function tab_copy(bc::Broadcasted{Style{Tuple}})
    dim = axes(bc)
    length(dim) == 1 || throw(DimensionMismatch("tuple only supports one dimension"))
    ElType = typeof(bc[1])
    ElType <: Tuple{Any,Vararg{Any}} && Base.isconcretetype(ElType) || 
        throw("$ElType is not a legal return type for @tab!")
    @inline maketuple(x, y) = tuple(x, y...)
    @inline getind(k) = @inbounds _broadcast_getindex(bc, k)
    N = length(dim[1])
    N <= 16 && return mapfoldr(getind, (x, y) -> maketuple.(x, y), ntuple(identity,Val(N)))
    mapfoldr(getind, (x, y) -> maketuple.(x, y), dim[1])
end

@inline function tab_copy(bc::Broadcasted)
    dest = toa_similar(bc)
    copyto!(dest, bc)
    parent(dest)
end
