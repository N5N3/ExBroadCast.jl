## materialize
@inline tab_materialize(x) = x
@inline tab_materialize(bc::Broadcasted) = instantiate(bc) |> tab_copy

@inline function materialize!(
    dest::NTuple{N,AbstractArray},
    bc::Broadcasted{Style},
) where {N,Style}
    materialize!(TupleDummy(dest), bc)
    dest
end

## copyto
@inline tab_copy(bc::FilledBC) = copy(bc)
@inline tab_copy(bc::Broadcasted) = copyto!(toa_similar(bc), bc) |> parent
@inline function tab_copy(bc::Broadcasted{Style{Tuple}})
    dim = axes(bc)
    length(dim) == 1 || throw(DimensionMismatch("tuple only supports one dimension"))
    @inline maketuple(x, y) = tuple(x, y...)
    @inline getind(k) = @inbounds bc[k]
    ElType = typeof(getind[1])
    ElType <: Tuple{Any,Vararg{Any}} && Base.isconcretetype(ElType) ||
        throw("$ElType is not a legal return type for @tab!")
    N = length(dim[1])
    N <= 16 && return mapfoldr(getind, (x, y) -> maketuple.(x, y), ntuple(identity,Val(N)))
    mapfoldr(getind, (x, y) -> maketuple.(x, y), dim[1])
end
