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
    N = length(dim[1])
    temp = ntuple(k -> @inbounds(_broadcast_getindex(bc, k)), Val(N))
    ElType = eltype(temp)
    ElType <: Tuple{Any,Any,Vararg{Any}} && Base.isconcretetype(ElType) ||
        error("Inlegal return type!")
    M = length(ElType.parameters)
    return ntuple(Val(M)) do i
        map(x->getfield(x,i),temp)
    end
end
@inline function tab_copy(bc::Broadcasted)
    dest, dest′ = toa_similar(bc)
    copyto!(dest′, bc)
    dest
end

@inline function copyto!(dest::TupleDummy, bc::Broadcasted{Nothing})
    getdevice(dest) == AnyGPU && return gpu_copyto!(dest, bc)
    invoke(copyto!, Tuple{AbstractArray, Broadcasted{Nothing}}, dest, bc)
end
