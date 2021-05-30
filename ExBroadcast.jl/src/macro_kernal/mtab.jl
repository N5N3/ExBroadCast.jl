## broadcast
mtab_broadcast(f::Tf, As...) where {Tf} = mtab_materialize(broadcasted(f, As...))

## materialize
@inline mtab_materialize(x) = x
@inline mtab_materialize(bc::Broadcasted) = mtab_copy(instantiate(bc))

@inline function mtb_materialize!(
    dest::NTuple{N,AbstractArray},
    bc::Broadcasted{Style},
) where {Style,N}
    mtb_materialize!(TupleDummy(dest), bc)
    dest
end

## copyto
@inline mtab_copy(bc::Broadcasted{Style{Tuple}}) = tab_copy(bc)
@inline mtab_copy(bc::FilledBC) = copy(bc)
@inline function mtab_copy(bc::Broadcasted)
    dest, dest′ = toa_similar(bc)
    mtb_copyto!(dest′, bc)
    dest
end

## copyto!
