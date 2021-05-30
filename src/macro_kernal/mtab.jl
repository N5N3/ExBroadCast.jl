## broadcast
mtab_broadcast(f::Tf, As...) where {Tf} = mtab_materialize(broadcasted(f, As...))

## materialize
@inline mtab_materialize(x) = x
@inline mtab_materialize(bc::Broadcasted) = mtab_copy(instantiate(bc))

@inline mtb_materialize!(dest::Tuple{Vararg{AbstractArray}}, bc::Broadcasted) = 
    mtb_materialize!(TupleDummy(dest), bc) |> parent

## copyto
@inline mtab_copy(bc::Broadcasted{Style{Tuple}}) = tab_copy(bc)
@inline mtab_copy(bc::FilledBC) = copy(bc)
@inline mtab_copy(bc::Broadcasted) = mtb_copyto!(toa_similar(bc), bc) |> parent
