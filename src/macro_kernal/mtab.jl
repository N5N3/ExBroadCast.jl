## materialize
@inline mtab_materialize(x, ::Integer) = x
@inline mtab_materialize(bc::Broadcasted, TN::Integer) = mtab_copy(instantiate(bc), TN)

@inline mtb_materialize!(dest::Tuple{Vararg{AbstractArray}}, bc::Broadcasted, TN::Integer) = 
    mtb_materialize!(TupleDummy(dest), bc, Val(TN)) |> parent

## copyto
@inline mtab_copy(bc::Broadcasted{Style{Tuple}}, ::Integer) = tab_copy(bc)
@inline mtab_copy(bc::FilledBC, ::Integer) = copy(bc)
@inline mtab_copy(bc::Broadcasted, TN::Integer) =
    mtb_copyto!(toa_similar(bc), bc, TN) |> parent
