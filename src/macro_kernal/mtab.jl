## materialize
@inline mtab_materialize(x, ::Any) = x
@inline mtab_materialize(bc::Broadcasted, ::Val{TN}) where TN = begin
    TN <= 1 && return tab_copy(instantiate(bc))
    mtab_copy(instantiate(bc), TN)
end

@inline mtb_materialize!(dest::Tuple{Vararg{AbstractArray}}, bc::Broadcasted, ::Val{TN}) where TN = 
    mtb_materialize!(TupleDummy(dest), bc, Val(TN)) |> parent

## copyto
@inline mtab_copy(bc::Broadcasted{Style{Tuple}}, ::Any) = tab_copy(bc)
@inline mtab_copy(bc::FilledBC, ::Any) = copy(bc)
@inline mtab_copy(bc::Broadcasted, TN) = mtb_copyto!(toa_similar(bc), bc, TN) |> parent
