import .StaticArrays: StaticArrayStyle, StaticArray
mtb_copy(bc::Broadcasted{<:StaticArrayStyle}) = copy(bc) # I believe no one need it
mtb_copyto!(dest::StaticArray, bc::Broadcasted{Nothing}) = copyto!(dest, bc) # I believe no one need it
_similar(bc::Broadcasted{<:StaticArrayStyle}, T) = throw("StaticArrays is not supported for non-inplace @tab")
