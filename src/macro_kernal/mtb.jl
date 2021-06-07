## materialize
@inline mtb_materialize(x, ::Integer) = x
@inline mtb_materialize(bc::Broadcasted, TN::Integer) =
    mtb_copy(instantiate(bc), TN)

@inline mtb_materialize!(dest, x, TN::Integer) =
    mtb_materialize!(dest, instantiate(Broadcasted(identity, (x,), axes(dest))), TN)

@inline mtb_materialize!(dest, bc::Broadcasted{Style}, TN::Integer) where {Style} =
    mtb_materialize!(combine_styles(dest, bc), dest, bc, TN)

@inline mtb_materialize!(::BroadcastStyle, dest, bc::Broadcasted{Style}, TN::Integer) where {Style} =
    mtb_copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))), TN)

## copyto
@inline mtb_copy(bc::Broadcasted{Style{Tuple}}, ::Integer) = copy(bc)
@inline mtb_copy(bc::FilledBC, ::Integer) = copy(bc)
@inline mtb_copy(bc::Broadcasted{<:Union{Nothing,Unknown}}, ::Integer) = copy(bc)
@inline function mtb_copy(bc::Broadcasted{Style}, TN::Integer) where {Style}
    ElType = combine_eltypes(bc.f, bc.args)
    if !Base.isconcretetype(ElType)
        @warn "$(ElType) is not concrete, invoke Base.copy"
        copy(bc)
    end
    mtb_copyto!(similar(bc, ElType), bc, TN)
end

## copyto!
@inline mtb_copyto!(dest::AbstractArray, bc::Broadcasted, TN::Integer) =
    mtb_copyto!(dest, convert(Broadcasted{Nothing}, bc), TN)

@inline mtb_copyto!(dest::AbstractArray, bc::FilledBC, ::Integer) = copyto!(dest, bc)

@inline function mtb_copyto!(dest::AbstractArray, bc::Broadcasted{Nothing}, TN::Integer)
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    if bc.f === identity && bc.args isa Tuple{AbstractArray}
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = preprocess(dest, bc)
    Inds = eachindex(bc′)
    len, Iᵉ, Iˢs = mtb_config(TN, Inds)

    @inline function broadcast_kernel(Iˢ)
        Inds′ = @inbounds view(Inds, Iˢ:min(Iˢ + len, Iᵉ))
        @inbounds @simd for I in Inds′
            dest[I] = bc′[I]
        end
        nothing
    end

    mtb_call(broadcast_kernel, Iˢs)
    dest
end

const bitcache_chunks_bits = 6
@inline function mtb_copyto!(dest::BitArray, bc::Broadcasted{Nothing}, TN::Integer)
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    ischunkedbroadcast(dest, bc) && return chunkedcopyto!(dest, bc)
    length(dest) < 256 &&
        return invoke(copyto!, Tuple{AbstractArray,Broadcasted{Nothing}}, dest, bc)
    destc = dest.chunks
    bc′ = preprocess(dest, bc)
    Inds = eachindex(bc′)
    len, Iᵉ, Iˢs = mtb_config(TN, Inds, bitcache_size)

    @inline function broadcast_kernel(Iˢ)
        tmp = Vector{Bool}(undef, bitcache_size)
        cind = 1 + Iˢ >> bitcache_chunks_bits
        @inbounds Inds′ = view(Inds, Iˢ:min(Iˢ + len, Iᵉ))
        for P in Iterators.partition(Inds′, bitcache_size)
            ind = 1
            @inbounds @simd for I in P
                tmp[ind] = bc′[I]
                ind += 1
            end
            @inbounds @simd for i = ind:bitcache_size
                tmp[i] = false
            end
            dumpbitcache(destc, cind, tmp)
            cind += bitcache_chunks
        end
        nothing
    end
    mtb_call(broadcast_kernel, Iˢs)
    dest
end
using Polyester
## mtb_call
@static if threads_provider == "Polyester"
    function mtb_call(@nospecialize(kernal::Function), inds::AbstractRange{Int})
        @batch for tid in eachindex(inds)
            kernal(inds[tid])
        end
        nothing
    end
else
    function mtb_call(@nospecialize(kernal::Function), inds::AbstractRange{Int})
        len = length(inds)
        @inbounds if len > 3
            len′ = len >> 1
            task = Threads.@spawn mtb_call(kernal, inds[1:len′])
            mtb_call(kernal, inds[1+len′:len])
            wait(task)
        elseif len == 3
            task₁ = Threads.@spawn kernal(inds[1])
            task₂ = Threads.@spawn kernal(inds[2])
            kernal(inds[3])
            wait(task₁)
            wait(task₂)
        else
            task = Threads.@spawn kernal(inds[1])
            kernal(inds[2])
            wait(task)
        end
        nothing
    end
end

##Multi-threading config
@inline function mtb_config(thread_num::Integer, ax::AbstractArray, min_size::Integer = 1)
    Iˢ, Iᵉ = Ref(ax) .|> (firstindex, lastindex)
    len = cld(length(ax), thread_num * min_size) * min_size
    len - 1, Iᵉ, Iˢ .+ len .* (0:cld(length(ax), len)-1)
end
