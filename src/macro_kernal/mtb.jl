## broadcast
mtb_broadcast(f::Tf, As...) where {Tf} = mtb_materialize(broadcasted(f, As...))
mtb_broadcast!(f::Tf, dest, As...) where {Tf} = mtb_materialize!(dest, broadcasted(f, As...))

## materialize
@inline mtb_materialize(x) = x
@inline mtb_materialize(bc::Broadcasted) = mtb_copy(instantiate(bc))

@inline function mtb_materialize!(dest, x)
    mtb_materialize!(dest, instantiate(Broadcasted(identity, (x,), axes(dest))))
end
@inline function mtb_materialize!(dest, bc::Broadcasted{Style}) where {Style}
    mtb_materialize!(combine_styles(dest, bc), dest, bc)
end

@inline function mtb_materialize!(
    ::BroadcastStyle,
    dest,
    bc::Broadcasted{Style},
) where {Style}
    mtb_copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end

## copyto
@inline mtb_copy(bc::Broadcasted{Style{Tuple}}) = copy(bc)
@inline mtb_copy(bc::FilledBC) = copy(bc)
@inline mtb_copy(bc::Broadcasted{<:Union{Nothing,Unknown}}) = copy(bc)
@inline function mtb_copy(bc::Broadcasted{Style}) where {Style}
    ElType = combine_eltypes(bc.f, bc.args)
    Base.isconcretetype(ElType) ||
        error("The type of output is not concrete! Please use Base.Broadcast.")
    mtb_copyto!(similar(bc, ElType), bc)
end

## copyto!
@inline mtb_copyto!(dest::AbstractArray, bc::Broadcasted) =
    mtb_copyto!(dest, convert(Broadcasted{Nothing}, bc))

@inline mtb_copyto!(dest::AbstractArray, bc::FilledBC) = copyto!(dest, bc)

@inline function mtb_copyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
    device(dest) == AnyGPU && return gpu_copyto!(dest, bc)
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    if bc.f === identity && bc.args isa Tuple{AbstractArray}
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = preprocess(dest, bc)
    Inds = eachindex(bc′)
    len, Iᵉ, Iˢs = mtb_config(Inds)

    @inline function broadcast_kernel(Iˢ)
        Inds′ = Iˢ isa Colon ? Inds : @inbounds view(Inds, Iˢ:min(Iˢ + len, Iᵉ))
        @inbounds @simd for I in Inds′
            dest[I] = bc′[I]
        end
        nothing
    end

    mtb_call(broadcast_kernel, Iˢs)
    dest
end

const bitcache_chunks_bits = 6
@inline function mtb_copyto!(dest::BitArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    ischunkedbroadcast(dest, bc) && return chunkedcopyto!(dest, bc)
    length(dest) < 256 && return invoke(copyto!, Tuple{AbstractArray,Broadcasted{Nothing}}, dest, bc)
    destc = dest.chunks
    bc′ = preprocess(dest, bc)
    Inds = eachindex(bc′)
    len, Iᵉ, Iˢs = mtb_config(Inds, bitcache_size)

    @inline function broadcast_kernel(Iˢ)
        tmp = Vector{Bool}(undef, bitcache_size)
        if Iˢ isa Colon
            cind, Inds′ = 1, Inds
        else
            cind = 1 + Iˢ >> bitcache_chunks_bits
            @inbounds Inds′ = view(Inds, Iˢ:min(Iˢ + len, Iᵉ))
        end
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
    function mtb_call(
        @nospecialize(kernal::Function),
        inds::AbstractRange,
    )
        len = length(inds)
        if len > 1
            @batch for tid in eachindex(inds)
                kernal(inds[tid])
            end
        else
            kernal(:)
        end
        nothing
    end
else
    function mtb_call(
        @nospecialize(kernal::Function), 
        inds::AbstractRange,
    )
        len = length(inds)
        @inbounds if len > 3
            len′ = len >> 1
            task = Threads.@spawn mtb_call(kernal, inds[1:len′])
            mtb_call(kernal, inds[1+len′:len])
            wait(task)
        elseif len == 3
            task1 = Threads.@spawn kernal(inds[1])
            task2 = Threads.@spawn kernal(inds[2])
            kernal(inds[3])
            wait(task1)
            wait(task2)
        elseif len == 2
            task = Threads.@spawn kernal(inds[1])
            kernal(inds[2])
            wait(task)
        else
            kernal(:)
        end
        nothing
    end
end

##Multi-threading config
@inline function mtb_config(
    ax::AbstractArray,
    min_size::Integer = 1,
)
    Iˢ, Iᵉ = Ref(ax) .|> (firstindex, lastindex) 
    len = cld(length(ax), num_threads() * min_size) * min_size
    len - 1, Iᵉ, Iˢ .+ len .* (0:cld(length(ax), len)-1)
end
