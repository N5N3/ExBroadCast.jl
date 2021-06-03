import .Interpolations: AbstractInterpolation, AbstractInterpolationWrapper, BSplineInterpolation,
                        LanczosInterpolation, ScaledInterpolation, GriddedInterpolation, MonotonicInterpolation,
                        Extrapolation, itptype, coefficients,  itpinfo, value_weights, WeightedAdjIndex,
                        allbetween, interp_getindex, WeightedIndex, weightedindex_parts, maybe_weightedindex
import Base: @propagate_inbounds
export UnSafeInterp, preweight

# a wrapper to turn off the boundary check when we use broadcast
struct UnSafeInterp{T,N,ITPT,IT} <: AbstractInterpolationWrapper{T,N,ITPT,IT}
    itp::ITPT
    UnSafeInterp(itp::AbstractInterpolation{T,N,IT}) where {T,N,IT} = new{T,N,typeof(itp),IT}(itp)
end
Base.parent(usitp::UnSafeInterp) = usitp.itp
@inline (usitp::UnSafeInterp{T,N})(args::Vararg{Any,N}) where {T,N} = @inbounds usitp.itp(args...)

struct WeightedInterp{A}
    coefs::A
end
Base.parent(witp::WeightedInterp) = witp.coefs
@inline (witp::WeightedInterp{<:AbstractArray{T,N}})(args::Vararg{Any,N}) where {T,N} = @inbounds witp.coefs[args...]

# pre_weight
unsaled(r::AbstractUnitRange, x) = @lzb x .- first(r) .+ oneunit(eltype(r))
unsaled(r::AbstractRange, x) = @lzb (x .- first(r)) .* inv(step(r)) .+ oneunit(eltype(r))

fastervec(x) = throw(ArgumentError("preweight only support Array/Number inputs!"))
fastervec(x::Number) = x
fastervec(x::AbstractArray) = vec(x)

# preweight
@inline preweight(itp::Extrapolation{T,N}, args::Vararg{Any,N}) where {T,N} =
    throw(ArgumentError("Extrapolation is not supported"))
@inline preweight(sitp::ScaledInterpolation{T,N}, args::Vararg{Any,N}) where {T,N} = begin
    @boundscheck (checkbounds(Bool, sitp, fastervec.(args)...) || Base.throw_boundserror(sitp, args))
    @inbounds preweight(sitp.itp, unsaled.(sitp.ranges, args)...)
end
@inline preweight(itp::BSplineInterpolation{T,N}, args::Vararg{Any,N}) where {T,N} = begin
    @boundscheck (checkbounds(Bool, itp, fastervec.(args)...) || Base.throw_boundserror(itp, args))
    function weight_ind(itpflag, knotvec, x)
        makewi(y, ::Any) = begin
            pos, coefs = weightedindex_parts((value_weights,), itpflag, knotvec, y)
            maybe_weightedindex(pos, coefs[1])
        end
        makewi.(x, Ref(itp))
    end
    (WeightedInterp(itp.coefs), weight_ind.(itpinfo(itp)..., args)...)
end

@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin

    import Adapt: adapt_structure, adapt
    import .CUDA: cudaconvert
    adapt_structure(to, itp::BSplineInterpolation{T,N}) where {T,N} = begin
        coefs, parentaxes, it = itp.coefs, itp.parentaxes, itp.it
        coefs′ = adapt(to, coefs)
        BSplineInterpolation{T,N,typeof.((coefs′,it,parentaxes))...}(coefs′, parentaxes, it)
    end

    adapt_structure(to, itp::LanczosInterpolation{T,N}) where {T,N} = begin
        coefs, parentaxes, it = itp.coefs, itp.parentaxes, itp.it
        coefs′ = adapt(to, coefs)
        parentaxes′ = adapt.(Ref(to), parentaxes)
        LanczosInterpolation{T,N,typeof.((it,coefs′,parentaxes′))...}(coefs′, parentaxes′, it)
    end

    adapt_structure(to, itp::ScaledInterpolation{T,N}) where {T,N} = begin
        ranges = itp.ranges
        itp′ = adapt(to, itp.itp)
        IT = itptype(itp)
        ScaledInterpolation{T,N,typeof(itp′),IT,typeof(ranges)}(itp′, ranges)
    end

    adapt_structure(to, itp::Extrapolation{T,N}) where {T,N} = begin
        et = itp.et
        itp′ = adapt(to,  itp.itp)
        IT = itptype(itp)
        Extrapolation{T,N,typeof(itp′),IT,typeof(et)}(itp′, et)
    end

    adapt_structure(to, itp::UnSafeInterp{T,N}) where {T,N} =
        adapt(to, itp.itp) |> UnSafeInterp

    adapt_structure(to, itp::WeightedInterp) =
        adapt(to, itp.coefs) |> WeightedInterp


    @inline getcoefs(x::AbstractInterpolationWrapper) = parent(x) |> getcoefs
    @inline getcoefs(x::AbstractInterpolation) = coefficients(x)
    @inline getcoefs(x::WeightedInterp) = parent(x)
    device(x::Union{WeightedInterp, AbstractInterpolation}) = getcoefs(x) |> device

    # With Adapt v"3.3.1", there's no need to use Ref to force adapt
    # but I think the style hack is still needed.
    broadcasted(itp::Union{WeightedInterp, AbstractInterpolation}, args...) = begin
        # using Ref on CPU mess up the speed, so we invoke to the general dispatch
        args′ = broadcastable.(args)
        style_hack = getcoefs(itp) |> Ref
        style = combine_styles(style_hack, args′...)
        broadcasted(style, itp, args′...)
    end

    #make checkbounds work on gpu
    allbetween(l::Real, xs::AbstractArray, u::Real) = begin
        getdevice(xs) == AnyGPU && return all(x -> l <= x <= u, xs)
        invoke(allbetween, Tuple{typeof(l),Any,typeof(u)}, l, xs, u)
    end

    # Type interfer and display fix
    using GPUArrays
    import GPUArrays: AbstractGPUArray
    GPUArrays._getindex(A::AbstractGPUArray{T,N}, I::Vararg{Union{Int,WeightedIndex},N}) where {T,N} =
        interp_getindex(A, I, ntuple(d->0, Val(N))...)

end
