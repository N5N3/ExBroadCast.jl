import .Interpolations: AbstractInterpolation, BSplineInterpolation, ScaledInterpolation,
                       itpflag, coefficients, dimension_wis, tcollect,
                       value_weights
ScaledBSplineInterpolation = ScaledInterpolation{T,N,<:BSplineInterpolation} where {T,N}
import Base.Broadcast: broadcasted
allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)
function broadcasted(itp::BSplineInterpolation, x...)
    @boundscheck (checkbounds(Bool, itp, x...) || Base.throw_boundserror(itp, x))
    if allequal(ndims.(x)...)
        itp′(x,y...) = @inbounds itp(x,y...)
        return broadcasted(itp′,x...)
    else
        itps = tcollect(itpflag, itp)
        wis = dimension_wis(value_weights, itps, axes(itp), x)
        coefs = coefficients(itp)
        itp″(x,y...) = getindex(coefs,x,y...)
        return broadcasted(itp″,wis...)
    end
end
unsaled(r::AbstractUnitRange, x) = x .- first(r) .+ oneunit(eltype(r))
unsaled(r::AbstractRange, x) = (x .- first(r)) .* inv(step(r)) .+ oneunit(eltype(r))
_vec(x) = x
_vec(x::AbstractArray) = vec(x)
function broadcasted(sitp::ScaledBSplineInterpolation, x...)
    @boundscheck (checkbounds(Bool, sitp, _vec.(x)...) || Base.throw_boundserror(sitp, x))
    if allequal(ndims.(x)...)
        sitp′(x,y...) = @inbounds sitp(x,y...)
        return broadcasted(sitp′,x...)
    end
    unsaledx = unsaled.(sitp.ranges,x)
    itp = sitp.itp
    itps = tcollect(itpflag, itp)
    wis = dimension_wis(value_weights, itps, axes(itp), unsaledx)
    coefs = coefficients(itp)
    itp′(x,y...) = @inbounds getindex(coefs,x,y...)
    return broadcasted(itp′,wis...)
end
