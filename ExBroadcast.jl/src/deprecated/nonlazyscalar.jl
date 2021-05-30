# module NonLazyScalar
#     import Base.Broadcast: broadcasted
#     import ExBroadcast.AbstractScalar
#     const Scalar = AbstractScalar{<:Number}

#     ## broadcasted expand
#     for op in (:+, :*, :&, :|, :xor, :min, :max, :kron)
#         @eval begin
#             @inline broadcasted(::typeof($op),x,y,z,args...) = 
#                 broadcasted($op, broadcasted($op,x,y), z, args...)
#         end
#     end

#     for op in (:muladd, :fma,)
#         @eval broadcasted(f::typeof($op), x::Scalar, y::Scalar, z::Scalar) = f(only(x),only(y),only(z))
#     end

#     for op in (
#         :+, :-, :*, :/, :\, :÷, :%, :^, :fld, :cld, :mod,
#         :&, :|, :⊻, :>>>, :>>, :<<,
#         :atan, :hypot,
#         :min, :max,
#         :binomial, :gcd, :lcm,
#         )
#         @eval broadcasted(f::typeof($op), x::Scalar, y::Scalar) = f(only(x),only(y))
#     end

#     for op in (
#         :+, :-, :!, :inv, :~,
#         :sin  ,:cos  ,:tan  ,:sind ,:cosd ,:tand ,:sinpi,:cospi,
#         :asin ,:acos ,:atan ,:asind,:acosd,:atand,
#         :sec  ,:csc  ,:cot  ,:secd ,:cscd ,:cotd ,
#         :asec ,:acsc ,:acot ,:asecd,:acscd,:acotd,
#         :sinh ,:cosh ,:tanh ,:sech ,:csch ,:coth ,
#         :asinh,:acosh,:atanh,:asech,:acsch,:acoth,
#         :sinc,:cosc,
#         :deg2rad,:rad2deg,
#         :log,:log2,:log10,:log1p,
#         :exp,:exp2,:exp10,:ldexp,:expm1,
#         :round,:ceil,:floor,:trunc,
#         :abs,:abs2,:sign,:sqrt,:cbrt,:real,:imag,:conj,:angle,:cis,
#         :factorial,:big,
#         :identity
#         )
#         @eval broadcasted(f::typeof($op), x::Scalar) = f(only(x))
#     end
#     broadcasted(::Type{T}, x::Scalar) where {T} = T(only(x))
# end
# using .NonLazyScalar
