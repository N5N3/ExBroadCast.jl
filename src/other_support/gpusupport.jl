## GPU support
import GPUArrays: backend, gpu_call, linear_index, launch_heuristic, launch_configuration,
    AbstractGPUArrayStyle
    
# check the backend
using ArrayInterface
const GPU = ArrayInterface.GPU
device(::Type{T}) where {T} = ArrayInterface.device(T)
device(x) = typeof(x) |> device
devices(::Type{<:NamedTuple{<:Any,T}}) where {T} = devices(T)
devices(::Type{T}) where {T<:Tuple} = getsame(device, T.parameters...)
backends(::Type{<:NamedTuple{<:Any,T}}) where {T} = backends(T)
backends(::Type{T}) where {T<:Tuple} = getsame(backend, T.parameters...)

# Adapt fix
import Adapt: adapt_structure, adapt
@inline adapts(to, x, ys...) = (adapt(to, x), adapts(to, ys...)...)
@inline adapts(to) = ()

device(::Type{T}) where {T<:TupleDummy} = devices(T.parameters[4])
backend(::Type{T}) where {T<:TupleDummy} = backends(T.parameters[4])
adapt_structure(to, td::TupleDummy{T,N,L}) where {T,N,L} =
    TupleDummy{T,N,L}(adapts(to, td.arrays...), td.ax)

@inline function copyto!(dest::TupleDummy, bc::Broadcasted{Nothing})
    device(dest) isa GPU && return gpu_copyto!(dest, bc)
    invoke(copyto!, Tuple{AbstractArray,Broadcasted{Nothing}}, dest, bc)
end

#math: override sincos and sincospi 7 times faster.
using GPUCompiler
@static if isdefined(Base.Experimental, Symbol("@overlay"))
    Base.Experimental.@MethodTable(method_table)
else
    const method_table = nothing
end
const overrides = quote end
macro device_override(ex)
    code = quote
        $GPUCompiler.@override($method_table, $ex)
    end
    if isdefined(Base.Experimental, Symbol("@overlay"))
        return esc(code)
    else
        push!(overrides.args, code)
        return
    end
end
const CuFloat = Union{Float32,Float64}
@device_override Base.sincos(x::CuFloat) = sin(x), cos(x)
@device_override Base.sincospi(x::CuFloat) = sinpi(x), cospi(x)

precompiling = ccall(:jl_generating_output, Cint, ()) != 0
if !precompiling
    eval(overrides)
end

# general gpu_copyto!(modified from GPUArrays.jl's implement to support OffsetArrays.jl)

@inline function gpu_copyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    bc′ = preprocess(dest, bc)

    function broadcast_kernel(ctx, dest, bc′, nelem)
        Inds = eachindex(bc′)
        Iˢ, Iᵉ = firstindex(Inds), lastindex(Inds)
        @inbounds for i = 1:nelem
            j = linear_index(ctx, i) + Iˢ - 1
            j > Iᵉ && return
            I = Inds[j]
            dest[I] = bc′[I]
        end
        nothing
    end

    heuristic = launch_heuristic(backend(dest), broadcast_kernel, dest, bc′, 1)
    config = launch_configuration(backend(dest), heuristic, length(dest), typemax(Int))
    gpu_call(broadcast_kernel, dest, bc′, config.elements_per_thread;
             threads = config.threads, blocks = config.blocks)
    dest
end
## use style to drop_mtb
function drop_mtb(Style)
    @eval @inline mtb_materialize(bc::Broadcasted{<:$Style}, ::Integer) =
        copy(instantiate(bc))
    @eval @inline mtab_materialize(bc::Broadcasted{<:$Style}, ::Integer) =
        tab_copy(instantiate(bc))
    @eval @inline mtb_materialize!(::$Style, dest, bc::Broadcasted{Style}, ::Integer) where {Style} =
        copyto!(dest, instantiate(Broadcasted{Style}(bc.f, bc.args, axes(dest))))
end
drop_mtb(:AbstractGPUArrayStyle)
## AbstractWrapper
# import Base: print_array, show
function map_show_copy(WrapperType::Symbol)
    # @eval trycollect(X::$WrapperType) = device(X) == AnyGPU ? adapt(Array, X) : X
    # for dispfun in (:print_array, :show)
    #     @eval $dispfun(io::IO, X::$WrapperType{T,N}) where {T,N} =
    #         invoke($dispfun, Tuple{IO,AbstractArray{T,N}}, io, trycollect(X))
    # end

    @eval @inline copyto!(dest::$WrapperType, bc::Broadcasted{Nothing}) = begin
        device(dest) isa GPU && return gpu_copyto!(dest, bc)
        invoke(copyto!, Tuple{AbstractArray,Broadcasted{Nothing}}, dest, bc)
    end

    @eval BroadcastStyle(::Type{Base.RefValue{AT}}) where {AT<:$WrapperType} =
        BroadcastStyle(AT) |> forcedim0
end

forcedim0(x) = x
forcedim0(::Style) where {Style<:AbstractArrayStyle} = Val(0) |> Style

@require OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881" begin
    import .OffsetArrays: OffsetArray
    ## general
    ## device has been defined in ArrayInterface
    backend(::Type{T}) where {T<:OffsetArray} = ArrayInterface.parent_type(T) |> backend
    ## adapt_structure has been defined in OffsetArrays.jl
    map_show_copy(:OffsetArray)
    ## unique
    Base.collect(A::OffsetArray) = collect(parent(A))
    BroadcastStyle(::Type{OA}) where {OA<:OffsetArray} = BroadcastStyle(OA.parameters[3])
end

@require StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a" begin
    import .StructArrays: StructArray, StructArrayStyle, components
    ## general
    device(::Type{T}) where {T<:StructArray} = T.parameters[3] |> devices
    backend(::Type{T}) where {T<:StructArray} = T.parameters[3] |> backends
    ## adapt_structure has been defined in StructArrays.jl
    map_show_copy(:StructArray)
    # unique
    drop_mtb(:(StructArrayStyle{<:AbstractGPUArrayStyle}))
    forcedim0(::StructArrayStyle{Style}) where {Style} = StructArrayStyle{typeof(forcedim0(Style()))}()

    function Base.similar(bc::Broadcasted{StructArrayStyle{S}}, ::Type{ElType}) where {S,ElType}
        bc′ = convert(Broadcasted{S}, bc)
        if isstructtype(ElType)
            return StructArrays.buildfromschema(T -> similar(bc′, T), ElType)
        else
            return similar(bc′, ElType)
        end
    end
end
