## GPU support
import GPUArrays: BroadcastGPUArray, backend,
                launch_heuristic, launch_configuration, gpu_call,
                linear_index, global_size, @cartesianidx
const BGA = BroadcastGPUArray
device(::BGA) = AnyGPU

# in-place soa support
backend(A::TupleDummy) = backends(parent(A))
@inline backends(nt::NamedTuple) = values(nt) |> backends
@inline backends(t::Tuple) = getsame(backend, t...)

# Adapt fix
import Adapt: adapt_structure, adapt
@inline adapts(to, x, ys...) = (adapt(to, x), adapts(to, ys...)...)
@inline adapts(to) = ()
adapt_structure(to, td::TupleDummy{T,N,L}) where {T,N,L} = 
    TupleDummy{T,N,L}(adapts(to, td.arrays...), td.ax)
    
@inline function copyto!(dest::TupleDummy, bc::Broadcasted{Nothing})
    device(dest) == AnyGPU && return gpu_copyto!(dest, bc)
    invoke(copyto!, Tuple{AbstractArray, Broadcasted{Nothing}}, dest, bc)
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
    bc′ = Broadcast.preprocess(dest, bc)

    function broadcast_kernel(ctx, dest, bc′, nelem)
        Inds = eachindex(bc′)
        Iˢ, Iᵉ = Ref(Inds) .|> (firstindex, lastindex)
        @inbounds for i in 1:nelem
            j = linear_index(ctx, i) + Iˢ - 1
            j > Iᵉ && return nothing
            I = Inds[j]
            dest[I] = bc′[I]
        end
        nothing
    end

    heuristic = launch_heuristic(backend(dest), broadcast_kernel, dest, bc′, 1)
    config = launch_configuration(backend(dest), heuristic, length(dest), typemax(Int))
    gpu_call(broadcast_kernel, dest, bc′, config.elements_per_thread;
             threads=config.threads, blocks=config.blocks)
    dest
end

## AbstractWrapper
import Base: print_array, show
function map_show_copy(WrapperType::Symbol) 
    @eval trycollect(X::$WrapperType) = device(X) == AnyGPU ? adapt(Array,X) : X
    for dispfun in (:print_array, :show)
        @eval $dispfun(io::IO, X::$WrapperType{T,N}) where {T,N} = 
            invoke($dispfun, Tuple{IO, AbstractArray{T,N}}, io, trycollect(X))
    end

    @eval @inline copyto!(dest::$WrapperType, bc::Broadcasted{Nothing}) = begin
        device(dest) == AnyGPU && return gpu_copyto!(dest, bc)
        invoke(copyto!, Tuple{AbstractArray, Broadcasted{Nothing}}, dest, bc)
    end

    @eval BroadcastStyle(::Type{Base.RefValue{AT}}) where AT<:$WrapperType =
        BroadcastStyle(AT) |> forcedim0
end

forcedim0(x) = x
forcedim0(::Style) where Style <: AbstractArrayStyle = Val(0) |> Style

@require OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881" begin
    import .OffsetArrays: OffsetArray
    ## general
    device(A::OffsetArray) = parent(A) |> device
    backend(A::OffsetArray) = parent(A) |> backend
    ## adapt_structure has been defined in OffsetArrays.jl
    map_show_copy(:OffsetArray)
    ## unique
    Base.collect(A::OffsetArray) = collect(parent(A))
    BroadcastStyle(::Type{OA}) where OA<:OffsetArray = BroadcastStyle(OA.parameters[3])
end

@require StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a" begin
    import .StructArrays: StructArray, StructArrayStyle, components
    ## general
    device(A::StructArray) = components(A) |> devices
    backend(A::StructArray) = components(A) |> backends
    ## adapt_structure has been defined in StructArrays.jl
    map_show_copy(:StructArray)
    # unique
    forcedim0(::StructArrayStyle{Style}) where Style = StructArrayStyle{typeof(forcedim0(Style()))}()

    function Base.similar(bc::Broadcasted{StructArrayStyle{S}}, ::Type{ElType}) where {S,ElType}
        bc′ = convert(Broadcasted{S}, bc)
        if isstructtype(ElType)
            return StructArrays.buildfromschema(T -> similar(bc′, T), ElType)
        else
            return similar(bc′, ElType)
        end
    end
end
