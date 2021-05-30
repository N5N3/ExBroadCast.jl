using Base.Meta
const Self = @__MODULE__
GR(x, y) = GlobalRef(x, y)

const Template_mtb = IdDict(
    :broadcast!                => GR(Self, :mtb_broadcast!),
    :broadcast                 => GR(Self, :mtb_broadcast),
    GR(Self, :tab_broadcast)   => GR(Self, :mtab_broadcast),
    GR(Base, :materialize)     => GR(Self, :mtb_materialize),
    GR(Base, :materialize!)    => GR(Self, :mtb_materialize!),
    GR(Self, :tab_materialize) => GR(Self, :mtab_materialize),
)

const Template_tab = IdDict(
    :broadcast                 => GR(Self, :tab_broadcast),
    GR(Self, :mtb_broadcast)   => GR(Self, :mtab_broadcast),
    GR(Base, :materialize)     => GR(Self, :tab_materialize),
    GR(Self, :mtb_materialize) => GR(Self, :mtab_materialize),
)

const Template_mtab = IdDict(
    :broadcast!                => GR(Self, :mtb_broadcast!),
    :broadcast                 => GR(Self, :mtab_broadcast),
    GR(Base, :materialize!)    => GR(Self, :mtb_materialize!),
    GR(Base, :materialize)     => GR(Self, :mtab_materialize),
)

const Template_lzb = IdDict(
    GR(Base, :materialize)     => GR(Base.Broadcast, :instantiate),
)
## general function
using MacroTools
# use @goto and @label to repalce Core.GotoNode Expr(:gotoifnot)
function correctgoto!(code)
    @inbounds for k in eachindex(code)
        if Meta.isexpr(code[k], :gotoifnot)
            condition, tar = code[k].args
            if Meta.isexpr(code[tar], :block) #target line has been repalced
                tag = code[tar].args[1].args[1]
                code[k] =
                    Expr(:if, condition, Expr(:block), Expr(:symbolicgoto, tag))
            else
                tag = gensym(:𝐠𝐨𝐭𝐨𝐧𝐨𝐝𝐞)
                code[tar] = Expr(:block, Expr(:symboliclabel, tag), code[tar])
                code[k] =
                    Expr(:if, condition, Expr(:block), Expr(:symbolicgoto, tag))
            end
        elseif code[k] isa Core.GotoNode #target line has been repalced
            tar = code[k].label
            if Meta.isexpr(code[tar], :block)
                tag = code[tar].args[1].args[1]
                code[k] = Expr(:symbolicgoto, tag)
            else
                tag = gensym(:𝐠𝐨𝐭𝐨𝐧𝐨𝐝𝐞)
                code[tar] = Expr(:block, Expr(:symboliclabel, tag), code[tar])
                code[k] = Expr(:symbolicgoto, tag)
            end
        end
    end
    code
end
# replace temparay variables <: Union{SlotNumber,SSAValue}
replacetemp(ex, args...) = ex
replacetemp(ex::Core.Slot, SSAs, Slots) = @inbounds Slots[ex.id]
replacetemp(ex::Core.SSAValue, SSAs, Slots) = @inbounds SSAs[ex.id]
# replace broadcast(!) and materialize(!)
replacecall(ex, args...) = ex
function replacecall(ex::Expr, temps::AbstractDict)
    @inbounds if ex.head === :call
        func = ex.args[1]
        ex.args[1] = get(temps, func, func)
    end
    ex
end
# replace return
replacereturn(ex, args...) = ex
function replacereturn(ex::Expr, flag::Base.RefValue{Bool}, Result, EndPoint)
    @inbounds if ex.head === :return
        ex′₁ = Expr(:local, Expr(:(=), Result, ex.args[1]))
        ex′₂ = Expr(:symbolicgoto, EndPoint)
        flag[] = true
        return Expr(:block, ex′₁, ex′₂)
    elseif ex.head === :call && ex.args[1] === :𝐫𝐞𝐭𝐮𝐫𝐧
        return Expr(:return, ex.args[2])
    end
    ex
end

keepreturn(ex) = ex
function keepreturn(ex::Expr)
    if ex.head === :return
        @inbounds return Expr(:call, :𝐫𝐞𝐭𝐮𝐫𝐧, ex.args[1])
    end
    ex
end

addeq(ex, args...) = ex
addeq(ex::Symbol, syms::Symbol) = _addeq(ex, syms)
addeq(ex::Core.Slot, syms::Symbol) = _addeq(ex, syms)
addeq(ex::GlobalRef, syms::Symbol) = _addeq(ex, syms)
function addeq(ex::Expr, syms::Symbol)
    if ex.head === :call && ex.args[1] !== :𝐫𝐞𝐭𝐮𝐫𝐧
        return _addeq(ex, syms)
    end
    ex
end
_addeq(ex, syms) = Expr(:local, Expr(:(=), syms, ex))

if VERSION >= v"1.6.0-rc1"
    catchreturn(x) = x.val
else
    catchreturn(x) = x.args[1]
end

function combine_expr(code::Array{Any,1}, exstart, exend)
    exblock = Expr(:block)
    if ~isnothing(exstart)
        push!(exblock.args, exstart)
    end
    for i = 1:length(code)-1
        @inbounds push!(exblock.args, code[i])
    end
    if ~isnothing(exend)
        push!(exblock.args, exend)
    end
    @inbounds final = catchreturn(code[end])
    exblock, final
end

function finaladd(exblock, flag, final, Result, EndPoint)
    if flag
        push!(exblock.args, Expr(:local, Expr(:(=), Result, final)))
        push!(exblock.args, Expr(:symboliclabel, EndPoint))
        push!(exblock.args, Result)
    else
        push!(exblock.args, final)
    end
    exblock
end

function lowerhack(
    mod::Module,
    ex::Expr,
    temps,
    exstart = nothing,
    exend = nothing,
)
    ex = MacroTools.postwalk(keepreturn, ex)
    ci = Meta.lower(mod, ex)
    ci.head === :error && error(ci.args[1])
    code = first(ci.args).code
    Slots = first(ci.args).slotnames
    SSAs = [gensym(:𝐭𝐞𝐦𝐩𝐯𝐚𝐫) for _ ∈ eachindex(code)]
    EndPoint = gensym(:𝐞𝐧𝐝𝐩𝐨𝐢𝐧𝐭)
    Result = gensym(:𝐫𝐞𝐬𝐮𝐥𝐭)
    code .= addeq.(code, SSAs)
    code = correctgoto!(code)
    exblock, final = combine_expr(code, exstart, exend)
    final′ = replacetemp(final, SSAs, Slots)
    needfinal = Ref(false)
    exblock = MacroTools.postwalk(exblock) do x
        x = replacetemp(x, SSAs, Slots)
        x = replacecall(x, temps)
        x = replacereturn(x, needfinal, Result, EndPoint)
        x = MacroTools.flatten1(x)
    end
    ## final add
    finaladd(exblock, needfinal[], final′, Result, EndPoint)
end


"""
**Example:**
```julia
@mtb @. a = sin(c)
@mtb a = sin.(c)
@mtb broadcast!(sin,a,c)
@mtb a = broadcast!(sin,c)
@mtb 2 a = broadcast!(sin,c) # use 2 threads
```
"""
macro mtb(args...)
    global Template_mtb
    na = length(args)
    if na == 1
        ex = args[1]
        return esc(lowerhack(Self, ex, Template_mtb))
    elseif na == 2
        nthread, ex = args
        nthread_old = num_threads()
        ex′ = :(ExBroadcast.set_num_threads($nthread))
        ex″ = :(ExBroadcast.set_num_threads($nthread_old))
        return esc(lowerhack(Self, ex, Template_mtb, ex′, ex″))
    else
        error("Invalid input")
    end
end

"""
**Example:**
```julia
@tab @. (a,b) = sincos(c)
@tab @. a,b = sincos(c)
@tab (a,b) = @. sincos(c)
```
**Note:**
This macro is only needed for non in-place situation.
"""
macro tab(ex)
    global Template_tab
    esc(lowerhack(Self, ex, Template_tab))
end

"""
    @mtab [n] ex = @mtb [n] @tab
"""
macro mtab(args...)
    global Template_mtab
    na = length(args)
    if na == 1
        ex = args[1]
        return esc(lowerhack(Self, ex, Template_mtab))
    elseif na == 2
        nthread, ex = args
        nthread_old = num_threads()
        ex′ = :(ExBroadcast.set_num_threads($nthread))
        ex″ = :(ExBroadcast.set_num_threads($nthread_old))
        return esc(lowerhack(Self, ex, Template_mtab, ex′, ex″))
    else
        error("Invalid input")
    end
end

"""
    @lzb  return lazy Broadcasted object
"""
macro lzb(ex)
    global Template_lzb
    esc(lowerhack(Self, ex, Template_lzb))
end
