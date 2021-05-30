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
    function get_tag(tar)
        if Meta.isexpr(code[tar], :block) #target line has been repalced
            tag = code[tar].args[1].args[1]
        else
            tag = gensym(:𝐠𝐨𝐭𝐨𝐧𝐨𝐝𝐞)
            code[tar] = Expr(:block, Expr(:symboliclabel, tag), code[tar])
        end
        tag
    end
    @inbounds for k in eachindex(code)
        if code[k] isa Core.GotoIfNot
            code[k] = Expr(:if, code[k].cond, Expr(:block), Expr(:symbolicgoto, get_tag(code[k].dest)))
        elseif code[k] isa Core.GotoNode 
            code[k] = Expr(:symbolicgoto, get_tag(code[k].label))
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
replacereturn(ex::Core.ReturnNode, Result, EndPoint) = 
    Expr(:block, :(local $Result = $(ex.val)), Expr(:symbolicgoto, EndPoint))
function replacereturn(ex::Expr, Result, EndPoint)
    ex.head === :call && ex.args[1] === :𝐫𝐞𝐭𝐮𝐫𝐧 && return :(return $(ex.args[2]))
    ex
end

function keepreturn(ex)
    Meta.isexpr(ex, :return) && return Expr(:call, :𝐫𝐞𝐭𝐮𝐫𝐧, ex.args[1])
    ex
end

function addeq(ex, syms::Symbol)
    Meta.isexpr(ex, :call) || ex isa Symbol || ex isa Core.Slot || ex isa GlobalRef || return ex
    :(local $syms = $ex)
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
    SSAs = [gensym(:𝐭𝐞𝐦𝐩𝐯𝐚𝐫) for _ in eachindex(code)]
    Result, EndPoint = gensym(:𝐫𝐞𝐬𝐮𝐥𝐭), gensym(:𝐞𝐧𝐝𝐩𝐨𝐢𝐧𝐭)
    code .= addeq.(code, SSAs)
    code .= replacereturn.(code, Result, EndPoint)
    exblock = Expr(:block, correctgoto!(code)...)
    exblock = MacroTools.postwalk(exblock) do x
        x = replacetemp(x, SSAs, Slots)
        x = replacecall(x, temps)
        x = MacroTools.flatten1(x)
    end
    ## final add
    exblock.args[end] = Expr(:symboliclabel, EndPoint)
    push!(exblock.args, Result)
    exblock
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
