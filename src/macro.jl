using Base.Meta
const Self = @__MODULE__
const Template_mtb = IdDict(
    GlobalRef(Base, :materialize) => GlobalRef(Self, :mtb_materialize),
    GlobalRef(Base, :materialize!) => GlobalRef(Self, :mtb_materialize!),
    GlobalRef(Self, :tab_materialize) => GlobalRef(Self, :mtab_materialize),
)
const Template_tab = IdDict(
    GlobalRef(Base, :materialize) => GlobalRef(Self, :tab_materialize),
    GlobalRef(Self, :mtb_materialize) => GlobalRef(Self, :mtab_materialize),
)
const Template_mtab = IdDict(
    GlobalRef(Base, :materialize!) => GlobalRef(Self, :mtb_materialize!),
    GlobalRef(Base, :materialize) => GlobalRef(Self, :mtab_materialize),
)

const Template_lzb = IdDict(
    GlobalRef(Base, :materialize) => GlobalRef(Base.Broadcast, :instantiate),
)
## general function
using MacroTools
# use @goto and @label to repalce Core.GotoNode Expr(:gotoifnot)
function correctgoto!(code)
    get_tag(tar) = begin
        if !Meta.isexpr(code[tar], :block)
            code[tar] = Expr(:block, Expr(:symboliclabel, Symbol(tag, tar)), code[tar])
        end
        Symbol(tag, tar)
    end
    replace(ex::Core.GotoIfNot) = Expr(:||, ex.cond, Expr(:symbolicgoto, get_tag(ex.dest)))
    replace(ex::Core.GotoNode) = Expr(:symbolicgoto, get_tag(ex.label))
    replace(ex) = begin
        if Meta.isexpr(ex, :block)
            ex.args[2] = replace(ex.args[2])
        end
        ex
    end
    tag = gensym(:𝐠𝐨𝐭𝐨𝐧𝐨𝐝𝐞)
    code .= replace.(code)
end
# replace temparay variables <: Union{SlotNumber,SSAValue}
replacetemp(syms) = Base.Fix2(replacetemp, syms)
replacetemp(ex, syms) = ex
replacetemp(ex::Core.Slot, syms) = Symbol(syms, "_slot_", ex.id)
replacetemp(ex::Core.SSAValue, syms) = Symbol(syms, "_ssa_", ex.id)
# replace broadcast(!) and materialize(!)
function replacecall(ex, temps::AbstractDict, addargs)
    Meta.isexpr(ex, :call) || return ex
    func = ex.args[1]
    ex.args[1] = get(temps, func, func)
    if !isnothing(addargs) && ex.args[1] !== func
        push!(ex.args, addargs)
    end
    ex
end
# replace return
function replacereturn(ex, Result, EndPoint) 
    ex isa Core.ReturnNode && return Expr(:block, :(local $Result = $(ex.val)), Expr(:symbolicgoto, EndPoint))
    Meta.isexpr(ex, :call) && ex.args[1] === :𝐫𝐞𝐭𝐮𝐫𝐧 && return :(return $(ex.args[2]))
    ex
end

function expandbroadcast(ex)
    Meta.isexpr(ex, :call) || return ex
    if ex.args[1] === :broadcast
        ex.args[1] = :(Base.broadcasted)
        return :(Base.materialize($ex))
    elseif ex.args[1] === :broadcast!
        ex′ = Expr(:call, :(Base.broadcasted), ex.args[2], ex.args[4:end]...)
        return :(Base.materialize!($(ex.args[3]), $ex′))
    end
    ex
end

function keepreturn(ex)
    Meta.isexpr(ex, :return) || return ex
    Expr(:call, :𝐫𝐞𝐭𝐮𝐫𝐧, ex.args[1])
end

function addeq(ex, syms::Symbol, id)
    Meta.isexpr(ex, :call) ||
        ex isa Symbol ||
        ex isa Core.Slot ||
        ex isa GlobalRef ||
        return ex
    :(local $(Symbol(syms, "_ssa_", id)) = $ex)
end

function lowerhack(mod::Module, ex::Expr, temps, addarg = nothing)
    ex = MacroTools.postwalk(∘(keepreturn, expandbroadcast), ex) 
    ci = Meta.lower(mod, ex)
    ci.head === :error && error(ci.args[1])
    code = only(ci.args).code
    TempVar = gensym(:𝐭𝐞𝐦𝐩𝐯𝐚𝐫)
    Result = gensym(:𝐫𝐞𝐬𝐮𝐥𝐭)
    EndPoint = gensym(:𝐞𝐧𝐝𝐩𝐨𝐢𝐧𝐭)
    proc(x, i) = begin
        x = replacereturn(x, Result, EndPoint)
        x = replacecall(x, temps, addarg)
        addeq(x, TempVar, i)
    end
    code .= proc.(code, 1:length(code))
    exblock = Expr(:block, correctgoto!(code)...)
    exblock = MacroTools.postwalk(replacetemp(TempVar), exblock)
    push!(exblock.args, Expr(:symboliclabel, EndPoint))
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
        nthread, ex = num_threads(), args[1]
    elseif na == 2
        nthread, ex = args
    else
        error("Invalid input")
    end
    (!isa(nthread, Integer) || nthread <= 1) && return esc(ex)
    nthread = min(nthread, Threads.nthreads())
    return esc(lowerhack(Self, ex, Template_mtb, nthread))
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
    global Template_tab
    na = length(args)
    if na == 1
        nthread, ex = num_threads(), args[1]
    elseif na == 2
        nthread, ex = args
    else
        error("Invalid input")
    end
    (!isa(nthread, Integer) || nthread <= 1) &&
        return esc(lowerhack(Self, ex, Template_tab))
    nthread = min(nthread, Threads.nthreads())
    return esc(lowerhack(Self, ex, Template_mtab, nthread))
end

"""
    @lzb  return lazy Broadcasted object
"""
macro lzb(ex)
    global Template_lzb
    esc(lowerhack(Self, ex, Template_lzb))
end
