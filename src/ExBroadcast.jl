module ExBroadcast

export @mtb, @tab, @mtab, @lzb

const _NTHREADS = Ref(Threads.nthreads())
@inline num_threads() = _NTHREADS[]

"""
    ExBroadcast.set_num_threads(n)
Set the threads' num.
"""
function set_num_threads(n::Int)
    _NTHREADS[] = max(1,min(n, Threads.nthreads()))
    nothing
end
include("thread_provider.jl")

include("util.jl")
include("macro_kernal\\mtb.jl")
include("macro_kernal\\tab.jl")
include("macro_kernal\\mtab.jl")
include("macro.jl")

# include("experiment\\modified_base.jl")

using Requires
@init @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("other_support\\gpusupport.jl")
@init @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("other_support\\staticarray.jl")
@init @require Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59" include("other_support\\interp.jl")


end
