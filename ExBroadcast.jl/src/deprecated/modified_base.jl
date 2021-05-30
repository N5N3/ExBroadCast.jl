# const AbstractScalar{T} = Union{T,AbstractArray{T,0},Ref{T},Tuple{T}}
# using Preferences
# function get_experiment()
#     default_set = "off"
#     set = @load_preference("experiment", default_set)
#     if set ∉ ("on", "off")
#         @error("Invalid setting \"$(provider)\"; valid settings include [\"on\", \"off\"], defaulting to \"off\"")
#         set = default_set
#     end
#     return set
# end

# # Read in preferences, see if any users have requested a particular backend
# const experiment = get_experiment()

# function set_experiment!(set; export_prefs::Bool = false)
#     if set !== nothing && set !== missing && set ∉ ("on", "off")
#         throw(ArgumentError("Invalid set '$(set)'"))
#     end
#     set_preferences!(@__MODULE__, "experiment" => set; export_prefs, force = true)
#     if set != experiment
#         # Re-fetch to get default values in the event that `nothing` or `missing` was passed in.
#         set = get_experiment()
#         @info("Experiment function changed; restart Julia for this change to take effect", set)
#     end
# end

# @static if experiment == "on"
#     @warn "Experiment on, some base behaviour changed"
#     include("nonlazyscalar.jl")
# end
# include("lazygenerator.jl")
