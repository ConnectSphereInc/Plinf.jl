import Gen: importance_sampling, DynamicChoiceMap
export VisionEnhancedPlanSearch, VIPS
export vips_run

"""
    VisionEnhancedPlanSearch()

Constructs an importance sampling algorithm
for the agent-environment model defined by `world_config`.

# Arguments

$(TYPEDFIELDS)
"""

@kwdef struct VisionEnhancedPlanSearch{W <: WorldConfig, D <: Domain}
    "Configuration of world model to perform inference over."
    world_config::W
    domain::D
end

const VIPS = VisionEnhancedPlanSearch

VIPS(world_config, domain; kwargs...) =VIPS(; world_config=world_config, domain=domain, kwargs...)

"""
    (::vips)(n_samples, obs_traj, obs_terms; kwargs...)

Runs VIPS
"""
(vips::VIPS)(args...; kwargs...) = vips_run(vips, args...; kwargs...)

function vips_run(vips::VIPS, n_samples::Int, obs_traj::Vector{State}, obs_terms::Vector{Term}; init_args=Dict{Symbol, Any}())
    num_steps = length(obs_traj)

    # Split the time series of observations into overlapping groups of two
    for t in 1:num_steps-1
        # Construct a choicemap containing the current state and the next state
        obs_current::ChoiceMap = state_choicemap(obs_traj[t], obs_terms)
        obs_next::ChoiceMap = state_choicemap(obs_traj[t+1], obs_terms)
        total_observations = choicemap()

        # Set the next observation
        set_submap!(total_observations, :init => :obs, obs_current)
        set_submap!(total_observations, :timestep => 1 => :obs, obs_next)

        (traces, log_norm_weights, _) = importance_sampling(world_model, (1, vips.world_config), total_observations, n_samples)

        # Calculate goal probabilities
        goal_probs = zeros(2)
        for (tr, log_weight) in zip(traces, log_norm_weights)
            goal = tr[:init => :agent => :goal => :goal]
            goal_probs[goal] += exp(log_weight)
        end
        goal_probs ./= sum(goal_probs)

        # Print goal probabilities
        println("t = $t: goal 1 prob = $(goal_probs[1]), goal 2 prob = $(goal_probs[2])")

        # Update the environment configuration with the next observed state
        vips.world_config.env_config = PDDLEnvConfig(vips.domain, obs_traj[t+1])
    end

    return 0
end