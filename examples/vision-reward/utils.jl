using DataStructures: OrderedDict
using PDDLViz: RGBA, to_color, set_alpha
using Base: @kwdef

"Gets the (x, y) position of the specified agent."
function get_agent_pos(state::State, agent::Symbol)
    return (state[Compound(:xloc, Term[Const(agent)])],
            state[Compound(:yloc, Term[Const(agent)])])
end

"Gets the (x, y) location of an object."
get_obj_loc(state::State, obj::Const) =
    (state[Compound(:xloc, Term[obj])], state[Compound(:yloc, Term[obj])])

function extract_items_from_spec(spec::Specification)
    goals = spec isa MinStepsGoal ? spec.terms : spec.terms
    goals = goals isa Vector && length(goals) == 1 ? goals[1] : goals
    subgoals = goals isa Term && goals.name == :or ? goals.args :
               goals isa Vector ? goals : [goals]
    
    return [subgoal.args[2].name for subgoal in subgoals 
            if subgoal isa Term && subgoal.name == :has && length(subgoal.args) == 2]
end

"""
    get_top_weighted_rewards(state::ParticleFilterState, n::Int)

    Get the top `n` most likely reward distributions based on the particle filter state.

"""
function get_top_weighted_rewards(state::ParticleFilterState, n::Int)
    traces = get_traces(state)
    weights = get_norm_weights(state)
    reward_weights = Dict()
    
    # Accumulate rewards and weights
    for (tr, w) in zip(traces, weights)
        rewards = Dict{String, String}()
        for gem in possible_gems
            rewards[gem] = string(tr[:reward => Symbol(gem)])
        end
        rewards_tuple = Tuple(sort(collect(rewards)))
        reward_weights[rewards_tuple] = get(reward_weights, rewards_tuple, 0.0) + w
    end
    
    total_weight = sum(values(reward_weights))
    weighted_rewards = [(Dict(rewards), weight / total_weight) 
                        for (rewards, weight) in reward_weights]
    
    # Sort by probability
    sort!(weighted_rewards, by = x -> x[2], rev = true)
    
    # Return top n results
    return weighted_rewards[1:min(n, length(weighted_rewards))]
end

function quantify_gem_certainty(weighted_rewards)
    total_weight = sum(wr[2] for wr in weighted_rewards)
    gem_values = Dict(gem => Dict() for gem in keys(weighted_rewards[1][1]))
    for (rewards, weight) in weighted_rewards
        for (gem, value) in rewards
            value_float = parse(Float64, value)
            if !haskey(gem_values[gem], value_float)
                gem_values[gem][value_float] = 0.0
            end
            gem_values[gem][value_float] += weight
        end
    end
    gem_certainty = Dict()
    for (gem, values) in gem_values
        probability, most_likely_value = findmax(values)
        certainty = (probability / total_weight) * 100
        gem_certainty[gem] = Dict(
            "most_likely_value" => most_likely_value,
            "certainty_percentage" => round(certainty, digits=1)
        )
    end
    return gem_certainty
end

function calculate_gem_utility(gem_certainty; risk_aversion=0.5)
    utilities = Dict()
    for (gem, info) in gem_certainty
        # Check if the value is already a number, if not, parse it
        value = if isa(info["most_likely_value"], Number)
            info["most_likely_value"]
        else
            parse(Float64, info["most_likely_value"])
        end
        certainty = info["certainty_percentage"] / 100
        
        # Calculate expected utility
        # Higher risk_aversion puts more weight on certainty
        utility = (1 - risk_aversion) * value + risk_aversion * certainty * value
        
        utilities[gem] = Dict(
            "value" => value,
            "certainty" => certainty,
            "utility" => utility
        )
    end
    return utilities
end

"""
    gem_from_utterance(utterance::String)

    Manually extract the color of a gem from an utterance using regex.
"""
function parse_gem(utterance::String)
    # First, try to match full color names
    color_pattern = r"\b(red|blue|yellow|green)\b"
    match_result = match(color_pattern, lowercase(utterance))
    if match_result !== nothing
        return String(match_result.match)
    end
    
    # If no match, try to extract color from gem names like "blue_gem2"
    gem_pattern = r"\b(red|blue|yellow|green)_gem\d*\b"
    match_result = match(gem_pattern, lowercase(utterance))
    if match_result !== nothing
        return split(match_result.match, "_")[1]
    end
    
    return nothing
end

"""
    parse_reward(utterance::String, gem::String)

    Manually extract the reward from an utterance using regex.
"""
function parse_reward(utterance::String)
    score_pattern = r"\b(-1|1|3|5)\b"
    match_result = match(score_pattern, utterance)
    if match_result !== nothing
        return parse(Int, match_result.match)
    end
    return nothing
end

