import GenParticleFilters: ParticleFilterView

export probvec, pddl_goals_to_strings, calculate_goal_probs, calculate_goal_probs_world_model, print_probs

"""
    probvec(pf::ParticleFilterView, addr)

Returns a probability vector for values of `addr` in the particle filter.
"""
function probvec(pf::ParticleFilterView, addr)
    pmap = proportionmap(pf, addr)
    keys = collect(keys(pmap))
    if hasmethod(isless, Tuple{eltype(keys), eltype(keys)})
        sort!(keys)
    end
    return [pmap[k] for k in keys]
end

function probvec(pf::ParticleFilterView, addr, support)
    pmap = proportionmap(pf, addr)
    return [get(pmap, k, 0.0) for k in support]
end

function compound_to_string(compound::Compound)
    name_str = string(compound.name)
    args_str = join(map(arg -> string(arg), compound.args), " ")
    return "($name_str $args_str)"
end

function pddl_goals_to_strings(goals)
    return map(compound_to_string, goals)
end

function get_goal_probs(traces, weights)
    probs = Dict{String, Float64}()
    for (tr, w) in zip(traces, weights)
        goal = tr[:goal]
        p = get(probs, goal, 0.0)
        probs[goal] = p + exp(w)
    end
    
    total_prob = sum(values(probs))
    for (goal, p) in probs
        probs[goal] = p / total_prob
    end
    return probs
end

"Calculate probabilities for each goal based on traces and weights."
function calculate_goal_probs(traces, weights)
    probs = Dict{String, Float64}()
    for (tr, w) in zip(traces, weights)
        goal = tr[:goal]
        p = get(probs, goal, 0.0)
        probs[goal] = p + exp(w)
    end
    
    total_prob = sum(values(probs))
    for (goal, p) in probs
        probs[goal] = p / total_prob
    end
    return probs
end

"Calculate probabilities for each goal based on traces and weights."
function calculate_goal_probs_world_model(traces, weights)
    probs = Dict{Int64, Float64}()
    for (tr, w) in zip(traces, weights)
        goal = tr[:init => :agent => :goal => :goal]
        p = get(probs, goal, 0.0)
        probs[goal] = p + exp(w)
    end
    
    total_prob = sum(values(probs))
    for (goal, p) in probs
        probs[goal] = p / total_prob
    end
    return probs
end

"Print goals and their probabilities."
function print_probs(traces, weights)
    probs = calculate_goal_probs(traces, weights)
    for (goal, p) in probs
        println("Goal: ", goal)
        println("Probability: ", round(p, digits=4))
    end  
end