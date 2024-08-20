using Base: @kwdef
using Parameters: @unpack
using SymbolicPlanners, Plinf
using SymbolicPlanners: PathNode, simplify_goal, LinkedNodeRef, @auto_hash, @auto_equals, reconstruct
using PDDL
using DataStructures: CircularBuffer

mutable struct SingleStepVisionPlanner <: Planner
    max_nodes::Int
    max_time::Float64
    save_search::Bool
    save_search_order::Bool
    verbose::Bool
    callback::Union{Nothing, Function}
    alternative_planner::Union{Nothing, Planner}
    agent::Symbol
    visited_positions::CircularBuffer{Tuple{Int, Int}}
    max_history::Int
    recency_weight::Float64
    believed_gem_position::Union{Nothing, Tuple{Int, Int}}

    function SingleStepVisionPlanner(;
        max_nodes::Int=typemax(Int),
        max_time::Float64=Inf,
        save_search::Bool=false,
        agent::Symbol=:agent,
        alternative_planner=AStarPlanner(GoalManhattan(), save_search=true),
        save_search_order::Bool=save_search,
        verbose::Bool=false,
        callback::Union{Nothing, Function}=verbose ? LoggerCallback() : nothing,
        max_history::Int=50,
        recency_weight::Float64=0.9
    )
        new(max_nodes, max_time, save_search, save_search_order, verbose, callback,
            alternative_planner, agent, CircularBuffer{Tuple{Int, Int}}(max_history),
            max_history, recency_weight, nothing)
    end
end

@auto_hash SingleStepVisionPlanner
@auto_equals SingleStepVisionPlanner

function (planner::SingleStepVisionPlanner)(domain::Domain, state::State, goal)
    solve(planner, domain, state, Specification(goal))
end

function (planner::SingleStepVisionPlanner)(domain::Domain, state::State, goal::MinStepsGoal)
    solve(planner, domain, state, Specification(goal.terms))
end

function solve(planner::SingleStepVisionPlanner, domain::Domain, state::State, spec::Specification)
    @unpack save_search, agent, alternative_planner, max_history, recency_weight, believed_gem_position = planner
    
    items = extract_items_from_spec(spec)
    available_actions = collect(PDDL.available(domain, state))
    
    current_pos = get_agent_pos(state, agent)
    push!(planner.visited_positions, current_pos)

    function is_agent_action(act)
        try
            return act.args[1].name == agent
        catch
            return false
        end
    end

    agent_actions = filter(is_agent_action, available_actions)

    # Check for communication actions
    communicate_actions = filter(act -> act.name == :communicate, agent_actions)
    if !isempty(communicate_actions)
        chosen_action = first(communicate_actions)
        return create_single_step_solution(planner, domain, state, chosen_action, save_search)
    end

    # Check for visible items
    visible_items = filter(item -> PDDL.satisfy(domain, state, pddl"(visible $agent $item)"), items)
    if !isempty(visible_items)
        item = first(visible_items)
        subgoal = PDDL.parse_pddl("(has $agent $item)")
        a_star_sol = alternative_planner(domain, state, subgoal)
        if !isempty(a_star_sol.plan)
            return create_single_step_solution(planner, domain, state, first(a_star_sol.plan), save_search)
        end
    end

    # If we have a believed gem position, plan to that location
    if !isnothing(believed_gem_position)
        target_x, target_y = believed_gem_position
        subgoal = PDDL.parse_pddl("(and (= (xloc $agent) $target_x) (= (yloc $agent) $target_y))")
        a_star_sol = alternative_planner(domain, state, subgoal)
        if !isempty(a_star_sol.plan)
            next_action = first(a_star_sol.plan)
            next_state = PDDL.transition(domain, state, next_action)
            next_pos = get_agent_pos(next_state, agent)
            
            # If we've reached the believed position, check for a gem
            if next_pos == believed_gem_position
                pickup_actions = filter(act -> act.name == :pickup, agent_actions)
                if !isempty(pickup_actions)
                    return create_single_step_solution(planner, domain, state, first(pickup_actions), save_search)
                else
                    # If no gem found, reset the belief and continue random search
                    planner.believed_gem_position = nothing
                end
            else
                return create_single_step_solution(planner, domain, state, next_action, save_search)
            end
        end
    end

    # If no items are visible or A* failed, take a move action based on recency-weighted probabilities
    move_actions = filter(act -> act.name in [:up, :down, :left, :right], agent_actions)
    if !isempty(move_actions)
        chosen_action = choose_recency_weighted_action(planner, domain, state, move_actions)
        return create_single_step_solution(planner, domain, state, chosen_action, save_search)
    end

    # If no actions are available, return a failure
    return NullSolution(:failure)
end

function choose_recency_weighted_action(planner::SingleStepVisionPlanner, domain::D, state::S, move_actions::Vector{T}) where {D, S, T}
    @unpack agent, visited_positions, recency_weight, max_history = planner
    
    # Calculate weights for each action
    weights = Float64[]
    for action in move_actions
        next_state = PDDL.transition(domain, state, action)
        next_pos = get_agent_pos(next_state, agent)
        
        if next_pos in visited_positions
            index = findfirst(pos -> pos == next_pos, visited_positions)
            # Exponential decay based on recency, with a stronger penalty for recent visits
            weight = recency_weight^(3 * (length(visited_positions) - index) / max_history)
        else
            # Strongly favor unvisited positions
            weight = 10.0
        end
        
        push!(weights, weight)
    end
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = weights ./ total_weight
    
    # Choose action based on weighted probabilities
    r = rand()
    cumulative_weight = 0.0
    for (i, weight) in enumerate(normalized_weights)
        cumulative_weight += weight
        if r <= cumulative_weight
            return move_actions[i]
        end
    end
    
    # Fallback to last action (should rarely happen due to floating-point precision)
    return move_actions[end]
end

function create_single_step_solution(planner::SingleStepVisionPlanner, domain::Domain, state::State, action::Term, save_search::Bool)
    next_state = PDDL.transition(domain, state, action)
    plan = Term[action]
    trajectory = [state, next_state]

    if save_search
        search_tree = Dict{UInt, PathNode{typeof(state)}}()
        search_tree[hash(state)] = PathNode(hash(state), state, 0.0, LinkedNodeRef(hash(state)))
        search_tree[hash(next_state)] = PathNode(hash(next_state), next_state, 1.0, LinkedNodeRef(hash(state), action))
        search_frontier = [hash(next_state)]
        search_order = [hash(state), hash(next_state)]
        return PathSearchSolution{typeof(state), typeof(search_frontier)}(
            :success, plan, trajectory, 1, search_tree, search_frontier, search_order
        )
    else
        return PathSearchSolution{typeof(state), Nothing}(
            :success, plan, trajectory, 1, nothing, nothing, UInt64[]
        )
    end
end

function Base.copy(planner::SingleStepVisionPlanner)
    SingleStepVisionPlanner(
        max_nodes = planner.max_nodes,
        max_time = planner.max_time,
        save_search = planner.save_search,
        save_search_order = planner.save_search_order,
        verbose = planner.verbose,
        callback = planner.callback,
        alternative_planner = isnothing(planner.alternative_planner) ? nothing : copy(planner.alternative_planner),
        agent = planner.agent,
        max_history = planner.max_history,
        recency_weight = planner.recency_weight
    )
end