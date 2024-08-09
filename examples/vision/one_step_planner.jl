using Base: @kwdef
using Parameters: @unpack
using SymbolicPlanners, Plinf
using SymbolicPlanners: PathNode, simplify_goal, LinkedNodeRef, @auto_hash, @auto_equals, reconstruct
using PDDL

mutable struct SingleStepVisionPlanner <: Planner
    max_nodes::Int
    max_time::Float64
    save_search::Bool
    save_search_order::Bool
    verbose::Bool
    callback::Union{Nothing, Function}
    alternative_planner::Union{Nothing, Planner}
    agent::Symbol
    recent_positions::Dict{Symbol, Vector{Tuple{Int, Int}}}
    memory_length::Int

    function SingleStepVisionPlanner(;max_nodes::Int=typemax(Int), max_time::Float64=Inf,
                  save_search::Bool=false, agent::Symbol=:nothing, alternative_planner=AStarPlanner(GoalManhattan(), save_search=true), save_search_order::Bool=save_search,
                  verbose::Bool=false, callback::Union{Nothing, Function}=verbose ? LoggerCallback() : nothing, memory_length::Int=5)
        new(max_nodes, max_time, save_search, save_search_order, verbose, callback, alternative_planner, agent, Dict{Symbol, Vector{Tuple{Int, Int}}}(), memory_length)
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
    @unpack save_search, agent, alternative_planner, memory_length = planner
    
    items = extract_items_from_spec(spec)
    available_actions = collect(PDDL.available(domain, state))
    
    # Initialize recent positions for this agent if not already present
    if !haskey(planner.recent_positions, agent)
        planner.recent_positions[agent] = Vector{Tuple{Int, Int}}()
    end
    
    current_pos = get_agent_pos(state, agent)
    planner.recent_positions[agent] = update_recent_positions(planner.recent_positions[agent], current_pos)

    # Function to safely check if an action belongs to the agent
    function is_agent_action(act)
        try
            return act.args[1].name == agent
        catch
            return false
        end
    end

    # Safely filter agent actions
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
        # Manually create the subgoal PDDL term
        item = first(visible_items)
        subgoal = PDDL.parse_pddl("(has $agent $item)")
        a_star_sol = alternative_planner(domain, state, subgoal)
        if !isempty(a_star_sol.plan)
            return create_single_step_solution(planner, domain, state, first(a_star_sol.plan), save_search)
        end
    end

    # If no items are visible or A* failed, take a random move action without revisiting recent positions
    move_actions = filter(act -> act.name in [:up, :down, :left, :right], agent_actions)
    if !isempty(move_actions)
        valid_actions = filter(act -> !leads_to_recent_position(domain, state, act, planner.recent_positions[agent], memory_length), move_actions)
        
        if !isempty(valid_actions)
            chosen_action = rand(valid_actions)
        else
            # If all actions lead to recent positions, choose any random action
            chosen_action = rand(move_actions)
        end
        
        return create_single_step_solution(planner, domain, state, chosen_action, save_search)
    end

    # If no actions are available, return a failure
    return NullSolution(:failure)
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
        memory_length = planner.memory_length
    )
end
