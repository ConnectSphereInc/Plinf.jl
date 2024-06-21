using Base: @kwdef
using Parameters: @unpack
using SymbolicPlanners, Plinf
using SymbolicPlanners: PathNode, simplify_goal, LinkedNodeRef, @auto_hash, @auto_equals, reconstruct
using PDDL

"""
    RandomPlanner(;
        max_nodes::Int = typemax(Int),
        max_time::Float64 = Inf,
        save_search::Bool = false,
        save_search_order::Bool = save_search,
        verbose::Bool = false,
        callback = verbose ? LoggerCallback() : nothing
    )

Random search planner. Nodes are expanded in a random order.

Returns a [`PathSearchSolution`](@ref) or [`NullSolution`](@ref).
"""
# @kwdef mutable struct RandomPlanner <: Planner
mutable struct RandomPlanner
    max_nodes::Int
    max_time::Float64
    save_search::Bool
    save_search_order::Bool
    verbose::Bool
    callback::Union{Nothing, Function}

    RandomPlanner(;max_nodes::Int=typemax(Int), max_time::Float64=Inf,
                  save_search::Bool=false, save_search_order::Bool=save_search,
                  verbose::Bool=false, callback::Union{Nothing, Function}=verbose ? LoggerCallback() : nothing) =
        new(max_nodes, max_time, save_search, save_search_order, verbose, callback)
end

@auto_hash RandomPlanner
@auto_equals RandomPlanner

function (planner::RandomPlanner)(domain::Domain, state::State, spec::Specification)
    solve(planner, domain, state, spec)
end

function (planner::RandomPlanner)(domain::Domain, state::State, goals)
    solve(planner, domain, state, Specification(goals))
end

function Base.copy(p::RandomPlanner)
    return RandomPlanner(p.max_nodes, p.max_time,
                         p.save_search, p.save_search_order,
                         p.verbose, p.callback)
end

function solve(planner::RandomPlanner,
               domain::Domain, state::State, spec::Specification)

    @unpack save_search = planner
    spec = simplify_goal(spec, domain, state)
    node_id = hash(state)
    node = PathNode(node_id, state, 0.0, LinkedNodeRef(node_id))

    search_tree = Dict(node_id => node)
    queue = [node_id]
    search_order = UInt[]
    sol = PathSearchSolution(:in_progress, Term[], Vector{typeof(state)}(),
                             0, search_tree, queue, search_order)
    sol = search!(sol, planner, domain, spec, state)

    if save_search
        return sol
    elseif sol.status == :failure
        return NullSolution(sol.status)
    else
        return PathSearchSolution(sol.status, sol.plan, sol.trajectory)
    end
end

function search!(sol::PathSearchSolution, planner::RandomPlanner,
                 domain::Domain, spec::Specification, state::State)
    start_time = time()
    sol.expanded = 0
    queue, search_tree = sol.search_frontier, sol.search_tree
    visited = Set{Tuple{Int64,Int64}}()
    initial_agent_pos = get_agent_pos(state)
    push!(visited, initial_agent_pos)

    while length(queue) > 0
        # Randomly select a state from the queue
        node_id = queue[rand(1:end)]
        node = search_tree[node_id]

        if is_goal(spec, domain, node.state, node.parent.action)
            sol.status = :success # Goal reached
        elseif sol.expanded >= planner.max_nodes
            sol.status = :max_nodes # Node budget reached
        elseif time() - start_time >= planner.max_time
            sol.status = :max_time # Time budget reached
        end

        if sol.status == :in_progress # Expand current node
            popfirst!(queue)
            expand!(planner, node, search_tree, queue, domain, spec, visited)
            sol.expanded += 1
            if planner.save_search && planner.save_search_order
                push!(sol.search_order, node_id)
            end
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, sol.expanded)
            end
        else # Reconstruct plan and return solution
            sol.plan, sol.trajectory = reconstruct(node_id, search_tree)
            if !isnothing(planner.callback)
                planner.callback(planner, sol, node_id, sol.expanded)
            end
            return sol
        end
    end
    sol.status = :failure
    return sol
end


function expand!(
    planner::RandomPlanner, node::PathNode{S},
    search_tree::Dict{UInt,PathNode{S}}, queue::Vector{UInt},
    domain::Domain, spec::Specification, visited::Set{Tuple{Int64,Int64}}
) where {S <: State}
    state = node.state

    available_actions = [act for act in available(domain, state)]
    if isempty(available_actions)
        return  # Return if there is no direction to proceed
    end

    # Remove visited states from available actions as much as possible
    unvisited_and_available_actions = []
    for act in available_actions
        next_state = transition(domain, state, act, check=false)
        next_agent_pos = get_agent_pos(next_state)
        if !(next_agent_pos in visited)
            push!(unvisited_and_available_actions, act)
        end
    end

    if isempty(unvisited_and_available_actions)
        unvisited_and_available_actions = available_actions
    end

    # Randomly select actions and transitions
    chosen_action = unvisited_and_available_actions[rand(1:length(unvisited_and_available_actions))]
    next_state = transition(domain, state, chosen_action, check=false)
    next_id = hash(next_state)
    next_agent_pos = get_agent_pos(next_state)

    # Add new state to search tree and queue
    if !haskey(search_tree, next_id)
        path_cost = node.path_cost + 1
        search_tree[next_id] = PathNode(next_id, next_state, path_cost, LinkedNodeRef(node.id, chosen_action))
        push!(visited, next_agent_pos)
    end
    push!(queue, next_id)
end

# Get agent position from CompiledState
# return: (xpos, ypos)
function get_agent_pos(state::State)
    return (PDDL.GenericState(state).values[:xpos], PDDL.GenericState(state).values[:ypos])
end