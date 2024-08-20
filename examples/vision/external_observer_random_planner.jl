using Base: @kwdef
using Parameters: @unpack
using SymbolicPlanners, Plinf
using SymbolicPlanners: PathNode, simplify_goal, LinkedNodeRef, @auto_hash, @auto_equals, reconstruct
using PDDL

mutable struct TwoStagePlanner <: Planner
    max_nodes::Int
    max_time::Float64
    save_search::Bool
    save_search_order::Bool
    verbose::Bool
    callback::Union{Nothing, Function}
    alternative_planner::Union{Nothing, Planner}

    TwoStagePlanner(;max_nodes::Int=typemax(Int), max_time::Float64=Inf,
                  save_search::Bool=false, alternative_planner=AStarPlanner(GoalManhattan(), save_search=true), save_search_order::Bool=save_search,
                  verbose::Bool=false, callback::Union{Nothing, Function}=verbose ? LoggerCallback() : nothing) =
        new(max_nodes, max_time, save_search, save_search_order, verbose, callback, alternative_planner)
end

@auto_hash TwoStagePlanner
@auto_equals TwoStagePlanner

function (planner::TwoStagePlanner)(domain::Domain, state::State, goal)
    solve(planner, domain, state, Specification(goal))
end

function (planner::TwoStagePlanner)(domain::Domain, state::State, goal::MinStepsGoal)
    solve(planner, domain, state, Specification(goal.terms))
end

function solve(planner::TwoStagePlanner, domain::Domain, state::State, spec::Specification)

    # Extract the item from the first 'has' goal in the specification
    @unpack save_search = planner

    item = nothing
    for goal in spec.terms
        if goal isa Term && goal.name == :has && length(goal.args) == 1
            item = goal.args[1]
            break
        end
    end
    if item === nothing
        error("No 'has' goal found in specification")
    end

    item_visible = PDDL.satisfy(domain, state, pddl"(visible ${item})")

    if !item_visible
        # Create the subgoal
        subgoal = Compound(:visible, Term[item])
        spec_intermediate = Specification(subgoal)
        spec_final = spec
        spec_intermediate = simplify_goal(spec_intermediate, domain, state)

        node_id = hash(state)
        node = PathNode(node_id, state, 0.0, LinkedNodeRef(node_id))

        search_tree = Dict(node_id => node)
        queue = [node_id]
        search_order = UInt[]
        sol1 = PathSearchSolution(:in_progress, Term[], Vector{typeof(state)}(),
                                0, search_tree, queue, search_order)
        sol1 = search!(sol1, planner, domain, spec_intermediate, state)

        if sol1.status != :failure && !isnothing(planner.alternative_planner)
            sol2 = planner.alternative_planner(domain, sol1.trajectory[end], spec_final)
        end

        sol = stack_solutions(sol1, sol2)

    else
        sol = planner.alternative_planner(domain, state, spec)
    end

    if save_search
        return sol
    elseif sol.status == :failure
        return NullSolution(sol.status)
    else
        return PathSearchSolution(sol.status, sol.plan, sol.trajectory)
    end
end

# Add the Base.copy method
function Base.copy(planner::TwoStagePlanner)
    TwoStagePlanner(
        max_nodes = planner.max_nodes,
        max_time = planner.max_time,
        save_search = planner.save_search,
        save_search_order = planner.save_search_order,
        verbose = planner.verbose,
        callback = planner.callback,
        alternative_planner = isnothing(planner.alternative_planner) ? nothing : copy(planner.alternative_planner)
    )
end

function search!(sol::PathSearchSolution, planner::TwoStagePlanner,
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
    planner::TwoStagePlanner, node::PathNode{S},
    search_tree::Dict{UInt,PathNode{S}}, queue::Vector{UInt},
    domain::Domain, spec::Specification, visited::Set{Tuple{Int64,Int64}}
) where {S <: State}
    state = node.state

    available_actions = [act for act in PDDL.available(domain, state)]
    if isempty(available_actions)
        return  # Return if there is no direction to proceed
    end

    # Remove visited states from available actions as much as possible
    unvisited_and_available_actions = []
    for act in available_actions
        next_state = PDDL.transition(domain, state, act, check=false)
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
    next_state = PDDL.transition(domain, state, chosen_action, check=false)
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

function stack_solutions(sol1::PathSearchSolution, sol2::PathSearchSolution)
    # Combine plans
    combined_plan = vcat(sol1.plan, sol2.plan)
    
    # Combine trajectories if they exist
    combined_trajectory = if sol1.trajectory !== nothing && sol2.trajectory !== nothing
        vcat(sol1.trajectory, sol2.trajectory[2:end])  # Avoid duplicating the connecting state
    else
        nothing
    end
    
    # Sum up expanded nodes
    total_expanded = sol1.expanded + sol2.expanded
    
    # Merge search trees if they exist
    combined_search_tree = if sol1.search_tree !== nothing && sol2.search_tree !== nothing
        merge(sol1.search_tree, sol2.search_tree)
    else
        nothing
    end
    
    # Combine search frontiers
    combined_frontier = if typeof(sol1.search_frontier) == typeof(sol2.search_frontier)
        vcat(sol1.search_frontier, sol2.search_frontier)
    else
        # If types are different, convert to Vector{Any}
        vcat(collect(sol1.search_frontier), collect(sol2.search_frontier))
    end
    
    # Combine search orders
    combined_search_order = vcat(sol1.search_order, sol2.search_order)
    
    # Determine the overall status
    combined_status = if sol1.status == :success && sol2.status == :success
        :success
    elseif sol1.status == :failure || sol2.status == :failure
        :failure
    else
        :partial
    end
    
    # Create and return the new combined solution
    return PathSearchSolution{eltype(sol1.trajectory), typeof(combined_frontier)}(
        combined_status,
        combined_plan,
        combined_trajectory,
        total_expanded,
        combined_search_tree,
        combined_frontier,
        combined_search_order
    )
end