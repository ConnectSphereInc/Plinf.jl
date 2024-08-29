using PDDL, SymbolicPlanners
using SymbolicPlanners: ManhattanHeuristic

struct VisionGemHeuristic <: Heuristic
    agent::Symbol
    gem_rewards::Dict{Symbol, Float64}
    visited_states::Dict{Tuple{Int,Int}, Int}
end

function VisionGemHeuristic(spec::MultiGoalReward, agent::Symbol)
    gem_rewards = Dict{Symbol, Float64}()
    for (goal, reward) in zip(spec.goals, spec.rewards)
        if goal isa Compound && goal.name == :has && goal.args[1] == Const(agent)
            gem = Symbol(goal.args[2].name)
            gem_rewards[gem] = reward
        end
    end
    return VisionGemHeuristic(agent, gem_rewards, Dict{Tuple{Int,Int}, Int}())
end

function SymbolicPlanners.compute(h::VisionGemHeuristic, domain::Domain, state::State, spec::Specification)
    agent = h.agent
    agent_pos = get_agent_position(domain, state, agent)
    visible_gems = get_visible_gems(domain, state, agent)
    positive_reward_gems = [(gem, pos) for (gem, pos) in visible_gems if get(h.gem_rewards, gem, 0.0) > 0]

    value = 0.0
    if !isempty(positive_reward_gems) # Exist gems with positive reward
        for (gem, gem_pos) in positive_reward_gems
            distance = astar_path_length(domain, state, h.agent, agent_pos, gem_pos)
            gem_value = get(h.gem_rewards, gem, 100.0)
            value -= gem_value / (distance + 1)  # Avoid division by zero
        end
    else # Apply backtracking penalty only when no gems are visible to encourage exploration
        visit_count = get(h.visited_states, agent_pos, 0)
        backtrack_penalty = 0.1 * visit_count
        value += backtrack_penalty
    end
    
    h.visited_states[agent_pos] = get(h.visited_states, agent_pos, 0) + 1
    
    return value
end

function get_agent_position(domain::Domain, state::State, agent::Symbol)
    x = state[Compound(:xloc, [Const(agent)])]
    y = state[Compound(:yloc, [Const(agent)])]
    return (x, y)
end

function get_visible_gems(domain::Domain, state::State, agent::Symbol)
    visible_gems = Tuple{Symbol, Tuple{Int64, Int64}}[]
    for obj in PDDL.get_objects(domain, state)
        if PDDL.satisfy(domain, state, PDDL.parse_pddl("(visible $agent $(obj.name))"))
            x = state[Compound(:xloc, [obj])]
            y = state[Compound(:yloc, [obj])]
            push!(visible_gems, (Symbol(obj.name), (x, y)))
        end
    end
    return visible_gems
end

function astar_path_length(domain::Domain, state::State, agent::Symbol, start_pos::Tuple{Int,Int}, gem_pos::Tuple{Int,Int})
    goal = PDDL.parse_pddl("(and (= (xloc $agent) $(gem_pos[1])) (= (yloc $agent) $(gem_pos[2])))")
    planner = AStarPlanner(GoalCountHeuristic())
    spec = MinStepsGoal(goal)
    solution = planner(domain, state, spec)
    return length(solution)
end