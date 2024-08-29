using PDDL, PlanningDomains, SymbolicPlanners
using PDDLViz, GLMakie
using Random
using SymbolicPlanners: get_value

include("heuristics.jl")

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem_name::String = "medium"
problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name*".pddl"))
initial_state = initstate(domain, problem)
domain, initial_state = PDDL.compiled(domain, initial_state)
items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
problem_goal = PDDL.get_goal(problem)

gridworld_only = false
renderer = PDDLViz.GridworldRenderer(
    resolution = (600,1100),
    has_agent = false,
    obj_renderers = Dict{Symbol, Function}(
        key => (d, s, o) -> MultiGraphic(
            (key == :agent ? RobotGraphic : GemGraphic)(color = (key == :agent ? :slategray : key)),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color = :black, font = :bold
            )
        )
        for key in [:agent, :red, :yellow, :blue, :green]
    ),
    show_inventory = !gridworld_only,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(agent), o])] for agent in agents
    ],
    inventory_types = [:item for agent in agents],
    inventory_labels = ["$agent Inventory" for agent in agents],
    show_vision = !gridworld_only,
    vision_fns = [
        (d, s, o) -> s[Compound(:visible, [Const(agent), o])] for agent in agents
    ],
    vision_types = [:item for agent in agents],
    vision_labels = ["$agent Vision" for agent in agents],
)
canvas = renderer(domain, initial_state)

output_folder = joinpath(@__DIR__, "output", problem_name)
mkpath(output_folder)

println("Saving initial state to file")
save(output_folder*"/initial_state.png", canvas)

# Construct MultiGoalReward specification
goals = Term[]
rewards = Float64[]

ground_truth_rewards::Dict{String, Int} = Dict("red" => 2, "blue" => -5, "yellow" => 1, "green" => -1)

for gem in items
    gem_obj = Const(gem)
    for agent in agents
        push!(goals, pddl"(has $agent $gem_obj)")
        color = split(string(gem), "_")[1]
        reward = Float64(get(ground_truth_rewards, color, 0))
        push!(rewards, reward)
    end
    # break
end

discount = 0.95
spec = MultiGoalReward(goals, rewards, discount)

global state = initial_state
heuristics = [VisionGemHeuristic(spec, agent) for agent in agents]
planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]

function best_action(sol::TabularVPolicy, state::State)
    best_val = -Inf
    best_acts = []
    for act in available(sol.domain, state)
        val = get_value(sol, state, act)
        if val > best_val
            best_val = val
            best_acts = [act]
        elseif val == best_val
            push!(best_acts, act)
        end
    end
    return isempty(best_acts) ? missing : rand(best_acts)
end

actions = []
max_steps = 100
for t in 1:max_steps
    for (agent, planner) in zip(agents, planners)
        global state
        println("T: $t, Agent: $agent")
        println("Agent position: $(get_agent_position(domain, state, agent))")
        println("available actions")
        for act in available(domain, state)
            println("   $act")
        end
        sol::ReusableTreePolicy = planner(domain, state, spec)

        # Use the value_policy from ReusableTreePolicy
        action = best_action(sol.value_policy, state)

        println("Selected action: $action")
        state = transition(domain, state, action; check=true)
        push!(actions, action)
    end
end

# Animate the plan
anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
save(output_folder*"/plan.mp4", anim)