using PDDL, Printf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DotEnv
using Random
import SymbolicPlanners: compute, get_goal_terms
include("planner.jl")
include("utterances.jl")
include("utils.jl")

global state, t, remaining_items

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "simple.pddl"))
initial_state = initstate(domain, problem)
spec = Specification(problem)
domain, initial_state = PDDL.compiled(domain, initial_state)
items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
problem_goal = PDDL.get_goal(problem)

gridworld_only = false
renderer = PDDLViz.GridworldRenderer(
    resolution = (600,1100),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> MultiGraphic(
            RobotGraphic(color = :slategray),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
        :red => (d, s, o) -> MultiGraphic(
            GemGraphic(color = :red),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
        :yellow => (d, s, o) -> MultiGraphic(
            GemGraphic(color = :yellow),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
        :blue => (d, s, o) -> MultiGraphic(
            GemGraphic(color = :blue),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
        :green => (d, s, o) -> MultiGraphic(
            GemGraphic(color = :green),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
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

# Make the output folder
output_folder = joinpath(@__DIR__, "output", "simple")
mkpath(output_folder)

# Save the canvas to a file
println("Saving initial state to file")
save(output_folder*"/initial_state.png", canvas)

remaining_items = copy(items)
all_actions = []
planners = [
    SingleStepVisionPlanner(
        agent = agent,
        alternative_planner = AStarPlanner(GoalManhattan(agent), save_search=true),
        save_search = true
    ) 
    for agent in agents
]

T=100
t=1

utterances = [
    "Found a blue gem for 3!",
    "This yellow gem gave me +1 reward.",
    "Picked up a red gem, +5!",
    "Found a green gem for -1!",
]

pf_state = particle_filter(utterances, 100, infer_gem=false)

top_rewards = get_top_weighted_rewards(pf_state, 10)
println("Top 5 most likely reward estimates:")
for (i, (rewards, weight)) in enumerate(top_rewards)
    println("$i. Rewards: $rewards, Weight: $(round(weight, digits=3))")
end

gem_certainty = quantify_gem_certainty(top_rewards)
println("Gem certainty:")
println(gem_certainty)

gem_utilities = calculate_gem_utility(gem_certainty, risk_aversion = 0)
println("Gem Utilities:")
for (gem, info) in gem_utilities
    println("$gem: value = $(info["value"]), certainty = $(round(info["certainty"], digits=2)), utility = $(round(info["utility"], digits=2))")
end

function gem_to_name(gem)
    if gem == "red"
        return "gem1"
    elseif gem == "blue"
        return "gem2"
    elseif gem == "yellow"
        return "gem3"
    elseif gem == "green"
        return "gem4"
    end
end

best_gem = gem_to_name(argmax(gem -> gem_utilities[gem]["utility"], keys(gem_utilities)))
goals = [PDDL.parse_pddl("(has robot1 $best_gem)")]

state = initial_state

while t <= T
    global state, t, remaining_items
    for (i, agent) in enumerate(agents)
        goal = goals[i]
        solution = planners[i](domain, state, goal)
        action = first(solution.plan)
        state = solution.trajectory[end]
        push!(all_actions, action)
        if action.name == :communicate
            # todo: implement communication code
        elseif action.name == :pickup
            remaining_items = filter(item -> item != best_gem, remaining_items)
        end
    end
    problem_solved = PDDL.satisfy(domain, state, goals[1])
    if problem_solved
        break
    end
    t += 1
end

# Animate the plan
anim = anim_plan(renderer, domain, initial_state, all_actions; format="gif", framerate=2)
save(output_folder*"/plan.mp4", anim)
