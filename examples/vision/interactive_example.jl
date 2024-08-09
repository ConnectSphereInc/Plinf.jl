using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DotEnv
using Random
include("utils.jl")
include("one_step_planner.jl")

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

global remaining_items
global current_state
global t

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "problem-5.pddl"))
state = initstate(domain, problem)
spec = Specification(problem)
domain, state = PDDL.compiled(domain, state)
items = [obj.name for obj in PDDL.get_objects(domain, state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, state, :agent)]

# Construct gridworld renderer
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
        :gem => (d, s, o) -> MultiGraphic(
            GemGraphic(color = :yellow),
            TextGraphic(
                string(o.name)[end:end], 0.3, 0.2, 0.5,
                color=:black, font=:bold
            )
        ),
    ),
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(agent), o])] for agent in agents
    ],
    inventory_types = [:item for agent in agents],
    inventory_labels = ["$agent Inventory" for agent in agents],
    show_vision = true,
    vision_fns = [

        (d, s, o) -> s[Compound(:visible, [Const(agent), o])] for agent in agents
    ],
    vision_types = [:object for agent in agents],
    vision_labels = ["$agent Vision" for agent in agents],
    obj_type_z_order = [:gem, :agent],
)
canvas = renderer(domain, state)

# Make the output folder
output_folder = "examples/vision/output"
if !isdir(output_folder)
    mkdir(output_folder)
end

# Save the canvas to a file
println("Saving initial state to file")
save(output_folder*"/initial_state.png", canvas)


remaining_items = copy(items)
planners = [
    SingleStepVisionPlanner(
        agent = agent,
        alternative_planner = AStarPlanner(GoalManhattan(agent), save_search=true),
        save_search = true
    ) 
    for agent in agents
]

current_state = state
full_plan = []

t_max = 100  # Set a maximum number of timesteps
t = 1

while !isempty(remaining_items) && t <= t_max
    global remaining_items, current_state, t

    for (i, agent) in enumerate(agents)
        goal_str = "(or " * join(["(has $agent $item)" for item in remaining_items], " ") * ")"
        goal = PDDL.parse_pddl(goal_str)
        
        solution = planners[i](domain, current_state, goal)
        
        if solution.status == :success && !isempty(solution.plan)
            action = first(solution.plan)
            current_state = solution.trajectory[end]
            push!(full_plan, action)

            if action.name == :pickup
                item = action.args[2].name
                println("Step $t: $agent picked up $item.")
                remaining_items = filter(x -> x != item, remaining_items)
            elseif action.name == :communicate
                println("Step $t: $agent communicated with $(action.args[2].name).")
                utterance = utterance_model(agent, domain, current_state)
                println("       $agent: $utterance")
            else
                # println("Step $t: $agent performed action: $action")
            end
        else
            println("Step $t: Agent $agent couldn't find a valid move.")
        end
    end
    t += 1
end

if isempty(remaining_items)
    println("All items collected after $t steps.")
else
    println("Reached maximum number of steps ($t_max) without collecting all items. Remaining items: $remaining_items")
end

# Animate the plan
anim = anim_plan(renderer, domain, state, full_plan; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan.mp4", anim)