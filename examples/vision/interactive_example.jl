using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DotEnv
using Random
include("utils.jl")
include("random_planner.jl")

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

global remaining_items
global current_state
global t

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "problem-2.pddl"))
state = initstate(domain, problem)
spec = Specification(problem)
domain, state = PDDL.compiled(domain, state)

# Construct gridworld renderer
renderer = PDDLViz.GridworldRenderer(
    resolution = (600,1100),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> RobotGraphic(),
        :gem => (d, s, o) -> GemGraphic(color = :yellow)
    ),
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:robot1), o])]
        (d, s, o) -> s[Compound(:has, [Const(:robot2), o])]

    ],
    inventory_types = [:item, :item],
    inventory_labels = ["Robot1 Inventory", "Robot2 Inventory"],
    show_vision = true,
    vision_fns = [
        (d, s, o) -> s[Compound(:visible, [Const(:robot1), o])],
        (d, s, o) -> s[Compound(:visible, [Const(:robot2), o])],
    ],
    vision_types = [:object, :object],
    vision_labels = ["Robot1 Vision", "Robot2 Vision"],
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

items = [obj.name for obj in PDDL.get_objects(domain, state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, state, :agent)]
remaining_items = copy(items)
planners = [
    TwoStagePlanner(
        agent = agent,
        alternative_planner = AStarPlanner(GoalManhattan(agent), save_search=true),
        save_search = true
    ) 
    for agent in agents
]

current_state = state
full_plan = []
agent_plans = [create_plan(planners[i], domain, current_state, remaining_items, agent) for (i, agent) in enumerate(agents)]

t_max = 100  # Set a maximum number of timesteps
t = 1

using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DotEnv
using Random
include("utils.jl")
include("random_planner.jl")

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

global remaining_items
global current_state
global t

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "problem-2.pddl"))
state = initstate(domain, problem)
spec = Specification(problem)
domain, state = PDDL.compiled(domain, state)

# Construct gridworld renderer
renderer = PDDLViz.GridworldRenderer(
    resolution = (600,1100),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> RobotGraphic(),
        :gem => (d, s, o) -> GemGraphic(color = :yellow)
    ),
    show_inventory = true,
    inventory_fns = [
        (d, s, o) -> s[Compound(:has, [Const(:robot1), o])]
        (d, s, o) -> s[Compound(:has, [Const(:robot2), o])]

    ],
    inventory_types = [:item, :item],
    inventory_labels = ["Robot1 Inventory", "Robot2 Inventory"],
    show_vision = true,
    vision_fns = [
        (d, s, o) -> s[Compound(:visible, [Const(:robot1), o])],
        (d, s, o) -> s[Compound(:visible, [Const(:robot2), o])],
    ],
    vision_types = [:object, :object],
    vision_labels = ["Robot1 Vision", "Robot2 Vision"],
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

items = [obj.name for obj in PDDL.get_objects(domain, state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, state, :agent)]
remaining_items = copy(items)
planners = [
    TwoStagePlanner(
        agent = agent,
        alternative_planner = AStarPlanner(GoalManhattan(agent), save_search=true),
        save_search = true
    ) 
    for agent in agents
]

current_state = state
full_plan = []
agent_plans = [create_plan(planners[i], domain, current_state, remaining_items, agent) for (i, agent) in enumerate(agents)]

t_max = 100  # Set a maximum number of timesteps
t = 1

while !isempty(remaining_items) && t <= t_max
    global remaining_items, current_state, t
    
    for (i, agent) in enumerate(agents)
        if isempty(agent_plans[i])
            # Replan if the agent has no more actions
            agent_plans[i] = create_plan(planners[i], domain, current_state, remaining_items, agent)
        end
        
        if !isempty(agent_plans[i])
            action = first(agent_plans[i])
            
            # Check if the action is a pickup and if the item is still available
            if action.name == :pickup
                item = action.args[2].name
                if !(item in remaining_items)
                    println("Step $t: Agent $agent's target item $item is no longer available. Replanning...")
                    agent_plans[i] = create_plan(planners[i], domain, current_state, remaining_items, agent)
                    continue  # Skip to the next agent
                end
            end
            
            # Apply the action
            action = popfirst!(agent_plans[i])
            if action !== nothing
                current_state = PDDL.transition(domain, current_state, action)
                push!(full_plan, action)
                
                if action.name == :pickup
                    item = action.args[2].name
                    println("Step $t: Agent $agent picked up $item")
                    remaining_items = get_offgrid_items(current_state)
                    
                    # Replan for all agents
                    if !isempty(remaining_items)
                        for (j, other_agent) in enumerate(agents)
                            agent_plans[j] = create_plan(planners[j], domain, current_state, remaining_items, other_agent)
                        end
                    end
                end
            end
        end
    end
    
    t += 1
end

if isempty(remaining_items)
    println("All items collected after $t steps.")
else
    println("Reached maximum number of steps ($t_max) without collecting all items.")
end

num_steps = length(full_plan)

# Corrected anim_plan call
anim = anim_plan(renderer, domain, state, full_plan; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan.mp4", anim)

# Print summary
println("Total steps: $num_steps")
println("Remaining items: $remaining_items")