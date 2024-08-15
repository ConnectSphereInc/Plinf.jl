"""
Two Agent Example

In this example, there are two agents. 
Each agent know their own coordinates in the gridworld.
Additionally, they are able to communicate with the other agent.

Each agent is after a gem with a specific color.
They can only pick up their own gem.
"""

using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DotEnv
using Random
include("utils.jl")
include("one_step_planner.jl")

global remaining_items
global current_state
global t

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domains/domain-global-interaction.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "two-agent.pddl"))
state = initstate(domain, problem)
spec = Specification(problem)
domain, state = PDDL.compiled(domain, state)
items = [obj.name for obj in PDDL.get_objects(domain, state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, state, :agent)]
problem_goal = PDDL.get_goal(problem)

gridworld_only = false
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
    vision_types = [:object for agent in agents],
    vision_labels = ["$agent Vision" for agent in agents],
    obj_type_z_order = [:gem, :agent],
)
canvas = renderer(domain, state)

# Make the output folder
output_folder = "examples/vision/output/two-agent/"
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
goals = ["(has $agent gem$i)" for (i, agent) in enumerate(agents)]
current_state = state
all_actions = []

t_max = 100  # Set a maximum number of timesteps
t = 1

while !isempty(remaining_items) && t <= t_max
    global remaining_items, current_state, t

    # iterate over over each agent
    for (i, agent) in enumerate(agents)
        goal = PDDL.parse_pddl(goals[i])
        solution = planners[i](domain, current_state, goal)

        if solution.status == :success
            action = first(solution.plan)
            current_state = solution.trajectory[end]
            push!(all_actions, action)

            if action.name == :communicate
                caller = agent
                callee = action.args[2].name
                found_gem = action.args[3].name
                gem_x_pos, gem_y_pos = get_agent_pos(current_state, found_gem)
                println("Step $t:")
                println("       $caller found $found_gem at ($gem_x_pos, $gem_y_pos) and tells $callee.")

                # Generate an utterance based on item visibility
                constraints = Gen.choicemap()
                constraints[:partners_gem_visible] = true
                constraints[:gem_x_pos] = gem_x_pos
                constraints[:gem_y_pos] = gem_y_pos

                (tr, _) = generate(utterance_model_global, (), constraints)
                utterance = get_retval(tr)
                println("       $caller utters: $utterance")

                # Listening agent infers if the agent can see a gem based on their utterance
                traces, weights = enum_inference(utterance, 10, 10)  # Adjust grid size if needed
                (gem_x_pos_belief, gem_y_pos_belief), _ = get_most_likely_global(traces, weights)
                println("       $callee thinks the gem is at position ($gem_x_pos_belief, $gem_y_pos_belief).")
                
                # Update the planner's believed gem position for the listening agent
                callee_index = findfirst(a -> a == Symbol(callee), agents)
                planners[callee_index].believed_gem_position = (gem_x_pos_belief, gem_y_pos_belief)
            elseif action.name == :pickup
                # If a pickup action is successful, reset the believed gem position
                planners[i].believed_gem_position = nothing
            end
        else
            # If the planner fails (e.g., reaches believed position but no gem), reset the belief
            planners[i].believed_gem_position = nothing
        end
    end

    problem_solved = PDDL.satisfy(domain, current_state, problem_goal)
    if problem_solved
        break
    end
    t += 1
end

# Animate the plan
anim = anim_plan(renderer, domain, state, all_actions; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan.mp4", anim)