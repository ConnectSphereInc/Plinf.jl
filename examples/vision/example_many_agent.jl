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
domain = load_domain(joinpath(@__DIR__, "domains/domain-local-interaction.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "many-agent.pddl"))
state = initstate(domain, problem)
spec = Specification(problem)
domain, state = PDDL.compiled(domain, state)
items = [obj.name for obj in PDDL.get_objects(domain, state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, state, :agent)]

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
output_folder = "examples/vision/output/many-agent/"
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

following = Dict{Symbol, Union{Nothing, Symbol}}(agent => nothing for agent in agents)

current_state = state
all_actions = []

t_max = 100  # Set a maximum number of timesteps
t = 1

while !isempty(remaining_items) && t <= t_max
    global remaining_items, current_state, t

    for (i, agent) in enumerate(agents)
        if isnothing(following[agent])
            # If not following anyone, use the original goal
            goal_str = "(or " * join(["(has $agent $item)" for item in remaining_items], " ") * ")"
            goal = PDDL.parse_pddl(goal_str)
            solution = planners[i](domain, current_state, goal)
        else
            # If following another agent, set the goal to their current position
            followed_agent = following[agent]
            followed_pos = get_agent_pos(current_state, followed_agent)
            agent_pos = get_agent_pos(current_state, agent)
            
            goal = PDDL.Compound(:and, [
                PDDL.Compound(:(==), [PDDL.Compound(:xloc, [PDDL.Const(agent)]), PDDL.Const(followed_pos[1])]),
                PDDL.Compound(:(==), [PDDL.Compound(:yloc, [PDDL.Const(agent)]), PDDL.Const(followed_pos[2])])
            ])
            solution = planners[i](domain, current_state, goal)
            
            # If A* fails to find a path, stop following
            if solution.status != :success
                following[agent] = nothing
                println("       $agent stops following $(followed_agent) due to unreachable position.")
                continue
            end
        end
        
        if solution.status == :success && !isempty(solution.plan)
            action = first(solution.plan)
            current_state = solution.trajectory[end]
            push!(all_actions, action)

            if action.name == :pickup
                item = action.args[2].name
                println("Step $t:")
                println("       $agent picked up $item.")
                remaining_items = filter(x -> x != item, remaining_items)
                following[agent] = nothing  # Stop following after picking up an item
            elseif action.name == :communicate
                communicator = agent
                listener = action.args[2].name
                println("Step $t:")
                println("       $communicator sees $listener and makes an utterance.")

                # Check if the agent actually sees a gem
                items_visible = !isempty([item.name for item in PDDL.get_objects(domain, current_state, :item) if PDDL.satisfy(domain, current_state, pddl"(visible $communicator $item)")])

                # Generate an utterance based on item visibility
                (tr, _) = generate(utterance_model, (), Gen.choicemap(:gem_visible => items_visible))
                utterance = get_retval(tr)
                println("       $communicator utters: $utterance")

                # Listening agent infers if the agent can see a gem based on their utterance
                traces, weights = Gen.importance_sampling(utterance_model, (), Gen.choicemap(:output => utterance), 2)
                believes_utterer_sees_gem, _ = get_most_likely(traces, weights)

                println("       $listener thinks $communicator $(believes_utterer_sees_gem ? "can" : "can not") see a gem.")

                # If the listener believes the utterer can see a gem, start following
                if believes_utterer_sees_gem
                    following[Symbol(listener)] = Symbol(communicator)
                    println("       $listener is following $communicator.")
                else
                    following[Symbol(listener)] = nothing
                    println("       $listener is not following $communicator.")
                end
            else
                # println("Step $t:")
                # println("       $agent moves: $action")
            end
        elseif solution.status == :at_target
            println("Step $t:")
            println("       $agent is already at the target position.")
        else
            println("Step $t:")
            println("       $agent couldn't find a valid action.")
            # If the agent can't find a valid action, stop following
            following[agent] = nothing
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
anim = anim_plan(renderer, domain, state, all_actions; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan.mp4", anim)
