using PDDL, Printf
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using DotEnv
using Random
import SymbolicPlanners: compute, get_goal_terms
import GenParticleFilters: pf_initialize, pf_update!, pf_resample!, pf_rejuvenate!, get_norm_weights, get_traces, effective_sample_size
include("planner.jl")
include("utterances.jl")
include("utils.jl")

global state, t, remaining_items

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "medium.pddl"))
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

state = initial_state

ground_truth_rewards::Dict{String, Int} = Dict("red" => 5, "blue" => -1, "yellow" => 1, "green" => -1)

global score = 0
global num_gems_picked_up = 0
global ess_thresh = 0.3
global gem_utilities = Dict(gem => Dict("utility" => 0, "value" => 0) for gem in ["red", "blue", "yellow", "green"])
num_particles = 100

while !isempty(remaining_items) && t <= T
    global state, t, remaining_items, score, num_gems_picked_up, ess_thresh, gem_utilities
    for (i, agent) in enumerate(agents)
        visible_gems = [item for item in remaining_items if PDDL.satisfy(domain, state, PDDL.parse_pddl("(visible $agent $item)"))]
        if isempty(visible_gems)
            # If no gems visible, search randomly
            goal_str = "(or " * join(["(has $agent $item)" for item in remaining_items], " ") * ")"
        else # gem(s) are visible
            if num_gems_picked_up == 0 # no gems picked up yet
                best_gem = visible_gems[1]
                goal_str = "(has $agent $best_gem)"
                PDDL.set_fluent!(state, true, pddl"(is-goal-item $best_gem)")
            else # gem(s) previously picked up
                # Filter for gems with non-negative utility
                positive_utility_gems = [
                    gem for gem in visible_gems 
                    if get(gem_utilities, gem_from_utterance(String(gem)), Dict("value" => 0))["value"] >= 0
                ]
                
                if !isempty(positive_utility_gems)
                    # Choose the gem with the highest utility
                    best_gem = argmax(
                        gem -> gem_utilities[gem_from_utterance(String(gem))]["utility"], 
                        positive_utility_gems
                    )
                    goal_str = "(has $agent $best_gem)"
                    PDDL.set_fluent!(state, true, pddl"(is-goal-item $best_gem)")
                else
                    # If all visible gems have negative utility, search randomly
                    other_gems = setdiff(remaining_items, visible_gems)
                    goal_str = "(or " * join(["(has $agent $item)" for item in other_gems], " ") * ")"
                end
            end
        end

        goal = PDDL.parse_pddl(goal_str)
        solution = planners[i](domain, state, goal)

        action = first(solution.plan)
        state = solution.trajectory[end]
        push!(all_actions, action)

        if action.name == :pickup
            item = action.args[2].name
            println("Step $t:")
            println("       $agent picked up $item.")
            remaining_items = filter(x -> x != item, remaining_items)

            # Create an utterance
            gem = gem_from_utterance(String(item))
            num_gems_picked_up += 1
            reward = ground_truth_rewards[gem]
            println("       $agent recieved $reward score.")

            observation::Gen.ChoiceMap = Gen.choicemap()
            observation[(1 => :gem_pickup)] = true
            observation[(1 => :gem)] = gem
            observation[(:reward => Symbol(gem))] = reward
            tr, _ = Gen.generate(utterance_model, (1,), observation)
            utterance = Gen.get_retval(tr)[1]
            println("       $agent communicated: $utterance.")
            
            # Broadcast and do inference (agents can share beliefs since they all hear everything)
            alt_observation = Gen.choicemap()
            alt_observation[num_gems_picked_up => :utterance => :output] = utterance
            alt_observation[num_gems_picked_up => :gem_pickup] = true
            alt_observation[num_gems_picked_up => :gem] = gem # todo: make this optional

            if num_gems_picked_up == 1
                pf_state = pf_initialize(utterance_model, (1,), alt_observation, num_particles)
            else
                if effective_sample_size(pf_state) < ess_thresh * num_particles
                    pf_resample!(pf_state, :stratified)
                    rejuv_sel = select()
                    pf_rejuvenate!(pf_state, mh, (rejuv_sel,))
                end
                pf_update!(pf_state, (num_gems_picked_up,), (UnknownChange(),), alt_observation)
            end

            top_rewards = get_top_weighted_rewards(pf_state, 10)
            gem_certainty = quantify_gem_certainty(top_rewards)
            gem_utilities = calculate_gem_utility(gem_certainty, risk_aversion = 0)
            println("       Estimated Rewards:")
            for (gem, info) in gem_utilities
                println("              $gem: value = $(info["value"]), certainty = $(round(info["certainty"], digits=2)), utility = $(round(info["utility"], digits=2))")
            end
        end
    end
    t += 1
end

# Animate the plan
anim = anim_plan(renderer, domain, initial_state, all_actions; format="gif", framerate=2)
save(output_folder*"/plan.mp4", anim)
