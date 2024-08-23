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

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

# ======= Config ======= #

share_beliefs::Bool = false
ess_thresh::Float64 = 0.3
num_particles::Int = 100
ground_truth_rewards::Dict{String, Int} = Dict("red" => 5, "blue" => -1, "yellow" => 1, "green" => -1)
gridworld_only = false
T = 100

# ====================== #

global state, t, remaining_items
PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem_name::String = "medium"
problem::Problem = load_problem(joinpath(@__DIR__, "problems", problem_name*".pddl"))
initial_state = initstate(domain, problem)
spec = Specification(problem)
domain, initial_state = PDDL.compiled(domain, initial_state)
items = [obj.name for obj in PDDL.get_objects(domain, initial_state, :gem)]
agents = Symbol[obj.name for obj in PDDL.get_objects(domain, initial_state, :agent)]
problem_goal = PDDL.get_goal(problem)

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

belief_mode = share_beliefs ? "shared" : "individual"
output_folder = joinpath(@__DIR__, "output", problem_name * "_" * belief_mode)
mkpath(output_folder)

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
state = initial_state

# Initialize particle filter states and gem utilities based on share_beliefs
if share_beliefs
    global pf_states::Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}} = nothing
    global gem_utilities = Dict(gem => Dict("utility" => 0., "value" => 0.) for gem in ["red", "blue", "yellow", "green"])
else
    global pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    global gem_utilities = Dict(agent => Dict(gem => Dict("utility" => 0., "value" => 0.) for gem in ["red", "blue", "yellow", "green"]) for agent in agents)
end

num_gems_picked_up::Dict{Symbol, Int} = Dict(agent => 0 for agent in agents)
global total_gems_picked_up::Int = 0
global combined_score::Int = 0

t::Int = 1
while !isempty(remaining_items) && t <= T
    global state, t, remaining_items, num_gems_picked_up, gem_utilities, combined_score, total_gems_picked_up, pf_states

    for (i, agent) in enumerate(agents)
        visible_gems = [item for item in remaining_items if PDDL.satisfy(domain, state, PDDL.parse_pddl("(visible $agent $item)"))]
        if isempty(visible_gems) # If no gems visible, search randomly
            goal_str = "(or " * join(["(has $agent $item)" for item in remaining_items], " ") * ")"
        else # gem(s) are visible
            current_utilities = share_beliefs ? gem_utilities : gem_utilities[agent]
            positive_utility_gems = [ # Filter for gems with non-negative utility
                gem for gem in visible_gems 
                if get(current_utilities, parse_gem(String(gem)), Dict("utility" => 0.0))["utility"] >= 0
            ]
            if !isempty(positive_utility_gems) # Choose the gem with the highest utility
                best_gem = argmax(
                    gem -> current_utilities[parse_gem(String(gem))]["utility"], 
                    positive_utility_gems
                )
                goal_str = "(has $agent $best_gem)"
                PDDL.set_fluent!(state, true, pddl"(is-goal-item $best_gem)")
            else # If all visible gems have negative utility, search randomly
                other_gems = setdiff(remaining_items, visible_gems)
                goal_str = "(or " * join(["(has $agent $item)" for item in other_gems], " ") * ")"
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
            gem = parse_gem(String(item))
            num_gems_picked_up[agent] += 1
            total_gems_picked_up += 1
            reward = ground_truth_rewards[gem]
            combined_score += reward
            println("       $agent received $reward score.")
            println("       Combined score is now $combined_score.")

            observation = Gen.choicemap()
            observation[(1 => :gem_pickup)] = true
            observation[(1 => :gem)] = gem
            observation[(:reward => Symbol(gem))] = reward
            tr, _ = Gen.generate(utterance_model, (1,), observation)
            utterance = Gen.get_retval(tr)[1]
            println("       $agent communicated: $utterance.")
            
            alt_observation::Gen.ChoiceMap = Gen.choicemap()
            gem_count::Int = share_beliefs ? total_gems_picked_up : num_gems_picked_up[agent]
            alt_observation[gem_count => :utterance => :output] = utterance
            alt_observation[gem_count => :gem_pickup] = true
            alt_observation[gem_count => :gem] = gem

            # Update beliefs
            if share_beliefs
                if pf_states === nothing
                    pf_states = pf_initialize(utterance_model, (gem_count,), alt_observation, num_particles)
                else
                    if effective_sample_size(pf_states) < ess_thresh * num_particles
                        pf_resample!(pf_states, :stratified)
                        rejuv_sel = select()
                        pf_rejuvenate!(pf_states, mh, (rejuv_sel,))
                    end
                    pf_update!(pf_states, (total_gems_picked_up,), (UnknownChange(),), alt_observation)
                end
                current_pf_state = pf_states
            else
                if pf_states[agent] === nothing
                    pf_states[agent] = pf_initialize(utterance_model, (gem_count,), alt_observation, num_particles)
                else
                    if effective_sample_size(pf_states[agent]) < ess_thresh * num_particles
                        pf_resample!(pf_states[agent], :stratified)
                        rejuv_sel = select()
                        pf_rejuvenate!(pf_states[agent], mh, (rejuv_sel,))
                    end
                    pf_update!(pf_states[agent], (gem_count,), (UnknownChange(),), alt_observation)
                end
                current_pf_state = pf_states[agent]
            end

            # Calculate and update gem utilities
            top_rewards = get_top_weighted_rewards(current_pf_state, 10)
            gem_certainty = quantify_gem_certainty(top_rewards)
            new_utilities = calculate_gem_utility(gem_certainty, risk_aversion = 0)
            
            # Update utilities for all agents if sharing beliefs, otherwise just for the current agent
            if share_beliefs
                gem_utilities = new_utilities
            else
                gem_utilities[agent] = new_utilities
            end
            
            println("       Estimated Rewards:")
            for (gem, info) in new_utilities
                println("              $gem: value = $(info["value"]), certainty = $(round(info["certainty"], digits=2)), utility = $(round(info["utility"], digits=2))")
            end
        end
    end
    t += 1
end

println("\nFinal Score: $combined_score\n")

# Animate the plan
anim = anim_plan(renderer, domain, initial_state, all_actions; format="gif", framerate=2)
save(output_folder*"/plan.mp4", anim)
