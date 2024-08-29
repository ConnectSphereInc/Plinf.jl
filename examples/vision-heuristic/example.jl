using PDDL, PlanningDomains, SymbolicPlanners
using Gen, GenParticleFilters
using PDDLViz, GLMakie
using Random
using SymbolicPlanners: get_value
using DotEnv
include("utterances.jl")
include("utils.jl")
include("heuristics.jl")

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

# ======= Config ======= #

share_beliefs::Bool = true
ess_thresh::Float64 = 0.3
num_particles::Int = 100
ground_truth_rewards::Dict{Symbol, Int} = Dict(:red => 5, :blue => -1, :yellow => 3, :green => -2)
gridworld_only = false
T = 100

# ====================== #

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

if share_beliefs # communicate
    global pf_states::Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}} = nothing
    global gem_utilities = Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green])
else # don't communicate
    global pf_states = Dict{Symbol, Union{Nothing, ParticleFilterState{Gen.DynamicDSLTrace}}}(agent => nothing for agent in agents)
    global gem_utilities = Dict(agent => Dict(gem => 5.0 for gem in [:red, :blue, :yellow, :green]) for agent in agents)
end

# todo: fix for shared
heuristics = [VisionGemHeuristic(agent, gem_utilities) for agent in agents]
planners = [RTHS(heuristic, n_iters=0, max_nodes=5) for heuristic in heuristics]

global t = 1
global state = initial_state
global num_gems_picked_up::Dict{Symbol, Int} = Dict(agent => 0 for agent in agents)
global total_gems_picked_up::Int = 0
global combined_score::Int = 0
global remaining_items = copy(items)
global possible_gems::Vector{Symbol} = collect(keys(ground_truth_rewards))
global possible_rewards::Vector{Int} = collect(values(ground_truth_rewards))

actions = []
while !isempty(remaining_items) && t <= T
    global state, t, num_gems_picked_up, remaining_items, gem_utilities, combined_score, total_gems_picked_up, pf_states, possible_gems, possible_rewards
    for (i, agent) in enumerate(agents)

        goals = Term[]
        rewards = Float64[]        
        current_beliefs = share_beliefs ? gem_utilities : gem_utilities[agent]
        for gem in remaining_items
            gem_obj = Const(gem)
            color = Symbol(split(string(gem), "_")[1])
            reward = current_beliefs[color]
            heuristics[i].rewards[color] = reward

            if reward >= 0
                push!(goals, pddl"(has $agent $gem_obj)")
                push!(rewards, reward)
            end
        end

        spec = MultiGoalReward(goals, rewards, 0.95)
        sol = planners[i](domain, state, spec)
        action = best_action(sol.value_policy, state, agent)
        state = transition(domain, state, action; check=true)

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
            tr, _ = Gen.generate(utterance_model, (1, possible_gems, possible_rewards), observation)
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
                    pf_states = pf_initialize(utterance_model, (gem_count, possible_gems, possible_rewards), alt_observation, num_particles)
                else
                    if effective_sample_size(pf_states) < ess_thresh * num_particles
                        pf_resample!(pf_states, :stratified)
                        rejuv_sel = select()
                        pf_rejuvenate!(pf_states, mh, (rejuv_sel,))
                    end
                    pf_update!(pf_states, (total_gems_picked_up, possible_gems, possible_rewards), (UnknownChange(),), alt_observation)
                end
                current_pf_state = pf_states
            else
                if pf_states[agent] === nothing
                    pf_states[agent] = pf_initialize(utterance_model, (gem_count, possible_gems, possible_rewards), alt_observation, num_particles)
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
            utilities, certainties = calculate_gem_utility(gem_certainty, risk_aversion=0.2)
            # Update utilities for all agents if sharing beliefs, otherwise just for the current agent
            if share_beliefs
                gem_utilities = utilities
            else
                gem_utilities[agent] = utilities
            end
            
            println("       Estimated Rewards:")
            for (gem, value) in utilities
                println("              $gem: value = $value, certainty = $(round(certainties[gem], digits=2))")
            end
        end

        push!(actions, action)
    end
    t += 1
end

println("\nFinal Score: $combined_score\n")

# Animate the plan
anim = anim_plan(renderer, domain, initial_state, actions; format="gif", framerate=2)
save(output_folder*"/plan.mp4", anim)