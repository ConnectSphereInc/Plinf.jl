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

PDDL.Arrays.register!()
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "problem-2.pddl"))
state = initstate(domain, problem)
spec = Specification(problem)
domain, state = PDDL.compiled(domain, state)

# Construct gridworld renderer
renderer_A = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    agent_renderer = (d, s) -> begin
        HumanGraphic(color=:black)
    end,
    obj_renderers = Dict(
        :carrot => (d, s, o) -> begin
            visible = !s[Compound(:has, [o])]
            CarrotGraphic(visible=visible)
        end,
        :onion => (d, s, o) -> begin
            visible = !s[Compound(:has, [o])]
            OnionGraphic(visible=visible)
        end
    ),
    show_inventory = true,
    inventory_fns = [(d, s, o) -> s[Compound(:has, [o])]],
    inventory_types = [:item],
    show_vision = true,
    vision_fns = [(d, s, o) -> s[Compound(:visible, [o])]],
    vision_types = [:item]
)
canvas_A = renderer_A(domain, state)

renderer_B = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    agent_renderer = (d, s) -> begin
        HumanGraphic(color=:black)
    end,
    obj_renderers = Dict(
        :carrot => (d, s, o) -> begin
            visible = !s[Compound(:has, [o])]
            CarrotGraphic(visible=visible)
        end,
        :onion => (d, s, o) -> begin
            visible = !s[Compound(:has, [o])]
            OnionGraphic(visible=visible)
        end
    ),
    show_inventory = true,
    inventory_fns = [(d, s, o) -> s[Compound(:has, [o])]],
    inventory_types = [:item],
    show_vision = true,
    vision_fns = [(d, s, o) -> s[Compound(:visible, [o])]],
    vision_types = [:item]
)
canvas_B = renderer_B(domain, state)

# Make the output folder
output_folder = "examples/vision/output"
if !isdir(output_folder)
    mkdir(output_folder)
end

#--- Model Configuration ---#
planner = TwoStagePlanner(save_search=true)

# Specify possible goals
goals = @pddl("(has carrot1)", "(has onion1)")
goals_string::Vector{String} = pddl_goals_to_strings(goals)
goal_count = length(goals)
goal_idxs = collect(1:goal_count)
goal_names = [write_pddl(g) for g in goals]
colors= PDDLViz.colorschemes[:vibrant]
goal_colors = colors[goal_idxs]

# Define uniform prior over possible goals
@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

goal_addr = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))
obs_terms = [
    pddl"(xpos)",
    pddl"(ypos)",
    pddl"(forall (?i - item) (visible ?i))",
    pddl"(forall (?i - item) (has ?i))",
    pddl"(forall (?i - item) (offgrid ?i))"
]
obs_terms = vcat([ground_term(domain, state, term) for term in obs_terms]...)
agent_config = AgentConfig(domain, planner; goal_config = StaticGoalConfig(goal_prior))

#--- Interactive Simulation ---#

# Human utters a command in natural language
utterance = "Can you grab an onion?"
println("Human uttered: $utterance")

# Assistant A infers the humans goal from the human's utterance
observations = choicemap((:utterance => :output, utterance))
traces, weights = importance_sampling(utterance_model, (goals_string, domain, state), observations, 50)
probs = calculate_goal_probs(traces, weights)
most_likely_goal = argmax(probs)
println("Assistant A believes the most likely goal of the human is: $most_likely_goal")

# Assistance A forms a plan to achieve the inferred goal of the human
parsed_goal = PDDL.parse_pddl(most_likely_goal)
sol_A = TwoStagePlanner()(domain, state, parsed_goal)
plan_A = [collect(sol_A);]
obs_traj_A::Vector{State} = PDDL.simulate(domain, state, plan_A)
num_steps_A = length(obs_traj_A)

initial_world_config_A = WorldConfig(
    agent_config = agent_config,
    env_config = PDDLEnvConfig(domain, state),
    obs_config = PerfectObsConfig(domain::Domain, obs_terms)
)
initial_world_config_B = WorldConfig(
    agent_config = agent_config,
    env_config = PDDLEnvConfig(domain, state),
    obs_config = PerfectObsConfig(domain::Domain, obs_terms)
)

vips_A = VIPS(initial_world_config_A, domain)
vips_B = VIPS(initial_world_config_B, domain)

callback_A = VIPSGridworldCallback(
    renderer_A,
    vips_A.domain,
    obs_trajectory=obs_traj_A,
    record=true
)

# todo: update T_max
T_max = num_steps_A
initial_goal_probs = fill(1.0 / goal_count, goal_count)
callback_A(0, [], initial_goal_probs)

global assistant_b_planned::Bool = false

# For all of the timesteps
for t in 1:T_max-1

    # Assistant B infers the goal of Assistant A
    obs_current::ChoiceMap = state_choicemap(obs_traj_A[t], obs_terms)
    obs_next::ChoiceMap = state_choicemap(obs_traj_A[t+1], obs_terms)
    total_observations = choicemap()
    set_submap!(total_observations, :init => :obs, obs_current)
    set_submap!(total_observations, :timestep => 1 => :obs, obs_next)
    traces_obs, weights_obs = importance_sampling(world_model, (1, initial_world_config_A), total_observations, 100)
    callback_A(t, traces_obs, weights_obs)
    probs_obs = calculate_goal_probs_world_model(traces_obs, weights_obs)
    most_likely_goal_obs = goals_string[argmax(probs_obs)]
    most_likely_goal_probs = probs_obs[argmax(probs_obs)]

    # If the Assistant B is certain of this goal, they will carry out another action to assist the human.
    if most_likely_goal_probs > 0.7 && assistant_b_planned == false
        # todo: this should be formalised into an assistant policy
        remaining_goals::Vector{String} = filter(goal -> goal != most_likely_goal_obs, goals_string)
        assistant_b_goal::String = rand(remaining_goals)
        println("Assistant B is confident that Assistant A is aiming to achieve the goal: $most_likely_goal_obs.")
        println("Assistant B chooses to achieve the goal: $assistant_b_goal")
        local parsed_goal = PDDL.parse_pddl(assistant_b_goal)
        sol_B = TwoStagePlanner()(domain, state, parsed_goal)
        global plan_B = [collect(sol_B);]
        obs_traj_B::Vector{State} = PDDL.simulate(domain, state, plan_B)
        global callback_B = VIPSGridworldCallback(renderer_B, vips_B.domain, obs_trajectory=obs_traj_B, record=true)
        global assistant_b_planned = true
    end

    # Assistant A takes the planned action
    vips_A.world_config.env_config = PDDLEnvConfig(vips_A.domain, obs_traj_A[t+1])

end

anim_A = anim_plan(renderer_A, domain, state, plan_A; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan_A.mp4", anim_A)


anim_B = anim_plan(renderer_B, domain, state, plan_B; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan_B.mp4", anim_B)

