using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie

using DotEnv

# Read in the OpenAI API key
overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

include("utils.jl")
include("random_planner.jl")

# Register PDDL array theory
println("Registering PDDL array theory")
PDDL.Arrays.register!()

# Load domain and problem
println("Loading domain")
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
println("Loading problem")
problem::Problem = load_problem(joinpath(@__DIR__, "problems", "problem-2.pddl"))

# Initialize state and construct goal specification
println("Initializing state")
state = initstate(domain, problem)
println("Constructing goal specification")
spec = Specification(problem)

# Compile domain for faster performance
println("Compiling domain")
domain, state = PDDL.compiled(domain, state)

#--- Define Renderer ---#
println("Defining renderer")

# Construct gridworld renderer
renderer = PDDLViz.GridworldRenderer(
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

# Visualize initial state
println("Visualizing initial state")
canvas = renderer(domain, state)

# Make the output folder
output_folder = "examples/vision/output"
if !isdir(output_folder)
    mkdir(output_folder)
end

# Save the canvas to a file
println("Saving initial state to file")
save(output_folder*"/initial_state.png", canvas)

#--- Model Configuration ---#
planner = TwoStagePlanner(save_search=true)

# Specify possible goals
goals = @pddl("(has carrot1)", "(has onion1)")
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

# Construct iterator over goal choicemaps for stratified sampling
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

agent_config = AgentConfig(domain,  planner; goal_config = StaticGoalConfig(goal_prior))

#--- Generate Trajectory ---#

# Human utters a command in natural language
utterance = " Can you grab an onion?"

# Assistant A infers the humans goal from the human's utterance
goals_string::Vector{String} = pddl_goals_to_strings(goals)
utterance, _ = utterance_model(goals_string, domain, state)

observations = choicemap((:utterance => :output, utterance))
traces, weights = importance_sampling(utterance_model, (goals_string, domain, state), observations, 100)
probs = calculate_goal_probs(traces, weights)
println("\nInferred Probabilities:")
println(probs)
most_likely_goal = argmax(probs)
parsed_goal = PDDL.parse_pddl(most_likely_goal)

# Assistance A carries out the task to achieve the humans goal

# Construct a trajectory with backtracking to perform inference on
sol = AStarPlanner(GoalManhattan(), save_search=true)(domain, state, parsed_goal)

plan = [collect(sol);]
obs_traj::Vector{State} = PDDL.simulate(domain, state, plan)
num_steps = length(obs_traj)

anim = anim_plan(renderer, domain, state, plan; format="gif", framerate=2, trail_length=10)
save(output_folder*"/plan_.mp4", anim)

#--- Online Goal Inference ---#

# Assistant B infers the goal of Assistant A from their actions

n_samples = 100

initial_world_config = WorldConfig(
    agent_config = agent_config,
    env_config = PDDLEnvConfig(domain, obs_traj[1]),
    obs_config = PerfectObsConfig(domain::Domain, obs_terms)
)

vips = VIPS(initial_world_config, domain)
callback = VIPSGridworldCallback(
    renderer,
    vips.domain,
    obs_trajectory=obs_traj,
    record=true
    )

vips(n_samples, obs_traj, obs_terms, callback, goal_count; init_args=(init_strata=goal_strata,))

anim = callback.record.animation
save(output_folder*"/infer.mp4", anim)
