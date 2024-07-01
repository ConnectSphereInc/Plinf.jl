using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie

include("utils.jl")
include("random_planner.jl")

# Register PDDL array theory
println("Registering PDDL array theory")
PDDL.Arrays.register!()

# Load domain and problem
println("Loading domain")
domain = load_domain(joinpath(@__DIR__, "domain.pddl"))
println("Loading problem")
problem = load_problem(joinpath(@__DIR__, "problems", "problem-2.pddl"))

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

# Save the canvas to a file
println("Saving initial state to file")
save("examples/vision/initial_state.png", canvas)

#--- Model Configuration ---#

planner = RandomPlanner(save_search=true)

# Specify possible goals
goals = @pddl("(has carrot1)", "(has onion1)")
goal_idxs = collect(1:length(goals))
goal_names = [write_pddl(g) for g in goals]
gem_colors = PDDLViz.colorschemes[:vibrant]
goal_colors = gem_colors[goal_idxs]

# Define uniform prior over possible goals
@gen function goal_prior()
    goal ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal])
end

# Construct iterator over goal choicemaps for stratified sampling
goal_addr = :init => :agent => :goal => :goal
goal_strata = choiceproduct((goal_addr, 1:length(goals)))

# Configure agent model with domain, planner, and goal prior
agent_config = AgentConfig(
    domain, 
    planner;
    goal_config = StaticGoalConfig(goal_prior),
    replan_args = (
        prob_replan = 0.1,
        budget_dist = shifted_neg_binom,
        budget_dist_args = (2, 0.05, 1)
    ),
    act_epsilon = 0.05
)
# Define observation noise model
obs_params = ObsNoiseParams(
    (pddl"(xpos)", normal, 1.0),
    (pddl"(ypos)", normal, 1.0),
    (pddl"(forall (?i - item) (visible ?i))", 0.01),
    (pddl"(forall (?i - item) (has ?i))", 0.05),
    (pddl"(forall (?i - item) (offgrid ?i))", 0.05)
)
obs_params = ground_obs_params(obs_params, domain, state)
obs_terms = collect(keys(obs_params))

# Configure world model with planner, goal prior, initial state, and obs params
world_config = WorldConfig(
    agent_config = agent_config,
    env_config = PDDLEnvConfig(domain, state),
    obs_config = MarkovObsConfig(domain, obs_params)
)

#--- Generate Trajectory ---#

# Construct a trajectory with backtracking to perform inference on
planner = RandomPlanner(save_search=true, alternative_planner=AStarPlanner(GoalManhattan(), save_search=true))
sol = planner(domain, state, pddl"(has carrot1)")
plan = [collect(sol);]
obs_traj = PDDL.simulate(domain, state, plan)
t_obs_iter = state_choicemap_pairs(obs_traj, obs_terms; batch_size=1)

anim = anim_plan(renderer, domain, state, plan;
                 format="gif", framerate=2, trail_length=10)

save("examples/vision/plan_.mp4", anim)

#--- Online Goal Inference ---#

# Construct callback for logging data and visualizing inference
callback = DKGCombinedCallback(
    renderer, domain;
    goal_addr = goal_addr,
    goal_names = ["Carrot", "Onion"],
    goal_colors = goal_colors,
    obs_trajectory = obs_traj,
    print_goal_probs = true,
    plot_goal_bars = true,
    plot_goal_lines = true,
    render = true,
    inference_overlay = true,
    record = true
)

# Configure SIPS particle filter
sips = SIPS(world_config, resample_cond=:ess, rejuv_cond=:periodic,
            rejuv_kernel=ReplanKernel(2), period=2)

# Run particle filter to perform online goal inference
n_samples = 120
pf_state = sips(
    n_samples, t_obs_iter;
    init_args=(init_strata=goal_strata,),
    callback=callback
);

# Extract animation
anim = callback.record.animation

save("examples/vision/infer_.mp4", anim)