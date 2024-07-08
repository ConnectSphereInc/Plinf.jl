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
# planner = AStarPlanner(GoalManhattan(), save_search=true)
planner = TwoStagePlanner(save_search=true)

# Specify possible goals
goals = @pddl("(has carrot1)", "(has onion1)")
goal_idxs = collect(1:length(goals))
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

obs_params = ObsNoiseParams(
    (pddl"(xpos)", normal, 1.0),
    (pddl"(ypos)", normal, 1.0),
    (pddl"(forall (?i - item) (visible ?i))", 0.01),
    (pddl"(forall (?i - item) (has ?i))", 0.01),
    (pddl"(forall (?i - item) (offgrid ?i))", 0.01)
)
obs_params = ground_obs_params(obs_params, domain, state)
obs_terms = collect(keys(obs_params))
println("Obs terms")
println(obs_terms)
agent_config = AgentConfig(domain,  planner; goal_config = StaticGoalConfig(goal_prior))

#--- Generate Trajectory ---#

# Construct a trajectory with backtracking to perform inference on
sol = AStarPlanner(GoalManhattan(), save_search=true)(domain, state, pddl"(has carrot1)")
plan = [collect(sol);]
obs_traj::Vector{State} = PDDL.simulate(domain, state, plan)
num_steps = length(obs_traj)

anim = anim_plan(renderer, domain, state, plan;
                 format="gif", framerate=2, trail_length=10)

save("examples/vision/plan_.mp4", anim)

#--- Online Goal Inference ---#

# Run particle filter to perform online goal inference

# Number of particles to sample
n_samples = 10000

states_split::Vector{Vector{State}} = []
t_obs_split::Vector{Vector{Pair{Int64, DynamicChoiceMap}}} = []
for t in eachindex(obs_traj)
    if t+1 < num_steps
        push!(states_split, [obs_traj[t],obs_traj[t+1]])
        t_to = state_choicemap_pairs([obs_traj[t],obs_traj[t+1]], obs_terms; batch_size=1)
        push!(t_obs_split, t_to)
    end
end

# Do new particle filtering over each step
for t in eachindex(obs_traj)
    if t+1 < num_steps
        local t_obs_iter = t_obs_split[t]
        # Construct callback for logging data and visualizing inference
        local callback = DKGCombinedCallback(
            renderer, domain;
            goal_addr = goal_addr,
            goal_names = ["Carrot", "Onion"],
            goal_colors = goal_colors,
            obs_trajectory = [obs_traj[t],obs_traj[t+1]],
            print_goal_probs = false,
            plot_goal_bars = true,
            plot_goal_lines = false,
            render = true,
            inference_overlay = true,
            record = true
        )
        world_config = WorldConfig(
            agent_config = agent_config,
            env_config = PDDLEnvConfig(domain, obs_traj[t]),
            obs_config = MarkovObsConfig(domain, obs_params)
        )
        sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none)

        pf_state = sips(
            n_samples, t_obs_iter;
            init_args=(init_strata=goal_strata,),
            callback=callback
        );

        local anim = callback.record.animation

        save("examples/vision/infer_$(t).mp4", anim)
    end

end
