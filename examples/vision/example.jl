using PDDL, Printf
using SymbolicPlanners, Plinf
using Gen, GenParticleFilters
using PDDLViz, GLMakie

include("utils.jl")

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
println("Initial state: $state")
println("Constructing goal specification")
spec = Specification(problem)

# Compile domain for faster performance
println("Compiling domain")
domain, state = PDDL.compiled(domain, state)

#--- Define Renderer ---#
println("Defining renderer")
# Construct gridworld renderer
gem_colors = PDDLViz.colorschemes[:vibrant]
renderer = PDDLViz.GridworldRenderer(
    resolution = (600, 700),
    agent_renderer = (d, s) -> begin
        println("Rendering agent")
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


#--- Visualize Plans ---#

# Check that A* heuristic search correctly solves the problem
planner = AStarPlanner(GoalManhattan(), save_search=true)
sol = planner(domain, state, spec)

# Visualize resulting plan
plan = collect(sol)


# Print visibility status after each step in the plan
for (i, step) in enumerate(plan)
    println("Step $i: $step")
    local current_state = sol.trajectory[i]
    println("Current state: $current_state")
end

canvas = renderer(canvas, domain, state, plan)
@assert satisfy(domain, sol.trajectory[end], problem.goal) == true

# Visualise search tree
canvas = renderer(canvas, domain, state, sol, show_trajectory=false)

# Animate plan
anim = anim_plan(renderer, domain, state, plan;
                 format="gif", framerate=5, trail_length=10)

save("examples/vision/plan.mp4", anim)
