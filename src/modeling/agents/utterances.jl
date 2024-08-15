export utterance_model, utterance_model_global, enum_inference, labeled_uniform, labeled_cat

using Gen, GenGPT3
using Random

#--- Utterance Model ---#

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

global EXAMPLES_LOCAL = [
    ("Visible: gem", "I see a gem!"),
    ("Visible: none", "I'm exploring."),
    ("Visible: gem", "I can see a gem!"),
    ("Visible: none", "I can't see one yet."),
    ("Visible: none", "I don't see anything."),
    ("Visible: gem", "There's a gem this way!"),
    ("Visible: gem", "I see a gem nearby."),
    ("Visible: gem", "A gem is in sight!"),
    ("Visible: gem", "I found a gem!"),
    ("Visible: none", "I'm still looking."),
    ("Visible: gem", "There's a gem over here!"),
    ("Visible: none", "No gems in view."),
    ("Visible: gem", "I spotted a gem!"),
    ("Visible: none", "I don't see anything yet."),
    ("Visible: gem", "A gem is ahead!"),
    ("Visible: gem", "Another gem is nearby."),
    ("Visible: none", "I'm still exploring."),
    ("Visible: gem", "A gem is visible!"),
    ("Visible: none", "No gems in sight at the moment."),
    ("Visible: none", "The search continues."),
    ("Visible: none", "Still on the lookout."),
    ("Visible: none", "No precious stones visible yet."),
    ("Visible: none", "The area seems empty of gems."),
    ("Visible: none", "Nothing shiny catches my eye."),
    ("Visible: none", "I'm scanning the area, but no gems so far."),
    ("Visible: none", "The hunt for gems goes on."),
    ("Visible: none", "No sign of any gems here."),
    ("Visible: none", "I'm keeping my eyes peeled, but nothing yet.")
]

global EXAMPLES_GLOBAL = [
    ("Visible: gem\nLocation: (1,2)", "I see a gem at (1,2)!"),
    ("Visible: none", "I'm exploring."),
    ("Visible: gem\nLocation: (3,4)", "I can see a gem at coordinates (3,4)!"),
    ("Visible: none", "I can't see one yet."),
    ("Visible: none", "I don't see anything."),
    ("Visible: gem\nLocation: (5,1)", "There's a gem at (5,1) this way!"),
    ("Visible: gem\nLocation: (2,6)", "I see a gem nearby at (2,6)."),
    ("Visible: gem\nLocation: (7,3)", "A gem is in sight at position (7,3)!"),
    ("Visible: gem\nLocation: (4,5)", "I found a gem at coordinates (4,5)!"),
    ("Visible: none", "I'm still looking."),
    ("Visible: gem\nLocation: (6,2)", "There's a gem over here at (6,2)!"),
    ("Visible: none", "No gems in view."),
    ("Visible: gem\nLocation: (8,4)", "I spotted a gem at location (8,4)!"),
    ("Visible: none", "I don't see anything yet."),
    ("Visible: gem\nLocation: (3,7)", "A gem is ahead at position (3,7)!"),
    ("Visible: gem\nLocation: (5,5)", "Another gem is nearby at (5,5)."),
    ("Visible: none", "I'm still exploring."),
    ("Visible: gem\nLocation: (1,8)", "A gem is visible at coordinates (1,8)!"),
    ("Visible: none", "No gems in sight at the moment."),
]

Random.seed!(0)
shuffle!(EXAMPLES_LOCAL)
shuffle!(EXAMPLES_GLOBAL)

function construct_utterance_prompt(context::String, examples::Vector{Tuple{String, String}})
    example_strs = ["$vis\nUtterance: $utt" for (vis, utt) in examples]
    example_str = join(example_strs, "\n")
    prompt = "$example_str\n$context\nUtterance:"
    return prompt
end

@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end

"""
Samples a goal and an utterance given a State.
"""
@gen function utterance_model()
    global EXAMPLES_LOCAL
    gem_visible = {:gem_visible} ~ Gen.bernoulli(0.5)

    if gem_visible
        context = "Visible: gem"
    else
        context = "Visible: none"
    end
    
    prompt = construct_utterance_prompt(context, EXAMPLES_LOCAL)
    utterance ~ gpt3(prompt)
    return strip(utterance)
end

"""
Samples a goal and an utterance given a State.
"""
@gen function utterance_model_global()
    global EXAMPLES_GLOBAL
    # Does the agent see the gem of another agent?
    partners_gem_visible = {:partners_gem_visible} ~ Gen.bernoulli(0.9)

    if partners_gem_visible
        # todo: change the upper bound to the actual grid size
        gem_x_pos = {:gem_x_pos} ~ uniform_discrete(1, 10)
        gem_y_pos = {:gem_y_pos} ~ uniform_discrete(1, 10)
        context = "Visible: gem\nLocation: ($gem_x_pos, $gem_y_pos)"
    else
        context = "Visible: none\nLocation: none"
    end
    
    prompt = construct_utterance_prompt(context, EXAMPLES_GLOBAL)
    utterance ~ gpt3(prompt)
    return strip(utterance)
end






"""
Samples a goal and an utterance given a State.
"""
@gen function utterance_model_old(goals::Vector{String}, domain::Domain, state::State)

    # Can manually set visiblity for testing purposes
    # PDDL.set_fluent!(state, true, pddl"(visible carrot1)")
    # PDDL.set_fluent!(state, false, pddl"(visible onion1)")

    items = PDDL.get_objects(domain, state, :item)
    item_visibilities = Dict(item => PDDL.satisfy(domain, state, pddl"(visible $item)") for item in items)

    # Count visible and non-visible items
    visible_count = count(values(item_visibilities))
    non_visible_count = length(items) - visible_count
    total_count = length(items)

    # Set base probabilities for visible and non-visible items
    visible_base_prob = 0.8
    non_visible_base_prob = 0.2

    # Scale non-visible base probability based on the number of items
    scaled_non_visible_base_prob = non_visible_base_prob / total_count

    # Calculate goal probabilities
    if visible_count == 0
        # If no items are visible, distribute probabilities evenly
        goal_probs = fill(1.0 / total_count, total_count)
    else
        # Calculate probabilities for visible and non-visible items
        visible_item_prob = visible_base_prob / visible_count
        non_visible_item_prob = scaled_non_visible_base_prob / max(1, non_visible_count)
        
        # Assign probabilities based on visibility
        goal_probs = [item_visibilities[item] ? visible_item_prob : non_visible_item_prob for item in items]
        
        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(goal_probs)
        goal_probs = goal_probs ./ total_prob
    end
    item_goal_probs = Dict(item => prob for (item, prob) in zip(items, goal_probs))
    # Verify that probabilities sum to 1
    prob_sum = sum(values(item_goal_probs))
    @assert isapprox(prob_sum, 1.0, atol=1e-10) "Probabilities do not sum to 1"
    goal ~ labeled_cat(goals, goal_probs)
    visible_items = [replace(item.name, r"\d+$" => "") for (item, visible) in item_visibilities if visible]
    if isempty(visible_items)
        context = "Visible objects: None"
    else
        context = "Visible objects: " * join(visible_items, ", ")
    end
    # Generate utterance based on goal and context
    prompts = construct_utterance_prompt(goal, context)
    utterance ~ gpt3_mixture(prompts)

    return (utterance, goal)
end

function enum_inference(utterance, grid_size_x, grid_size_y)
    traces = []
    weights = []
    for x in 1:grid_size_x
        for y in 1:grid_size_y
            # Gem is visible at (x, y)
            tr, w = generate(utterance_model_global, (), 
                             choicemap(:partners_gem_visible => true, :gem_x_pos => x, :gem_y_pos => y, :output => utterance))
            push!(traces, tr)
            push!(weights, w)
        end
    end
    # Gem is not visible
    tr, w = generate(utterance_model_global, (), 
                     choicemap(:partners_gem_visible => false, :output => utterance))
    push!(traces, tr)
    push!(weights, w)
    
    return traces, weights
end
