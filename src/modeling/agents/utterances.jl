export utterance_model, labeled_uniform, labeled_cat

using Gen, GenGPT3
using Random

#--- Utterance Model ---#

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

EXAMPLES = [
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

Random.seed!(0)
shuffle!(EXAMPLES)

function construct_utterance_prompt(context::String, examples = EXAMPLES)
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

    gem_visible = {:gem_visible} ~ Gen.bernoulli(0.5)

    if gem_visible
        context = "Visible: gem"
    else
        context = "Visible: none"
    end
    
    prompt = construct_utterance_prompt(context)
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


