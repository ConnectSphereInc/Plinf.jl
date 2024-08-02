export utterance_model, labeled_uniform, labeled_cat

using Gen, GenGPT3
using Random

#--- Utterance Model ---#

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3_mixture = GPT3Mixture(model="davinci-002", stop="\n", max_tokens=512)

COMMAND_EXAMPLES = [
    ("has(carrot)", "Visible: carrot", "Can you get that?"),
    ("has(carrot)", "Visible: apple, banana", "Grab some carrots."),
    ("has(onion)", "Visible: onion", "Please take that."),
    ("has(onion)", "Visible: potato, garlic", "We need to grab some onions."),
    ("has(apple)", "Visible: apple", "Go get that."),
    ("has(apple)", "Visible: orange, banana", "Let's grab some apples."),
    ("has(bread)", "Visible: bread", "Add that to the cart."),
    ("has(bread)", "Visible: cheese, milk", "Grab some bread."),
    ("has(cheddar_cheese)", "Visible: cheddar_cheese", "Take that one."),
    ("has(cheddar_cheese)", "Visible: milk, eggs", "We should get some cheddar cheese."),
]

# Random.seed!(0)
shuffle!(COMMAND_EXAMPLES)

function construct_utterance_prompt(command::String, context::String, examples = COMMAND_EXAMPLES)
    example_strs = ["Context: $ctx\nInput: $cmd\nOutput: $utt" for (cmd, ctx, utt) in examples]
    example_str = join(example_strs, "\n\n")
    prompt = "$example_str\n\nContext: $context\nInput: $command\nOutput:"
    return prompt
end

@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end

"""
Samples a goal and an utterance given a State.
"""
@gen function utterance_model(goals::Vector{String}, domain::Domain, state::State)

    # Can manually set visiblity for testing purposes
    # PDDL.set_fluent!(state, true, pddl"(visible carrot1)")

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
    visible_items = [item.name for (item, visible) in item_visibilities if visible]
    if isempty(visible_items)
        context = "Visible objects: None"
    else
        context = "Visible objects: " * join(visible_items, ", ")
    end
    # Generate utterance based on goal and context
    prompts = [construct_utterance_prompt(goal, context)]
    utterance ~ gpt3_mixture(prompts)

    return (utterance, goal)
end



# function utterance_inference(utterance:: String)

# end

# utterance = " Could you get that?"
# observations = choicemap(
#     (:utterance => :output, utterance),
#     (:carrot_visible, false),
#     (:onion_visible, true)
# )

# traces, weights = importance_sampling(utterance_model, (), observations, 100)



# println("\nInferred Probabilities:")
# print_probs(traces, weights)
