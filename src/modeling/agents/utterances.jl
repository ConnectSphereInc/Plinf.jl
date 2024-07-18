using Gen, GenGPT3
using Random
using DotEnv

# Read in the OpenAI API key
overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

#--- Utterance Model ---#

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3_mixture = GPT3Mixture(model="davinci-002", stop="\n", max_tokens=512)

COMMAND_EXAMPLES = [
    ("has(apple)", "Can you get the apple?"),
    ("has(bread)", "Could you find some bread?"),
    ("has(cheddar_cheese)", "Go grab a block of that cheese."),
    ("has(green_tea)", "Add some tea to the cart."),
    ("has(tofu) has(seitan)", "I need some tofu and seitan."),
    ("has(frozen_mango) has(ice_cream)", "Get the mango and ice cream."),
    ("has(strawberries) has(milk)", "Find me strawberries and milk."),
    ("has(frozen_broccoli) has(frozen_cauliflower)", "We'll need frozen broccoli and cauliflower."),
    ("has(fries) checkout()", "Let's has some fries then checkout."),
]
# Random.seed!(0)
shuffle!(COMMAND_EXAMPLES)

function construct_utterance_prompt(command::String, context::String, examples = COMMAND_EXAMPLES)
    example_strs = ["Input: $cmd\nOutput: $utt" for (cmd, utt) in examples]
    example_str = join(example_strs, "\n")
    prompt = "$example_str\nContext: $context\nInput: $command\nOutput:"
    return prompt
end

@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end

@gen function utterance_model()
    carrot_visible ~ bernoulli(0.5)
    onion_visible ~ bernoulli(0.5)
    
    # Adjust goal probabilities based on visibility
    if carrot_visible && !onion_visible
        goal_probs = [0.8, 0.2]
    elseif !carrot_visible && onion_visible
        goal_probs = [0.2, 0.8]
    else
        goal_probs = [0.5, 0.5]
    end
    
    goals = ["(has carrot1)", "(has onion1)"]
    goal ~ labeled_cat(goals, goal_probs)
    
    
    # Construct context string
    context = "Visible objects: " * 
              (carrot_visible ? "carrot, " : "") * 
              (onion_visible ? "onion, " : "")
    context = String(chop(context, tail=2))  # Remove trailing comma and space
    
    # Generate utterance based on goal and context
    prompts = [construct_utterance_prompt(goal, context)]
    utterance ~ gpt3_mixture(prompts)
    
    return (utterance, goal)
end

utterance = " Could you get that?"
observations = choicemap(
    (:utterance => :output, utterance),
    (:carrot_visible, false),
    (:onion_visible, true)
)

traces, weights = importance_sampling(utterance_model, (), observations, 100)

function print_probs(traces, weights)
    probs = Dict{String, Float64}()
    for (tr, w) in zip(traces, weights)
        goal = tr[:goal]
        p = get(probs, goal, 0.0)
        probs[goal] = p + exp(w)
    end
    
    total_prob = sum(values(probs))
    for (goal, p) in probs
        println("Goal: ", goal)
        println("Probability: ", round(p / total_prob, digits=4))
    end    
end

println("\nInferred Probabilities:")
print_probs(traces, weights)