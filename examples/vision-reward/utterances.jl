using Gen, GenGPT3
using Random

#--- Utterance Model ---#

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

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
    # tr, w = generate(utterance_model_global, (), 
    #                  choicemap(:partners_gem_visible => false, :output => utterance))
    # push!(traces, tr)
    # push!(weights, w)
    
    return traces, weights
end
