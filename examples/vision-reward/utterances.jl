using Gen, GenGPT3
using Random
using DotEnv
import Gen: ParticleFilterState
import GenParticleFilters: pf_initialize, pf_update!, pf_resample!, pf_rejuvenate!, pf_move_accept!, get_norm_weights, get_traces, effective_sample_size

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

global gems = ["red", "blue", "yellow", "green"]
global g_rewards::Dict{String, Int} = Dict("red" => 5, "blue" => 3, "yellow" => 1, "green" => -1)

global EXAMPLES_PICKUP = [
    ("Gem: red\nReward: +5", "I picked up a red gem and got +5 reward!"),
    ("Gem: blue\nReward: +2", "Blue gave me +2 reward."),
    ("Gem: yellow\nReward: +1", "yellow gem gave me +1 reward."),
    ("Gem: green\nReward: -1", "I got a -1 reward from the green gem."),
    ("Gem: red\nReward: +5", "Wow! A red gem just gave me a +5 reward boost!"),
    ("Gem: blue\nReward: +2", "Found a blue gem. Nice +2 reward."),
    ("Gem: yellow\nReward: +1", "Picked up a yellow, small +1 reward but still good."),
    ("Gem: green\nReward: -1", "Ouch, green gem with a -1 penalty."),
    ("Gem: red\nReward: +5", "Jackpot! red gem with a solid +5 reward."),
    ("Gem: blue\nReward: +2", "Not bad, blue gem adding +2 to my score."),
    ("Gem: yellow\nReward: +1", "yellow gem, just a +1 bump but I'll take it."),
    ("Gem: green\nReward: -1", "Darn, picked up a green. There goes 1 point."),
]
global EXAMPLES_NO_PICKUP = [
    ("Gem: none\nReward: +0", "I haven't come across a gem."),
    ("Gem: none\nReward: +0", "I haven't seen one yet."),
    ("Gem: none\nReward: +0", "No gems in sight so far."),
    ("Gem: none\nReward: +0", "Still searching for gems."),
    ("Gem: none\nReward: +0", "Haven't found any gems to pick up."),
    ("Gem: none\nReward: +0", "My inventory is empty, no gems collected."),
    ("Gem: none\nReward: +0", "I'm yet to encounter any gems on my path."),
    ("Gem: none\nReward: +0", "No luck finding gems at the moment."),
    ("Gem: none\nReward: +0", "I'm gem-less right now."),
    ("Gem: none\nReward: +0", "My gem count remains at zero."),
    ("Gem: none\nReward: +0", "The search for gems continues, none found yet."),
    ("Gem: none\nReward: +0", "I'm still on the lookout for gems."),
]

# Random.seed!(0)
shuffle!(EXAMPLES_PICKUP)
shuffle!(EXAMPLES_NO_PICKUP)

function construct_prompt(context::String, examples::Vector{Tuple{String, String}})
    example_strs = ["$ctx\nUtterance: $utt" for (ctx, utt) in examples]
    example_str = join(example_strs, "\n")
    prompt = "$example_str\n$context\nUtterance:"
    return prompt
end

function get_top_weighted_rewards(state::ParticleFilterState, n::Int)
    traces = get_traces(state)
    weights = get_norm_weights(state)
    reward_weights = Dict()
    for (tr, w) in zip(traces, weights)
        rewards = Dict{String, Int}()
        for gem in gems
            rewards[gem] = tr[:reward => Symbol(gem)]
        end
        rewards_tuple = Tuple(sort(collect(rewards)))
        reward_weights[rewards_tuple] = get(reward_weights, rewards_tuple, 0.0) + w
    end
    weighted_rewards = [(Dict(zip(gems, k)), v) for (k, v) in reward_weights]
    sort!(weighted_rewards, by = x -> x[2], rev = true)
    return values(weighted_rewards[1:min(n, length(weighted_rewards))])
end


@gen function utterance_model(T::Int)
    global EXAMPLES_PICKUP, EXAMPLES_NO_PICKUP, gems, g_rewards    
    
    # What are the rewards for each gem?
    rewards::Dict{String, Int} = Dict()
    for gem in gems
        rewards[gem] = {:reward => Symbol(gem)} ~ labeled_uniform(collect(values(g_rewards)))
    end

    utterances = []
    for t = 1:T
        # Picked up a gem?
        gem_pickup = {t => :gem_pickup} ~ bernoulli(0.5)
        if gem_pickup
            # Which gem?
            gem = {t => :gem} ~ labeled_uniform(gems)
            # reward = {t => :reward => Symbol(gem)} ~ labeled_uniform(collect([rewards[gem]]))
            reward = rewards[gem]
            context = "Gem: $gem\nReward: $reward"
            prompt = construct_prompt(context, EXAMPLES_PICKUP)
        else
            gem = "none"
            prompt = construct_prompt("Gem: $gem\nReward: 0", EXAMPLES_NO_PICKUP)
        end

        # What was uttered?
        utterance = {t => :utterance} ~ gpt3(prompt)
        push!(utterances, String(strip(utterance)))
    end
    return utterances
end

function gem_from_utterance(utterance::String)
    probs = Dict{String, Float64}()
    constraints = Gen.choicemap()
    constraints[(1 => :utterance => :output)] = utterance
    constraints[(1 => :gem_pickup)] = true
    for gem in gems
        constraints[(1 => :gem)] = gem
        _, w = Gen.generate(utterance_model, (1,), constraints)
        probs[gem] = exp(w)
    end
    most_likely = argmax(probs)
    return most_likely
end

function reward_from_utterance(utterance::String, gem::String)
    global rewards
    probs = Dict{Int, Float64}()
    constraints = Gen.choicemap()
    constraints[(1 => :utterance => :output)] = utterance
    constraints[(1 => :gem_pickup)] = true
    constraints[(1 => :gem)] = gem
    for reward in values(g_rewards)
        constraints[(:reward => Symbol(gem))] = reward
        _, w = Gen.generate(utterance_model, (1,), constraints)
        probs[reward] = exp(w)
    end
    most_likely = argmax(probs)
    return most_likely
end

# function gem_from_utterance(utterance::String)
#     color_pattern = r"\b(red|blue|yellow|green)\b"
#     match_result = match(color_pattern, lowercase(utterance))
    
#     if match_result !== nothing
#         return String(match_result.match)
#     end

#     return nothing
# end

# function reward_from_utterance(utterance::String, gem::String)
#     score_pattern = r"\b(-1|1|3|5)\b"
#     match_result = match(score_pattern, utterance)
    
#     if match_result !== nothing
#         return parse(Int, match_result.match)
#     end

#     return nothing
# end

function particle_filter(utterances, n_particles, ess_thresh=0.5)
    n_obs = length(utterances)
    observations = []

    for (i, utterance) in enumerate(utterances)
        gem = gem_from_utterance(utterance)
        reward = reward_from_utterance(utterance, gem)

        println("gem: $gem, reward: $reward")

        observation = Gen.choicemap()
        observation[i => :utterance => :output] = utterance
        observation[i => :gem_pickup] = true
        observation[i => :gem] = gem
        push!(observations, observation)
    end

    state = pf_initialize(utterance_model, (1,), observations[1], n_particles)
    for t=2:n_obs
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, :residual)
            rejuv_sel = select()
            pf_rejuvenate!(state, mh, (rejuv_sel,))
        end
        pf_update!(state, (t,), (UnknownChange(),), observations[t])
    end
    return state
end

utterances = [
    "Found a blue gem for 3!",
    "This yellow gem gave me +1 reward.",
    # "Picked up a red gem, +5!",
]

state = particle_filter(utterances, 20)

# trace = get_traces(state)[1]
# println(display(Gen.get_choices(trace)))

top_rewards = get_top_weighted_rewards(state, 10)
println("Top 5 most likely reward estimates:")
for (i, (rewards, weight)) in enumerate(top_rewards)
    println("$i. Rewards: $rewards, Weight: $(round(weight, digits=3))")
end
