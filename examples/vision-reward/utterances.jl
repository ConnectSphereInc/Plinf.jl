using Gen, GenGPT3
using Random
using DotEnv
import Gen: ParticleFilterState
import GenParticleFilters: pf_initialize, pf_update!, pf_resample!, pf_rejuvenate!, get_norm_weights, get_traces, effective_sample_size

overlay = DotEnv.config()
api_key = get(overlay, "OPENAI_API_KEY", nothing)
ENV["OPENAI_API_KEY"] = api_key

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

global possible_gems = ["red", "blue", "yellow", "green"]
global possible_rewards = [-1, 1, 3, 5]

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

"""
    construct_prompt(context::String, examples::Vector{Tuple{String, String}})

    Construct a prompt for the GPT-3 model based on a context and examples.
"""
function construct_prompt(context::String, examples::Vector{Tuple{String, String}})
    example_strs = ["$ctx\nUtterance: $utt" for (ctx, utt) in examples]
    example_str = join(example_strs, "\n")
    prompt = "$example_str\n$context\nUtterance:"
    return prompt
end

"""
    get_top_weighted_rewards(state::ParticleFilterState, n::Int)

    Get the top `n` most likely reward distributions based on the particle filter state.

"""
function get_top_weighted_rewards(state::ParticleFilterState, n::Int)
    traces = get_traces(state)
    weights = get_norm_weights(state)
    reward_weights = Dict()
    
    # Accumulate rewards and weights
    for (tr, w) in zip(traces, weights)
        rewards = Dict{String, String}()
        for gem in possible_gems
            rewards[gem] = string(tr[:reward => Symbol(gem)])
        end
        rewards_tuple = Tuple(sort(collect(rewards)))
        reward_weights[rewards_tuple] = get(reward_weights, rewards_tuple, 0.0) + w
    end
    
    total_weight = sum(values(reward_weights))
    weighted_rewards = [(Dict(rewards), weight / total_weight) 
                        for (rewards, weight) in reward_weights]
    
    # Sort by probability
    sort!(weighted_rewards, by = x -> x[2], rev = true)
    
    # Return top n results
    return weighted_rewards[1:min(n, length(weighted_rewards))]
end

"""
    utterance_model(T::Int)

    Generate a sequence of utterances based on simulated gem pickups and rewards.

    First, a reward is assigned to each gem. Then, for each timestep `t` in `1:T`,
    an agent either picks up a gem or not. If a gem is picked up, the agent recieves
    an award corresponding to the gem. The agent then utters a sentence based on the
    gem picked up and the reward received. If no gem is picked up, the agent utters
    a sentence based on the fact that no gem was picked up.

    At inference time, the agent observes a sequence of utterances and infers the
    rewards for each gem.
"""
@gen function utterance_model(T::Int)
    global EXAMPLES_PICKUP, EXAMPLES_NO_PICKUP, possible_gems, possible_rewards    
    
    # What are the rewards for each gem?
    rewards::Dict{String, Int} = Dict()
    for gem in possible_gems
        rewards[gem] = {:reward => Symbol(gem)} ~ labeled_uniform(possible_rewards)
    end

    utterances = []
    for t = 1:T
        # Picked up a gem?
        gem_pickup = {t => :gem_pickup} ~ bernoulli(0.5)
        if gem_pickup
            # Which gem?
            gem = {t => :gem} ~ labeled_uniform(possible_gems)
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

"""
    gem_from_utterance(utterance::String)

    Manually extract the color of a gem from an utterance using regex.
"""
function gem_from_utterance(utterance::String)
    color_pattern = r"\b(red|blue|yellow|green)\b"
    match_result = match(color_pattern, lowercase(utterance))
    
    if match_result !== nothing
        return String(match_result.match)
    end

    return nothing
end

"""
    reward_from_utterance(utterance::String, gem::String)

    Manually extract the reward from an utterance using regex.
"""
function reward_from_utterance(utterance::String)
    score_pattern = r"\b(-1|1|3|5)\b"
    match_result = match(score_pattern, utterance)
    
    if match_result !== nothing
        return parse(Int, match_result.match)
    end

    return nothing
end

"""
    particle_filter(utterances, n_particles; ess_thresh=0.5, infer_gem=false)

    Perform particle filtering to infer the rewards for each gem based on a sequence of utterances.
    
    The agent observed a series of gem pickups and utterances and infers the rewards for each type
    of gem. The agent can either infer the gem type from the utterance or observe the gem type directly.
"""
function particle_filter(utterances, n_particles; ess_thresh=0.5, infer_gem=false)
    n_obs = length(utterances)
    observations = []

    for (i, utterance) in enumerate(utterances)

        observation = Gen.choicemap()
        observation[i => :utterance => :output] = utterance
        observation[i => :gem_pickup] = true

        if !infer_gem
            gem = gem_from_utterance(utterance)
            observation[i => :gem] = gem
        end

        push!(observations, observation)
    end

    state = pf_initialize(utterance_model, (1,), observations[1], n_particles)
    for t=2:n_obs
        if effective_sample_size(state) < ess_thresh * n_particles
            pf_resample!(state, :stratified)
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

state = particle_filter(utterances, 100, infer_gem=false)

top_rewards = get_top_weighted_rewards(state, 10)
println("Top 5 most likely reward estimates:")
for (i, (rewards, weight)) in enumerate(top_rewards)
    println("$i. Rewards: $rewards, Weight: $(round(weight, digits=3))")
end
