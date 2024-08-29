using Gen, GenGPT3
using Random
import Gen: ParticleFilterState
import GenParticleFilters: pf_initialize, pf_update!, pf_resample!, pf_rejuvenate!, get_norm_weights, get_traces, effective_sample_size

@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

gpt3 = GPT3GF(model="davinci-002", stop="\n", max_tokens=512, temperature=1)

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
@gen function utterance_model(T::Int, possible_gems::Vector{Symbol}, possible_rewards::Vector{Int})
    global EXAMPLES_PICKUP, EXAMPLES_NO_PICKUP
    # What are the rewards for each gem?
    rewards::Dict{Symbol, Int} = Dict()
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

# =======  Example Inference ======= #

# """
#     particle_filter(utterances, n_particles; ess_thresh=0.5, infer_gem=false)

#     Perform particle filtering to infer the rewards for each gem based on a sequence of utterances.
    
#     The agent observed a series of gem pickups and utterances and infers the rewards for each type
#     of gem. The agent can either infer the gem type from the utterance or observe the gem type directly.
# """
# function particle_filter(utterances, n_particles; ess_thresh=0.5, infer_gem=false)
#     n_obs = length(utterances)
#     observations = []
#     for (i, utterance) in enumerate(utterances)
#         observation = Gen.choicemap()
#         observation[i => :utterance => :output] = utterance
#         observation[i => :gem_pickup] = true
#         if !infer_gem
#             gem = parse_gem(utterance)
#             observation[i => :gem] = gem
#         end
#         push!(observations, observation)
#     end
#     state = pf_initialize(utterance_model, (1,), observations[1], n_particles)
#     for t=2:n_obs
#         if effective_sample_size(state) < ess_thresh * n_particles
#             pf_resample!(state, :stratified)
#             rejuv_sel = select()
#             pf_rejuvenate!(state, mh, (rejuv_sel,))
#         end
#         pf_update!(state, (t,), (UnknownChange(),), observations[t])
#     end
#     return state
# end

# utterances = [
#     "Found a blue gem for 3!",
#     "This yellow gem gave me +1 reward.",
#     "Picked up a red gem, +5!",
#     "Found a green gem for -1!",
# ]

# pf_state = particle_filter(utterances, 100, infer_gem=false)
# top_rewards = get_top_weighted_rewards(pf_state, 10)
# println("Top 5 most likely reward estimates:")
# for (i, (rewards, weight)) in enumerate(top_rewards)
#     println("$i. Rewards: $rewards, Weight: $(round(weight, digits=3))")
# end

# gem_certainty = quantify_gem_certainty(top_rewards)
# println("Gem certainty:")
# println(gem_certainty)

# gem_utilities = calculate_gem_utility(gem_certainty, risk_aversion = 0)
# println("Gem Utilities:")
# for (gem, info) in gem_utilities
#     println("$gem: value = $(info["value"]), certainty = $(round(info["certainty"], digits=2)), utility = $(round(info["utility"], digits=2))")
# end

# ================================== #
