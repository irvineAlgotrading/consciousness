import numpy as np
import random
import matplotlib.pyplot as plt

def world_environment(values, iteration):
    randomness = 1 - np.exp(-iteration / 100)
    return [random.uniform(-1 * randomness, randomness) for _ in values]

def rebalance_weights(weights):
    min_weight = min(weights)
    max_weight = max(weights)
    rebalanced_weights = [((w - min_weight) / (max_weight - min_weight)) * 2 - 1 for w in weights]
    return rebalanced_weights

def sgd_update(rebalanced_weights, responses, learning_rate):
    gradients = [r for r in responses]
    updated_weights = [w - learning_rate * g for w, g in zip(rebalanced_weights, gradients)]
    return updated_weights

def main():
    #these are the things it will be optimizing f
    self_awareness_aspects = [
        'body_awareness',
        'emotional_awareness',
        'introspection',
        'reflection',
        'theory_of_mind',
        'temporal_awareness',
        'self-recognition',
        'self-esteem',
        'agency',
        'self-regulation',
        'self-concept',
        'self-efficacy',
        'self-monitoring',
        'metacognition',
        'moral_awareness',
        'social_awareness',
        'situational_awareness',
        'motivation',
        'goal-setting',
        'self-development'
    ]

    # Rank the states of consciousness by their benefit (1 being the most beneficial)
    consciousness_rank = {
        1: 'self-development',
        2: 'goal-setting',
        3: 'motivation',
        4: 'self-regulation',
        5: 'self-efficacy',
        6: 'self-monitoring',
        7: 'metacognition',
        8: 'moral_awareness',
        9: 'social_awareness',
        10: 'situational_awareness',
        11: 'agency',
        12: 'self-esteem',
        13: 'self-recognition',
        14: 'temporal_awareness',
        15: 'theory_of_mind',
        16: 'reflection',
        17: 'introspection',
        18: 'emotional_awareness',
        19: 'self-concept',
        20: 'body_awareness'
    }

    # Assign initial values based on the rank
    num_aspects = len(self_awareness_aspects)
    increment = 2 / (num_aspects - 1)
    initial_values = {aspect: -1 + increment * (rank - 1) for rank, aspect in consciousness_rank.items()}
    weight_dict = {aspect: initial_values[aspect] for aspect in self_awareness_aspects}

    learning_rate = 0.01

    print("Initial Weights:")
    print(f"Aspect{' ':<20}Weight")
    for aspect, weight in sorted(weight_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{aspect:<24}{weight:.4f}")

    plot_weights = []
    plot_iterations = []
    plot_randomness = []

    for iteration in range(10000):
        rebalanced_weights = rebalance_weights(list(weight_dict.values()))
        weight_dict = dict(zip(self_awareness_aspects, rebalanced_weights))
        estimated_output = sum(rebalanced_weights) / len(rebalanced_weights)

        responses = world_environment(values=rebalanced_weights, iteration=iteration)

        # Update weights using Stochastic Gradient Descent
        updated_weights = sgd_update(rebalanced_weights, responses, learning_rate)
        weight_dict = dict(zip(self_awareness_aspects, updated_weights))

        plot_weights.append(updated_weights)
        plot_iterations.append(iteration)
        plot_randomness.append(1 - np.log10(iteration + 1) / 3.33)

    print("\nFinal Weights:")
    print(f"Aspect{' ':<20}Weight")
    for aspect, weight in sorted(weight_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{aspect:<24}{weight:.4f}")

        # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Plot the weight stabilization
    for i, aspect in enumerate(self_awareness_aspects):
        ax1.plot(plot_iterations, [w[i] for w in plot_weights], label=aspect)
    ax1.set_ylabel("Weights")
    ax1.set_title("Weights Stabilization as Randomness Decreases")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Plot the degradation of the randomness function
    ax2.plot(plot_iterations, plot_randomness)
    ax2.set_ylabel("Randomness")
    ax2.set_xlabel("Iteration")

    # Set the chart settings
    plt.subplots_adjust(left=0.065, bottom=0.125, right=0.802, top=0.88, wspace=0.2, hspace=0.2)

    plt.show()

if __name__ == "__main__":
    main()