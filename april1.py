import numpy as np
import random
import matplotlib.pyplot as plt

def world_environment(value, iteration):
    randomness = 1 - (iteration * 0.01)
    return random.uniform(-1 * randomness, randomness)

def rebalance_weights(weights):
    min_weight = min(weights)
    max_weight = max(weights)
    rebalanced_weights = [((w - min_weight) / (max_weight - min_weight)) * 2 - 1 for w in weights]
    return rebalanced_weights

def main():
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
    print(f"Initial weights: {weight_dict}")

    plot_weights = []
    plot_iterations = []

    for iteration in range(100):
        rebalanced_weights = rebalance_weights(list(weight_dict.values()))
        weight_dict = dict(zip(self_awareness_aspects, rebalanced_weights))
        estimated_output = sum(rebalanced_weights) / len(rebalanced_weights)

        response = world_environment(estimated_output, iteration)

        # Update weights based on the environment's response
        updated_weights = [w + response for w in rebalanced_weights]
        weight_dict = dict(zip(self_awareness_aspects, updated_weights))

        plot_weights.append(updated_weights)
        plot_iterations.append(iteration)

    print(f"Final weights: {weight_dict}")

    # Create the plot
    plt.figure()
    for i, aspect in enumerate(self_awareness_aspects):
        plt.plot(plot_iterations, [w[i] for w in plot_weights], label=aspect)
    plt.xlabel("Iteration")
    plt.ylabel("Weights")
    plt.title("Weights Stabilization as Randomness Decreases")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.show()

if __name__ == "__main__":
    main()