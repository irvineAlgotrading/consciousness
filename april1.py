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
    weights = np.random.uniform(-5, 5, 20)
    print(f"Original weights: {weights}")

    # Initialize lists for storing weights and iteration numbers for the plot
    plot_weights = []
    plot_iterations = []

    for iteration in range(100):
        rebalanced_weights = rebalance_weights(weights)
        estimated_output = sum(rebalanced_weights) / len(rebalanced_weights)

        response = world_environment(estimated_output, iteration)

        # Update weights based on the environment's response
        weights = [w + response for w in rebalanced_weights]

        plot_weights.append(weights)
        plot_iterations.append(iteration)

    print(f"Final weights: {weights}")

    # Create the plot
    plt.figure()
    plt.plot(plot_iterations, plot_weights)
    plt.xlabel("Iteration")
    plt.ylabel("Weights")
    plt.title("Weights Stabilization as Randomness Decreases")
    plt.show()

if __name__ == "__main__":
    main()
