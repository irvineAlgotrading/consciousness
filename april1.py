import numpy as np
import random

def world_environment(value):
    return random.uniform(-1, 1)

def rebalance_weights(weights):
    min_weight = min(weights)
    max_weight = max(weights)
    rebalanced_weights = [((w - min_weight) / (max_weight - min_weight)) * 2 - 1 for w in weights]
    return rebalanced_weights

def main():
    weights = np.random.uniform(-5, 5, 20)
    print(f"Original weights: {weights}")

    rebalanced_weights = rebalance_weights(weights)
    print(f"Rebalanced weights: {rebalanced_weights}")

    estimated_output = sum(rebalanced_weights) / len(rebalanced_weights)
    print(f"Estimated output: {estimated_output}")

    response = world_environment(estimated_output)
    print(f"World environment response: {response}")

if __name__ == "__main__":
    main()
