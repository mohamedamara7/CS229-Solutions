import matplotlib.pyplot as plt
import numpy as np
import util

# Plotting dataset
def plot_dataset(char):
    X, Y = util.load_csv(f'ds1_{char}.csv', add_intercept=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='b', marker='x', linewidth=2)
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='g', marker='o', linewidth=2)
    plt.title(f'Dataset {char.upper()}')
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.savefig(f'dataset_{char}.png')

plot_dataset('a')
plot_dataset('b')

