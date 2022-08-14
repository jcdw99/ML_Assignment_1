import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from random import randint


def get_n_colors(n):
    color = []
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    return color


def plot_tree_results():
    criterions = ['gini', 'entropy', 'log_loss']
    splitters = ['best', 'random']
    dex = 0

    raw_data = {}
    for crit in criterions:
        for split in splitters:
            test_name =  split + '_' + crit
            file_name = 'results/tree/' + test_name + '.csv'
            raw_data[test_name] = np.linalg.norm(np.array(pd.read_csv(file_name).loc[20]) - np.array([1, 0]))

    furthest = raw_data[str(max(raw_data, key=raw_data.get))]
    closest = raw_data[str(min(raw_data, key=raw_data.get))]
    inter = furthest - closest
    best_combo = str(min(raw_data, key=raw_data.get))
    worst_combo = str(max(raw_data, key=raw_data.get))

    for i in raw_data:
        raw_data[str(i)] = (raw_data[str(i)] - closest) / inter

    sort_data = dict(sorted(raw_data.items(), key=lambda item: item[1]))
    color_data = []
    for crit in criterions:
        for split in splitters:
            color_data.append(sort_data[split + '_' + crit])

    cmap = matplotlib.cm.get_cmap('RdYlGn_r')
    for crit in criterions:
        for split in splitters:

                test_name = split + '_' + crit
                file_name = 'results/tree/' + test_name + '.csv'
                observation = pd.read_csv(file_name)
                sizes = [20] * 20
                sizes.append(100)
                alphas = list(np.array(sizes) / 120)
                if test_name == best_combo:
                    alphas[len(alphas) - 1] = 1
                    sizes[len(sizes) - 1] = sizes[len(sizes) - 1] * 1.5
                    plt.scatter(observation['means']*100, observation['std'], s=sizes, alpha=alphas, color='green', label='Best: ' + best_combo, edgecolors='black')            
                elif test_name == worst_combo:
                    alphas[len(alphas) - 1] = 1
                    sizes[len(sizes) - 1] = sizes[len(sizes) - 1] * 1.5
                    plt.scatter(observation['means']*100, observation['std'], s=sizes, alpha=alphas, color='Red', label='Worst: ' + worst_combo, edgecolors='black')
                else:
                    plt.scatter(observation['means']*100, observation['std'], s=sizes, alpha=alphas, color=cmap(color_data[dex]))
                dex = dex + 1
    
    plt.title("Classification Tree Configuration Comparison")
    plt.colorbar(label="Classification Performance")
    plt.legend(loc='upper left')
    plt.xlabel("Classification Accuracy (%)")
    plt.ylabel('Std Deviation')
    plt.show()


def plot_knn_results():
    ks = [3, 5, 7, 9]
    weights = ['uniform', 'distance']
    distances = ['manhattan', 'euclidean', 'minkowski', 'chebyshev']
    dex = 0
    # colors = get_n_colors(len(ks) * len(weights) * len(distances))
    # build ranks?
    raw_data = {}
    for k in ks:
        for weight in weights:
            for dist in distances:
                test_name = str(k) + '_' + weight + '_' + dist
                file_name = 'results/knn/' + test_name + '.csv'
                raw_data[test_name] = np.linalg.norm(np.array(pd.read_csv(file_name).loc[20]) - np.array([1, 0]))
    
    furthest = raw_data[str(max(raw_data, key=raw_data.get))]
    closest = raw_data[str(min(raw_data, key=raw_data.get))]
    inter = furthest - closest
    best_combo = str(min(raw_data, key=raw_data.get))
    worst_combo = str(max(raw_data, key=raw_data.get))
    for i in raw_data:
        raw_data[str(i)] = (raw_data[str(i)] - closest) / inter
    sort_data = dict(sorted(raw_data.items(), key=lambda item: item[1]))
    
    color_data = []

    for k in ks:
        for weight in weights:
            for dist in distances:
                color_data.append(sort_data[str(k) + '_' + weight + '_' + dist])

    cmap = matplotlib.cm.get_cmap('RdYlGn_r')

    for k in ks:
        for weight in weights:
            for dist in distances:
            
                test_name = str(k) + '_' + weight + '_' + dist
                file_name = 'results/knn/' + test_name + '.csv'
                observation = pd.read_csv(file_name)

                sizes = [20] * 20
                sizes.append(100)
                alphas = list(np.array(sizes) / 120)
                if test_name == best_combo:
                    alphas[len(alphas) - 1] = 1
                    sizes[len(sizes) - 1] = sizes[len(sizes) - 1] * 1.5
                    plt.scatter(observation['means']*100, observation['std'], s=sizes, alpha=alphas, color='green', label='Best: ' + best_combo, edgecolors='black')
                
                elif test_name == worst_combo:
                    alphas[len(alphas) - 1] = 1
                    sizes[len(sizes) - 1] = sizes[len(sizes) - 1] * 1.5
                    plt.scatter(observation['means']*100, observation['std'], s=sizes, alpha=alphas, color='Red', label='Worst: ' + worst_combo, edgecolors='black')
                
                else:
                    plt.scatter(observation['means']*100, observation['std'], s=sizes, alpha=alphas, color=cmap(color_data[dex]))
               
                dex = dex + 1
    plt.title("KNN Configuration Comparison")
    plt.colorbar(label="Classification Performance")
    plt.legend(loc='upper left')
    plt.xlabel("Classification Accuracy (%)")
    plt.ylabel('Std Deviation')
    plt.show()


plot_tree_results()
plot_knn_results()