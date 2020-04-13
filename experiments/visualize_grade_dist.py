import sys
import csv, os
from tqdm import tqdm
import matplotlib.pyplot as plt, numpy as np
from music21 import converter
import pandas as pd
from itertools import chain

sys.path[0] += '/../'
from transformer_bach.DatasetManager.chorale_dataset import ChoraleDataset
from transformer_bach.DatasetManager.dataset_manager import DatasetManager
from transformer_bach.DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from transformer_bach.utils import *
from Grader.grader import FEATURES
from Grader.helpers import get_threshold


def main():
    plot_selections_per_iteration()


def plot_distributions(data_files=['results/bach_grades.csv', 'results/unconstrained_mock_grades.csv'],
                       labels=['Bach chorales', 'Unconstrained mock chorales'],
                       feature='grade',
                       plt_title='Grade distributions of Bach chorales and unconstrained generations',
                       plt_dir='plots/',
                       threshold=None):
    """
    Arguments
        data_files: list of files containing distributions to compare
        labels: list of corresponding labels
        feature: feature of interest (either overall grade or a feature distance)
        plt_title: title of plot
        plt_dir: directory to save plots
        threshold: lower threshold for inclusion

    compare grade distributions as boxplot and as histogram
    """
    assert len(data_files) == len(labels)
    
    data_dict = {}
    for data_file, label in zip(data_files, labels):
        df = pd.read_csv(data_file)
        df = df.replace(float('-inf'), np.nan).dropna(subset=['grade'])
        data_dict[label] = df[feature]
        
        if threshold:
            data_dict[label] = [x for x in data_dict[label] if x > -200]

    # plot distributions
    plt.figure()
    bins = np.histogram(list(chain(*data_dict.values())), bins=100)[1]
    for label, data in data_dict.items():
        plt.hist(data, label=label, alpha=0.5, bins=bins)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(plt_title)
    plt.legend()
    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'{feature}_dist'))
    plt.close()

    # plot boxplots
    plt.figure()
    fig, ax = plt.subplots()
    ax.boxplot(list(data_dict.values()))
    ax.set_xticklabels(labels)
    plt.ylabel(feature)
    plt.title(plt_title)
    plt.savefig(os.path.join(plt_dir, f'{feature}_boxplot'))
    plt.close()


def plot_boxplot_per_iteration(data_file='results/update_grades_over_bach_chorales.csv', 
                               feature='grade', 
                               plt_dir='plots/', 
                               threshold=None):
    """
    Arguments
        data_file: file containing upgrade grades
        feature: feature of interest (either overall grade or a feature distance)
        plt_dir: directory to save plots
        threshold: lower threshold for inclusion

    visualize model updates by plotting boxplot for grade distribution at each iteration
    """
    # read update data as dictionary
    data_dict = read_update_data(data_file=data_file, feature=feature, threshold=threshold)
    
    # format plot
    plt.figure()
    fig, ax = plt.subplots()
    ax.boxplot(list(data_dict.values()))
    ax.set_xticklabels([str(i) for i in data_dict.keys()])
    plt.xlabel('Iteration')
    plt.title(f'{feature} distribution of generations after each update iteration')
    plt.ylabel(feature)
    
    thres = get_threshold(data_file='results/bach_grades.csv', feature=feature)
    plt.axhline(y=thres, color='steelblue', linestyle='-')

    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'{feature}_update_boxplots.png'))


def plot_histogram_per_iteration(data_file='results/update_grades_over_bach_chorales.csv', 
                                 feature='grade',
                                 plot_dir='plots/',
                                 threshold=None):
    """
    visualize model updates by plotting histogram for grade distribution at each iteration
    """
    # read update data as dictionary
    data_dict = read_update_data(data_file=data_file, feature=feature, threshold=threshold)

    plt.figure(figsize=(20,10))
    for it, data in data_dict.items():
        plt.subplot(2, 5, it)
        plt.hist(data, alpha=0.7)
        plt.xlabel(feature)
        plt.title(f'Iteration {it}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'{feature} distribution of generations at each iteration of training', fontsize=20)
    ensure_dir(plot_dir)
    plt.savefig(os.path.join(plot_dir, f'{feature}_update_dist.png'))
    plt.close()


def plot_selections_per_iteration(data_file='results/update_grades_over_bach_chorales.csv',
                                  plot_dir='plots/'):
    """
    plot number of selections each iteration
    """
    thres = get_threshold(feature='grade')
    data_dict = read_update_data(data_file=data_file, feature='grade')
    picked = [np.sum([1 for x in data if x > thres]) for data in data_dict.values()]
    
    plt.figure()
    rects = plt.bar(range(1, len(picked)+1), picked)
    label_bars(rects)
    plt.xlabel('Iteration')
    plt.ylabel('Number of selected generations')
    plt.title('Number of selected generations in each update iteration')
    plt.savefig(os.path.join(plot_dir, 'selections_per_iteration.png'))


def label_bars(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')


def read_update_data(data_file, feature, threshold=None):
    df = pd.read_csv(data_file)
    update_iterations = np.max(df['iter'])
    data_dict = {}
    for it in range(update_iterations + 1):
        grades = df.loc[df['iter'] == it][feature]
        if threshold:
            grades = [x for x in grades if x > -200]
        data_dict[it+1] = grades
    
    return data_dict


if __name__ == '__main__':
    main()
