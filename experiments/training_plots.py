import sys
sys.path[0] += '/../'
import os
import click

import csv
import pandas as pd
import matplotlib.pyplot as plt
from transformer_bach.utils import ensure_dir
from experiments.helpers import read_training_data, label_bars
from Grader.helpers import get_threshold
import numpy as np

# model_dir = 'models/aug-gen_05-07_22:32'
model_dir = 'models/base_05-07_22:29'
@click.command()
@click.option('--model_dir', type=click.Path(exists=True))
def main(model_dir):
    plot_learning_curves(model_dir)

    data_file = f'{model_dir}/grades.csv'
    plt_dir = f'{model_dir}/plots/'

    plot_boxplot_per_epoch(data_file=data_file, plt_dir=plt_dir, threshold=-100)
    # plot_histogram_per_iteration(data_file=data_file, plt_dir=plt_dir)
    plot_selections_per_epoch(data_file=data_file, plt_dir=plt_dir)
    plot_unique_per_epoch(data_file=f'{model_dir}/dataset_sizes.csv', plt_dir=plt_dir)
    plot_dataset_sizes_per_epoch(data_file=f'{model_dir}/dataset_sizes.csv', plt_dir=plt_dir)


def plot_learning_curves(gen_folder):
    with open(f'{gen_folder}/loss.csv', 'r') as fin:
        df = pd.read_csv(fin)
        train_loss = df['train_loss']
        val_loss = df['val_loss']

    print(df[df['val_loss'] == df['val_loss'].min()])

    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.title('Training curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ensure_dir(f'{gen_folder}/plots')
    plt.savefig(f'{gen_folder}/plots/training_curves.png')


def plot_boxplot_per_epoch(data_file='results/update_grades_over_bach_chorales.csv', 
                           feature='grade', 
                           plt_dir='plots/augmented-generation/', 
                           threshold=None):
    """
    Arguments
        data_file: file containing upgrade grades
        feature: feature of interest (either overall grade or a feature distance)
        plt_dir: directory to save plots
        threshold: lower threshold for inclusion

    visualize model updates by plotting boxplot for grade distribution at each epoch
    """
    # read update data as dictionary
    data_dict = read_training_data(data_file=data_file, feature=feature)
    
    # format plot
    plt.figure()
    fig, ax = plt.subplots()
    ax.boxplot(list(data_dict.values()))
    ax.set_xticklabels([str(i) for i in data_dict.keys()])
    for label in ax.get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)
    plt.xlabel('Epoch')
    plt.title(f'{feature} distribution of generations after each update epoch')
    plt.ylabel(feature)
    plt.ylim([threshold, 50])
    
    thres = get_threshold(feature=feature)
    plt.axhline(y=thres, color='steelblue', linestyle='-')

    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'{feature}_update_boxplots.png'))


def plot_histogram_per_iteration(data_file, 
                                 feature='grade',
                                 plt_dir='plots/augmented-generation/',
                                 threshold=None):
    """
    visualize model updates by plotting histogram for grade distribution at each iteration
    """
    # read update data as dictionary
    data_dict = read_training_data(data_file=data_file, feature=feature, threshold=threshold)

    plt.figure(figsize=(20,10))
    for it, data in data_dict.items():
        plt.subplot(2, 5, it)
        plt.hist(data, alpha=0.7)
        plt.xlabel(feature)
        plt.title(f'Iteration {it}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'{feature} distribution of generations at each iteration of training', fontsize=20)
    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'{feature}_update_dist.png'))
    plt.close()


def plot_selections_per_epoch(data_file='results/update_grades_over_bach_chorales.csv',
                              plt_dir='plots/augmented-generation/'):
    """
    plot number of selections each epoch
    """
    thres = get_threshold(feature='grade')
    data_dict = read_training_data(data_file=data_file, feature='grade')
    picked = [np.sum([1 for x in data if x > thres]) for data in data_dict.values()]
    
    plt.figure()
    rects = plt.bar(range(1, len(picked)+1), picked)
    label_bars(rects)
    plt.xlabel('Epoch')
    plt.ylabel('Number of generations passing threshold')
    plt.title('Number of generations passing threshold in each epoch')
    plt.savefig(os.path.join(plt_dir, 'generations_passing_threshold_per_epoch.png'))


def plot_unique_per_epoch(data_file, plt_dir):
    """
    plot number of unique generations each epoch
    """
    dataset_sizes_df = pd.read_csv(data_file)
    unique = dataset_sizes_df['num_unique_generations']
    
    plt.figure()
    rects = plt.bar(range(1, len(unique)+1), unique)
    label_bars(rects)
    plt.xlabel('Epoch')
    plt.ylabel('Number of unique generations')
    plt.title('Number of unique generations in each update epoch')
    plt.savefig(os.path.join(plt_dir, 'unique_per_epoch.png'))


def plot_dataset_sizes_per_epoch(data_file, plt_dir):
    """
    plot dataset size in each epoch
    """
    mock_examples = pd.read_csv(data_file)['num_examples']
    mock_chorales = pd.read_csv(data_file)['num_chorales']
    num_epochs = len(mock_examples)
    bach_examples = mock_examples[0] * num_epochs
    bach_chorales = mock_chorales[0] * num_epochs

    plt.figure()
    ind = np.arange(num_epochs)
    bach = plt.bar(ind, bach_examples, color='steelblue')
    mock = plt.bar(ind, mock_examples, bottom=bach_examples, color='lightgray')
    plt.legend((bach[0], mock[0]), ('Bach', 'Generated'))
    plt.title('Number of Bach and generated examples in dataset')
    plt.savefig(os.path.join(plt_dir, 'num_examples_per_epoch.png'))

    plt.figure()
    ind = np.arange(num_epochs)
    bach = plt.bar(ind, bach_chorales, color='steelblue')
    mock = plt.bar(ind, mock_chorales, bottom=bach_chorales, color='lightgray')
    plt.legend((bach[0], mock[0]), ('Bach', 'Generated'))
    plt.title('Number of Bach and generated chorales in dataset')
    plt.savefig(os.path.join(plt_dir, 'num_chorales_per_epoch.png'))


if __name__ == '__main__':
    main()