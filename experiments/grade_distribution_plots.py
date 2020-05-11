import sys
sys.path[0] += '/../'

import csv, os
from tqdm import tqdm
import matplotlib.pyplot as plt, numpy as np
from music21 import converter
import pandas as pd
from itertools import chain
import seaborn as sns

from transformer_bach.DatasetManager.chorale_dataset import ChoraleDataset
from transformer_bach.DatasetManager.dataset_manager import DatasetManager
from transformer_bach.DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from transformer_bach.utils import ensure_dir
from Grader.grader import FEATURES
from Grader.helpers import get_threshold
from experiments.helpers import label_bars, read_training_data

aug_gen_model_dir = 'models/aug-gen_05-09_06:09'
base_model_dir = 'models/base_05-10_07:20'

def main():
    # aug_gen_data = read_training_data(f'{aug_gen_model_dir}/grades.csv', feature='grade', threshold=-1000)[19]
    # base_data = read_training_data(f'{base_model_dir}/grades.csv', feature='grade', threshold=-1000)[19]

    aug_gen_df = pd.read_csv(f'{aug_gen_model_dir}/unconstrained_mocks_s1/grades.csv')
    # aug_gen_data = aug_gen_df['grade']
    aug_gen_data = [x for x in aug_gen_df['grade'] if x >= -1000]
    print(np.median(aug_gen_data))
    print(np.sum([1 for x in aug_gen_data if x > get_threshold()]))
    base_df = pd.read_csv(f'{base_model_dir}/unconstrained_mocks_s1/grades.csv')
    base_data = [x for x in base_df['grade'] if x >= -1000]
    print(np.median(base_data))
    print(np.sum([1 for x in base_data if x > get_threshold()]))
    # base_data = base_df['grade']
    data_dict = {'Augmentative generation': aug_gen_data, 'Base model': base_data}

    plot_boxplots(
        data_dict=data_dict, 
        plt_title='Grade distributions', 
        plt_dir='plots/'    
    )

    plot_histograms(
        data_dict=data_dict, 
        plt_title='Grade distributions', 
        plt_dir='plots/'    
    )

    plot_violinplots(
        data_dict=data_dict, 
        plt_title='Grade distributions', 
        plt_dir='plots/'    
    )


def plot_histograms(data_dict,
                    plt_title,
                    plt_dir):
    """
    Arguments
        data_dict: a dictionary of data with key as label and value as list of grades/distances
            {'Bach chorales': [10, 12, ...], 'Generations': [20, 15, ...]}
        feature: feature of interest (either overall grade or a feature distance)
        plt_title: title of plot
        plt_dir: directory to save plots
        threshold: lower threshold for inclusion

    compare grade distributions as boxplot and as histogram
    """
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    ax.xaxis.grid(False)
    bins = np.histogram(list(chain(*data_dict.values())), bins=100)[1]

    for label, data in data_dict.items():
        plt.hist(data, label=label, alpha=0.5, bins=bins)
    
    plt.xlabel('Grade')
    plt.ylabel('Frequency')
    plt.title(plt_title)
    plt.legend()
    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'grade_dist'))
    plt.close()


def plot_boxplots(data_dict,
                  plt_title,
                  plt_dir):
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    ax.xaxis.grid(False)
    fig, ax = plt.subplots()
    ax.boxplot(list(data_dict.values()))
    ax.set_xticklabels(data_dict.keys())
    plt.ylabel('Grade')
    plt.title(plt_title)
    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'grade_boxplot'))
    plt.close()


def plot_violinplots(data_dict,
                     plt_title,
                     plt_dir):
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    plt.style.use('seaborn-whitegrid')
    ax.grid(False)
    r = ax.violinplot(list(data_dict.values()), showmeans=True, showmedians=True)
    r['cmedians'].set_label('Median grade')
    r['cmedians'].set_color('rebeccapurple')
    r['cmeans'].set_label('Mean grade')
    r['cmeans'].set_color('steelblue')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(data_dict.keys()) + 1))
    ax.set_xticklabels(data_dict.keys())
    ax.set_xlabel('Model')
    ax.set_ylabel('Grade')
    plt.title(plt_title)
    plt.legend(loc='lower right')
    ensure_dir(plt_dir)
    plt.savefig(os.path.join(plt_dir, f'grade_violinplot'))
    plt.close()


if __name__ == '__main__':
    main()
