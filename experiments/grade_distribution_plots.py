import sys
sys.path[0] += '/../'

import csv, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from music21 import converter
import pandas as pd
from itertools import chain
import seaborn as sns

from transformer_bach.DatasetManager.chorale_dataset import ChoraleDataset
from transformer_bach.DatasetManager.dataset_manager import DatasetManager
from transformer_bach.DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from transformer_bach.utils import ensure_dir
from experiments.helpers import label_bars, read_training_data
from Grader.grader import FEATURES
from Grader.helpers import get_threshold

aug_gen_dir = 'models/aug-gen_06-04_23:10'
base_dir = 'models/base_06-02_06:55'
baseline_dir = 'models/baseline_06-03_00:00'
bach_dir = 'chorales/bach_chorales'

def main():
    bach_data = list(pd.read_csv(f'{bach_dir}/grades.csv')['grade'])
    # aug_gen_data = list(read_training_data(f'{aug_gen_dir}/grades.csv', feature='grade')[20])
    # base_data = list(read_training_data(f'{base_dir}/grades.csv', feature='grade')[17])
    baseline_data = list(read_training_data(f'{baseline_dir}/grades.csv', feature='grade')[17])

    aug_gen_df = pd.read_csv(f'{aug_gen_dir}/351_mocks/grades.csv')
    aug_gen_data = aug_gen_df['grade']
    base_df = pd.read_csv(f'{base_dir}/351_mocks/grades.csv')
    base_data = base_df['grade']
    base_data = [x for x in base_data if x < 50]
    baseline_data = [x for x in baseline_data if x < 50]
    data_dict = {
        'Bach': bach_data, 
        'Aug-Gen\n' + r'($t=Q_3$ of Bach grades)': aug_gen_data, 
        'Baseline-none\n' + r'($t=-\infty$)': base_data, 
        'Baseline-all\n' + r'($t=\infty$)': baseline_data,
    }

    for model, data in data_dict.items():
        print(model)
        print(np.median(data))
        print(np.std(data))

    plot_violinplots(
        data_dict=data_dict, 
        plt_title='Grade Distribution of Generations from Different Models', 
        plt_dir='plots/'    
    )


def plot_histograms(data_dict,
                    plt_title,
                    plt_dir,
                    plt_name=None,  
                    threshold=None):
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
    # remove grades of -inf
    for label in data_dict:
        data_dict[label] = [x for x in data_dict[label] if x != float('-inf')]

    if threshold is not None:
        data_dict[label] = [x for x in data_dict[label] if x < threshold]
    
    bins = np.histogram(list(chain.from_iterable(data_dict.values())), bins=100)[1]
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(6,4))
    plt.style.use('seaborn-whitegrid')
    ax.xaxis.grid(False)
    for label, data in data_dict.items():
        plt.hist(data, label=label, alpha=0.5, bins=bins)
    plt.xlabel('Grade')
    plt.ylabel('Frequency')
    plt.title(plt_title)
    plt.legend()
    ensure_dir(plt_dir)
    if plt_name is None:
        plt_name = 'grade_dist'
    fig.tight_layout()
    plt.savefig(os.path.join(plt_dir, f'{plt_name}.png'))
    plt.close()


def plot_boxplots(data_dict,
                  plt_title,
                  plt_dir):
    plt.figure()
    fig, ax = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    ax.xaxis.grid(False)
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
    fig, ax = plt.subplots(figsize=(7.2,4.4))
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
    ax.set_xticklabels(data_dict.keys(), fontsize=11)
    ax.set_ylabel('Grade', fontsize=11)
    plt.title(plt_title)
    plt.text(0.2, 1, 'better')
    plt.text(0.2, 46, 'worse')
    plt.text(0.75, 17, f'Median: ' + r'$4.91$' + '\n    ' + r'$\sigma: 1.63$')
    plt.text(1.75, 27, f'Median: ' + r'$8.02$' + '\n    ' + r'$\sigma: 2.92$')
    plt.text(2.75, 42, f'Median: ' + r'$10.62$' + '\n    ' + r'$\sigma: 5.57$')
    plt.text(3.75, 12, f'Median: ' + r'$24.10$' + '\n    ' + r'$\sigma: 7.96$')
    plt.legend(loc='upper left')
    ensure_dir(plt_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(plt_dir, f'grade_violinplot'))
    plt.close()


if __name__ == '__main__':
    main()
