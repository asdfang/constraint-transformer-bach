import sys
sys.path.insert(0, '../')

from tqdm import tqdm
import random
import click
import os, shutil
import importlib
from datetime import datetime
import csv
import music21
import numpy as np
import pandas as pd
from collections import defaultdict

from transformer_bach.utils import ensure_dir
from Grader.helpers import get_threshold
from transformer_bach.bach_dataloader import BachDataloaderGenerator
from transformer_bach.DatasetManager.helpers import load_or_pickle_distributions
from Grader.grader import Grader, FEATURES
from human_evaluation.helpers import is_midi

BASE_DEEPBACH_GENERATIONS = '../../deepbach/generations/10/'
BASE_TRANSFORMER_GENERATIONS = '../models/bach_decoder_config_2020-03-24_20:19:31/unconstrained_mocks/'
EXPERIMENT_DIR = '../chorales/paired_evaluation/4-14/'
NUM_COMPARISONS=351

def main():
    make_pairs(mock_dirs={'base': BASE_TRANSFORMER_GENERATIONS}, 
               num_comparisons=NUM_COMPARISONS)


def make_pairs(mock_dirs,
               num_comparisons=10):
    """
    Arguments:
        mock_dirs: dictionary with key: mock_model, value: folder containing generations
        experiment_dir: output folder for answer_key.csv
        num_comparisons: number of comparisons to make for each model
    
    create answer_key.csv in experiment_dir
    """
    print('--- Creating experiment directory ----')

    bach_dir = '../chorales/cleaned_bach_chorales'
    
    pairs_dict = defaultdict(dict)
    pair_id = 0
    bach_csv = open(f'{bach_dir}/bach_grades.csv', 'r')
    bach_df = pd.read_csv(bach_csv)
    for model, mock_dir in mock_dirs.items():
        bach_ids = random.sample([int(f[:-4]) for f in os.listdir(bach_dir) if is_midi(f)], num_comparisons)
        mock_ids = random.sample([int(f[:-4]) for f in os.listdir(mock_dir) if is_midi(f)], num_comparisons)
        mock_csv = open(f'{mock_dir}/mock_grades.csv', 'r')
        mock_df = pd.read_csv(mock_csv)
        for i, (bach_id, mock_id) in enumerate(zip(bach_ids, mock_ids)):
            pairs_dict[pair_id]['bach_id'] = bach_id
            pairs_dict[pair_id]['mock_id'] = mock_id
            pairs_dict[pair_id]['mock_model'] = model
            pairs_dict[pair_id]['bach_grade'] = bach_df.at[bach_id, 'grade']
            pairs_dict[pair_id]['mock_grade'] = mock_df.at[mock_id, 'grade']
            label = random.choice(['a', 'b'])
            pairs_dict[pair_id]['bach_is'] = label
            pair_id += 1

    print('Writing answer key')
    with open(f'answer_key.csv', 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['pair_id', 'bach_id', 'mock_id', 'mock_model','bach_grade', 'mock_grade', 'bach_is'])
        for i, pair in pairs_dict.items():
            writer.writerow([i, pair['bach_id'], pair['mock_id'], pair['mock_model'], pair['bach_grade'], pair['mock_grade'], pair['bach_is']])


def grade_pairs():
    """
    evaluate grading function at the paired discrimination task
    """
    print('---- Evaluating grading function ----')
    picks = []

    with open(f'answer_key.csv', 'r') as fin:
        answer_key = pandas.read_csv(fin)
        bach_grades = answer_key['bach_grade']
        mock_grades = answer_key['mock_grade']
        paired_grades = zip(bach_grades, mock_grades)

    correct_ct = np.sum([1 for b, m in paired_grades if b > m])

    return correct_ct/len(paired_grades)


if __name__ == '__main__':
    main()
