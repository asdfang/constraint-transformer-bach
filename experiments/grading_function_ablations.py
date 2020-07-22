import sys
sys.path[0] += '/../'

import os
import music21
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from tqdm import tqdm

from experiments.plot_grade_distributions import plot_histograms
from experiments.generate_and_grade import grade_folder
from Grader.grader import Grader, FEATURES
from transformer_bach.utils import ensure_dir, parse_xml

# ----- modify these -----
# directory of Bach chorales and mock chorales
BACH_DIR = 'chorales/bach_chorales'
MOCK_DIR = 'models/base_06-02_06:55/351_mocks'
# directory of ablation experiments
ABLATIONS_DIR = 'experiments/ablations'
# name of csv file that will be created in the directories of chorales
BACH_GRADES_CSV = 'bach_grades.csv'
MOCK_GRADES_CSV = 'mock_grades.csv'

ALL_ABLATIONS = {
    'original': FEATURES,
}
# ------------------------

for ablation in ALL_ABLATIONS:
    print(f'----- current ablation: {ablation} -----')
    print('Parsing XML to create bach_dataset')
    bach_dataset = [parse_xml(f'chorales/bach_chorales/{i}.xml') for i in tqdm(range(351))]
    
    features = ALL_ABLATIONS[ablation]
    PICKLE_DIR = ablation
    ensure_dir(f'{ABLATIONS_DIR}/{PICKLE_DIR}/')

    # record the features in this ablation in features.txt
    with open(f'{ABLATIONS_DIR}/{PICKLE_DIR}/features.txt', 'w') as readme:
        readme.write('Features:\n')
        readme.write('\n'.join(features))

    # create Grader object with the features in this ablation
    grader = Grader(
        features=features,
        iterator=bach_dataset,
        pickle_dir=PICKLE_DIR,
    )

    # grade Bach and Mock chorales and move them to ABLATIONS_DIR
    grade_folder(chorale_dir=BACH_DIR, grader=grader, grades_csv=BACH_GRADES_CSV)
    grade_folder(chorale_dir=MOCK_DIR, grader=grader, grades_csv=MOCK_GRADES_CSV)
    os.rename(f'{BACH_DIR}/{BACH_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{BACH_GRADES_CSV}')
    os.rename(f'{MOCK_DIR}/{MOCK_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{MOCK_GRADES_CSV}')

    # read grades from CSV
    mock_df = pd.read_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/{MOCK_GRADES_CSV}').dropna()
    bach_df = pd.read_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/{BACH_GRADES_CSV}')
    data_dict = {'Bach': list(bach_df['grade']), 'Mock': list(mock_df['grade'])}
    
    # plot grade distributions
    plot_histograms(
        data_dict=data_dict, 
        plt_title='Grade distributions', 
        plt_dir=f'{ABLATIONS_DIR}/{PICKLE_DIR}/',
    )

    # print some statistics
    print(f'Wasserstein distance: {wasserstein_distance(bach_df["grade"], mock_df["grade"])}')
    print(f'KS test: {ks_2samp(np.array(bach_df["grade"]), np.array(mock_df["grade"]))}')

    print('--- Overall grade ---')
    print(f'Bach mean: {np.round(np.mean(bach_df["grade"]),3)}')
    print(f'Bach std: {np.round(np.std(bach_df["grade"]),3)}')
    print(f'Mock mean: {np.round(np.mean(mock_df["grade"]), 3)}')
    print(f'Mock std: {np.round(np.std(mock_df["grade"]),3)}')

    for feature in features:
        print(f'--- {feature} ---')
        print(f'Bach mean: {np.round(np.mean(bach_df[feature]), 3)}')
        print(f'Bach std: {np.round(np.std(bach_df[feature]), 3)}')
        print(f'Mock mean: {np.round(np.mean(mock_df[feature]), 3)}')
        print(f'Mock std: {np.round(np.std(mock_df[feature]), 3)}')

