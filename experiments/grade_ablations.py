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

from experiments.grade_distribution_plots import plot_histograms
from experiments.generate_and_grade import grade_folder
from Grader.grader import Grader, FEATURES
from transformer_bach.utils import ensure_dir, parse_xml

# ----- modify these -----
# features in your grading function
# original:
# directory of Bach chorales and mock chorales
BACH_DIR = 'chorales/bach_chorales' # uncleaned
MOCK_DIR = 'models/base_05-10_07:20/xml_mocks'
LOOPY_MOCK_DIR = 'chorales/loopy_mocks'
# directory of ablation experiments
ABLATIONS_DIR = 'experiments/new_wasserstein_midi_ablations'
# name of csv file that will be created in the directories of chorales
# GRADES_CSV = 'tmp.csv'
BACH_GRADES_CSV = 'bach_grades.csv'
MOCK_GRADES_CSV = 'mock_grades.csv'
LOOPY_MOCK_GRADES_CSV = 'loopy_mock_grades.csv'
# ------------------------

# ------ ablations -------
ALL_ABLATIONS = {
    'reg_pe_no_oe' : FEATURES,
}

# ------------------------

for ablation in ALL_ABLATIONS:
    print(f'----- current ablation: {ablation} -----')
    print('Parsing XML to create bach_dataset')
    bach_dataset = [parse_xml(f'chorales/bach_chorales/{i}.xml') for i in tqdm(range(351))]
    
    features = ALL_ABLATIONS[ablation]
    PICKLE_DIR = ablation
    ensure_dir(f'{ABLATIONS_DIR}/{PICKLE_DIR}/')

    with open(f'{ABLATIONS_DIR}/{PICKLE_DIR}/features.txt', 'w') as readme:
        readme.write('Features:\n')
        readme.write('\n'.join(features))

    grader = Grader(
        features=features,
        iterator=bach_dataset,
        pickle_dir=PICKLE_DIR,
    )

    grade_folder(chorale_dir=BACH_DIR, grader=grader, grades_csv=BACH_GRADES_CSV)
    grade_folder(chorale_dir=MOCK_DIR, grader=grader, grades_csv=MOCK_GRADES_CSV)
    # grade_folder(chorale_dir=LOOPY_MOCK_DIR, grader=grader, grades_csv=LOOPY_MOCK_GRADES_CSV)

    os.rename(f'{BACH_DIR}/{BACH_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{BACH_GRADES_CSV}')
    os.rename(f'{MOCK_DIR}/{MOCK_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{MOCK_GRADES_CSV}')
    # os.rename(f'{LOOPY_MOCK_DIR}/{LOOPY_MOCK_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{LOOPY_MOCK_GRADES_CSV}')

    mock_grades = pd.read_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/{MOCK_GRADES_CSV}').dropna()['grade']
    mock_grades = list(mock_grades)
    bach_grades = pd.read_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/{BACH_GRADES_CSV}')['grade']
    bach_grades = list(bach_grades)

    data_dict = {'Bach': bach_grades, 'Mock': mock_grades}
    print(f'Wasserstein distance: {wasserstein_distance(bach_grades, mock_grades)}')
    bach_grades = np.array(bach_grades)
    mock_grades = np.array(mock_grades)
    print(f'KS test: {ks_2samp(bach_grades, mock_grades)}')
    
    plot_histograms(
        data_dict=data_dict, 
        plt_title='Grade distributions', 
        plt_dir=f'{ABLATIONS_DIR}/{PICKLE_DIR}/',
        threshold=-250,
    )
    

