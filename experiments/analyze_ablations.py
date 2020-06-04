import sys
sys.path[0] += '/../'

import os
import music21
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

from experiments.grade_distribution_plots import plot_histograms
from experiments.generate_and_grade import grade_folder
from Grader.grader import Grader, FEATURES
from transformer_bach.utils import ensure_dir

# ----- modify these -----
# features in your grading function
# original:
# the directory for the pickles/ablation (a nickname for this set of features)
PICKLE_DIR = 'original'
# directory of Bach chorales and mock chorales
BACH_DIR = 'chorales/bach_chorales' # uncleaned
MOCK_DIR = 'models/base_05-10_07:20/xml_mocks'
LOOPY_MOCK_DIR = 'chorales/loopy_mocks'
AUG_GEN_DIR = 'models/aug-gen_05-09_06:09/unconstrained_mocks_s1'
# directory of ablation experiments
ABLATIONS_DIR = 'experiments/ablations'
# name of csv file that will be created in the directories of chorales
# GRADES_CSV = 'tmp.csv'
BACH_GRADES_CSV = 'bach_grades.csv'
MOCK_GRADES_CSV = 'mock_grades.csv'
LOOPY_MOCK_GRADES_CSV = 'loopy_mock_grades.csv'
AUG_GEN_GRADES_CSV = 'aug-gen_grades.csv'
# ------------------------

# ------ ablations -------
original = FEATURES
note = ['note']
rhythm = ['rhythm']
parallel_error = ['parallel_error']
note_rhythm = ['note', 'rhythm']
note_rhythm_pe = ['note', 'rhythm', 'parallel_error']
SATB_intervals = ['S_directed_interval', 'A_directed_interval', 'T_directed_interval', 'B_directed_interval']
all_intervals = ['directed_interval', 'S_directed_interval', 'A_directed_interval', 'T_directed_interval', 'B_directed_interval']
no_oe_oi_hq = [f for f in FEATURES if f not in ['error', 'directed_interval', 'harmonic_quality']]

note_sequence = ['note', 'repeated_sequence']
note_SATB_sequence = ['note', 'S_repeated_sequence', 'A_repeated_sequence', 'T_repeated_sequence', 'B_repeated_sequence']
self_similarity = ['self_similarity']
note_self_similarity = ['note', 'self_similarity']
sequence = ['repeated_sequence']
SATB_sequence = ['S_repeated_sequence', 'A_repeated_sequence', 'T_repeated_sequence', 'B_repeated_sequence']
no_oi = [f for f in FEATURES if f != 'directed_interval']

ALL_ABLATIONS = {
    # 'note_sequences_1': note_sequence,
    # 'note_sequences_2': note_sequence,
    # 'note_SATB_sequences': note_SATB_sequence,
    # 'self_similarity': self_similarity,
    # 'SATB_sequence': SATB_sequence,
    # 'note_self_similarity': note_self_similarity
    # 'original_seq1': FEATURES + ['repeated_sequence'],
    # 'sequence_1': ['sequence_1'],
    # 'sequence_2': ['repeated_sequence_2'],
    # 'self_similarity': ['self_similarity'],
    'reg_pe_no_oe': FEATURES
}


for ablation in ALL_ABLATIONS:
    print(f'----- current ablation: {ablation} -----')
    features = ALL_ABLATIONS[ablation]
    weights = [1] * len(features)
    
    PICKLE_DIR = ablation
    ensure_dir(f'{ABLATIONS_DIR}/{PICKLE_DIR}/')

    grader = Grader(
        features=features,
        iterator=None,
    )

    # grade_folder(chorale_dir=BACH_DIR, grader=grader, grades_csv=BACH_GRADES_CSV)
    # grade_folder(chorale_dir=MOCK_DIR, grader=grader, grades_csv=MOCK_GRADES_CSV)
    # grade_folder(chorale_dir=LOOPY_MOCK_DIR, grader=grader, grades_csv=LOOPY_MOCK_GRADES_CSV)
    # grade_folder(chorale_dir=AUG_GEN_DIR, grader=grader, grades_csv=AUG_GEN_GRADES_CSV)

    # os.rename(f'{BACH_DIR}/{BACH_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{BACH_GRADES_CSV}')
    # os.rename(f'{MOCK_DIR}/{MOCK_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{MOCK_GRADES_CSV}')
    # os.rename(f'{LOOPY_MOCK_DIR}/{LOOPY_MOCK_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{LOOPY_MOCK_GRADES_CSV}')
    # os.rename(f'{AUG_GEN_DIR}/{AUG_GEN_GRADES_CSV}', f'{ABLATIONS_DIR}/{PICKLE_DIR}/{AUG_GEN_GRADES_CSV}')

    bach_df = pd.read_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/{BACH_GRADES_CSV}')
    mock_df = pd.read_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/{MOCK_GRADES_CSV}').dropna()
    bach_vectors = bach_df[features].values.tolist()
    mock_vectors = mock_df[features].values.tolist()

    bach_grades = []
    for bach_vector in bach_vectors:
        bach_grades.append(np.dot(weights, bach_vector))

    mock_grades = []
    for mock_vector in mock_vectors:
        mock_grades.append(np.dot(weights, mock_vector))
    
    bach_df['grade'] = bach_grades
    bach_df.to_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/sum_bach_grades.csv', index=False)
    mock_df['grade'] = mock_grades
    mock_df.to_csv(f'{ABLATIONS_DIR}/{PICKLE_DIR}/sum_mock_grades.csv', index=False)
    
    data_dict = {'Bach': bach_grades, 'Mock': mock_grades}

    plot_histograms(
        data_dict=data_dict, 
        plt_title='Grade distributions', 
        plt_dir=f'{ABLATIONS_DIR}/{PICKLE_DIR}/',
        plt_name=f'sum_grade_dist',
    )
    
    # print(f'Wasserstein distance: {wasserstein_distance(bach_grades, mock_grades)}')
    bach_grades = np.array(bach_grades)
    mock_grades = np.array(mock_grades)
    print(f'KS test: {ks_2samp(bach_grades, mock_grades)}')


    # for feature in features:
    #     bach_grades = bach_df[feature]
    #     mock_grades = mock_df[feature]
    
    #     print(f'{feature} wasserstein distance: {wasserstein_distance(bach_grades, mock_grades)}')
    #     data_dict = {'Bach': bach_grades, 'Mock': mock_grades}

    #     plot_histograms(
    #         data_dict=data_dict, 
    #         plt_title='Grade distributions', 
    #         plt_dir=f'{ABLATIONS_DIR}/{PICKLE_DIR}/',
    #         plt_name=f'{feature}_dist',
    #         threshold=None,
    #     )

    # visualize Bach Gaussian with PCA reduction
    # if len(features) > 1:
    #     pca = PCA(n_components=2)
    #     pca.fit(bach_vectors)
    #     bach_pca = pca.transform(bach_vectors)
    #     loopy_mock_pca = pca.transform(loopy_mock_vectors)
    #     plt.figure()
    #     plt.style.use('seaborn-whitegrid')
    #     plt.scatter(bach_pca[:,0], bach_pca[:,1], alpha=0.2, color='steelblue', label='Bach')
    #     plt.scatter(loopy_mock_pca[:,0], loopy_mock_pca[:,1], alpha=0.2, color='salmon', label='Loopy mock')
    #     plt.legend()
    #     plt.savefig(f'{ABLATIONS_DIR}/{PICKLE_DIR}/bach_gaussian_pca.png')

