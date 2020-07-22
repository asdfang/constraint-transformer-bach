"""
Script for grading an existing chorale in XML format, plus some example analysis from the grading function.
"""

import sys
sys.path[0] += '/../'
from Grader.grader import Grader, FEATURES
from Grader.compute_chorale_histograms import get_note_histogram
from Grader.distribution_helpers import histogram_to_distribution
from Grader.voice_leading_helpers import find_parallel_8ve_5th_errors
from transformer_bach.utils import parse_xml

# specify the chorale here (example code assumes Bach dataset has been created)
BACH_DIR = 'chorales/bach_chorales'
chorale = parse_xml(f'{BACH_DIR}/0.xml')

grader = Grader(
    # use default features (see paper)
    features=FEATURES,
    pickle_dir='original',
)

grade, feature_vector = grader.grade_chorale(chorale)
print(f'Grade: {grade}')

for f, g in zip(FEATURES, feature_vector):
    print(f'{f}: {g}')

# show the distribution of notes in the given chorale (this can be modified for other features)
key = chorale.analyze('key')
chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
print(f'Chorale distribution: {chorale_distribution}')
dataset_distribution = grader.distributions[f'{key.mode}_note_distribution']
print(f'Bach distribution: {dataset_distribution}')

# example to find and print parallel errors in chorale
error_histogram, errors = find_parallel_8ve_5th_errors(chorale)
print(errors)