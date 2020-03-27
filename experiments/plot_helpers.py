"""
Call things from main()
"""

import sys
sys.path.insert(0, '../')

import music21
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pickle

from grader.distribution_helpers import *
from grader.compute_chorale_histograms import *
from scipy.stats import wasserstein_distance
    
'''
Saves a plot, and returns a score

Put in main() of this file
ex: get_chorale_note_distribution_and_score('../generations/4/c0.mid', '../plots/bach_dist/genex_note_distribution.png', 'Generated Example Note Distribution',
                                distributions['major_note_distribution'], distributions['minor_note_distribution'],
                                major_note_order, minor_note_order)
'''
def get_chorale_note_distribution_and_score(chorale_filename, plot_filename, plot_title,
                                                major_note_distribution, minor_note_distribution,
                                                major_note_order, minor_note_order):
    """
    Arguments
        chorale_filename: String that holds path to a music21.stream.Stream
        plot_filename: String that holds where user wants the plot to be saved
        major_note_distribution: Counter holding note distribution of major keys
        minor_note_distribution: Counter holding note distribution of minor keys
        major_note_order: tuple holding order of notes to display on plot for major keys
        minor_note_order: tuple holding order of notes to display on plot for minor keys

    Saves a plot, and returns a score
    """
    chorale = music21.converter.parse(chorale_filename)
    key = chorale.analyze('key')
    chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
    note_distribution = major_note_distribution if key.mode == 'major' else minor_note_distribution
    notes = major_note_order if key.mode == 'major' else minor_note_order

    chorale_list = distribution_to_list(chorale_distribution, note_distribution)[0]
    y_pos = np.arange(len(notes))
    y_vals = chorale_list

    plt.figure()
    plt.bar(y_pos, y_vals, align='center')
    plt.xticks(y_pos, notes)
    plt.xlabel('Scale Degree')
    plt.ylabel('Proportion')
    plt.title(plot_title)

    plt.savefig(plot_filename)
    return wasserstein_distance(*distribution_to_list(chorale_distribution, note_distribution))

'''
Example to plot major note distribution (Put in main())
note_distribution_plot('../plots/bach_dist/major_note_distribution.png', 'Major Note Distribution', distributions['major_note_distribution'], major_note_order)
'''
def note_distribution_plot(plot_filename, plot_title, note_distribution, note_order):
    """
    Arguments
        plot_filename: String that holds where user wants the plot to be saved
        plot_title: what plot title should be
        note_distribution: Counter holding note distribution
        note_order: Tuple holding order o notes to display on plot
    """
    y_pos = np.arange(len(note_order))
    y_vals = [x[1] for x in note_distribution.most_common()]

    plt.figure()
    plt.bar(y_pos, y_vals, align='center')
    plt.xticks(y_pos, note_order)
    plt.xlabel('Scale Degree')
    plt.ylabel('Proportion')

    # might want to modify this title
    plt.title(plot_title)
    plt.savefig(plot_filename)

    return

def main():
    distributions_file = '../grader/bach_distributions.txt'
    error_note_ratio_file = '../grader/error_note_ratio.txt'
    parallel_error_note_ratio_file = '../grader/parallel_error_note_ratio.txt'

    major_note_order = ('5', '1', '3', '2', '6', '4', '7', '4♯', '7♭', 'Rest', '1♯', '5♯', '3♭', '2♯', '6♭', '2♭')
    minor_note_order = ('5', '1', '3', '4', '2', '7', '6', '7♯', '6♯', '3♯', 'Rest', '4♯', '2♭', '5♭', '1♯', '1♭')

    with open(distributions_file, 'rb') as fin:
        distributions = pickle.load(fin)
    with open(error_note_ratio_file, 'rb') as fin:
        error_note_ratio = pickle.load(fin)
    with open(parallel_error_note_ratio_file, 'rb') as fin:
        parallel_error_note_ratio = pickle.load(fin)

    # # bug me if this doesn't work
    note_distribution_plot('../plots/bach_dist/major_note_distribution.png', 'Major Note Distribution',
                           distributions['major_note_distribution'], major_note_order)
    note_distribution_plot('../plots/bach_dist/minor_note_distribution.png', 'Minor Note Distribution',
                           distributions['minor_note_distribution'], minor_note_order)
    get_chorale_note_distribution_and_score('../generations/4/c0_temp.mid', '../plots/bach_dist/gen4_c0_note_distribution.png',
                                            'Note Distribution of Generated Chorale',
                                            distributions['major_note_distribution'], distributions['minor_note_distribution'],
                                            major_note_order, minor_note_order)


if __name__ == '__main__':
    main()