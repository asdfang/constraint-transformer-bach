"""
For visualizing feature distributions
"""

import sys
sys.path[0] += '/../'

import music21
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pickle

from Grader.grader import Grader, FEATURES
from Grader.distribution_helpers import *
from Grader.compute_chorale_histograms import *
from scipy.stats import wasserstein_distance
from transformer_bach.utils import parse_xml

MAJOR_NOTE_ORDER = ('5', '1', '3', '2', '6', '4', '7', '4♯', '7♭', 'Rest', '1♯', '5♯', '3♭', '2♯', '6♭', '2♭')
MINOR_NOTE_ORDER = ('5', '1', '3', '4', '2', '7', '6', '7♯', '6♯', '3♯', 'Rest', '4♯', '2♭', '5♭', '1♯', '1♭')


def get_chorale_note_distribution_and_grade(chorale_filename, plot_filename,
                                            major_note_distribution, minor_note_distribution,
                                            plot_title='Note distribution',):
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
    chorale = parse_xml(chorale_filename)
    key = chorale.analyze('key')
    print(key)
    chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
    note_distribution = major_note_distribution if key.mode == 'major' else minor_note_distribution
    notes = MAJOR_NOTE_ORDER if key.mode == 'major' else MINOR_NOTE_ORDER

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


def plot_distribution(distribution, key_order=None, xlabel=None, title=None, outfile=None, prob_threshold=None):
    """
    Arguments
        plot_filename: String that holds where user wants the plot to be saved
        plot_title: what plot title should be
        note_distribution: Counter holding note distribution
        note_order: Tuple holding order o notes to display on plot
    """
    if not key_order:
        key_order = list(x[0] for x in distribution.most_common())

    y_vals = [x[1] for x in distribution.most_common()]

    if prob_threshold:
        y_vals = [x for x in y_vals if x >= prob_threshold]
        print(key_order[len(y_vals):])      # print excluded keys

    y_pos = np.arange(len(y_vals))
    plt.figure()
    plt.bar(y_pos, y_vals, align='center')
    plt.xticks(y_pos, key_order)
    plt.xlabel(xlabel)
    plt.ylabel('Proportion')
    plt.title(title)
    plt.savefig(outfile)


def main():
    grader = Grader(
        features=FEATURES,
        iterator=None,
    )

    # get_chorale_note_distribution_and_grade(
    #     chorale_filename='models/base_05-07_22:29/generations/10/3.mid', 
    #     plot_filename='plots/note_dist.png',
    #     major_note_distribution=grader.distributions['major_note_distribution'],
    #     minor_note_distribution=grader.distributions['minor_note_distribution'],
    #     plot_title='Note Distribution for Chorale 1'
    # )

    plot_distribution(grader.distributions['major_note_distribution'],
                      title='Major Note Distribution for Bach Chorales',
                      key_order=MAJOR_NOTE_ORDER,
                      xlabel='Scale Degree',
                      outfile='plots/bach_major_note_dist')
    # plot_distribution(distributions['minor_note_distribution'],
    #                   title='Minor Note Distribution',
    #                   key_order=minor_note_order,
    #                   outfile=plt_folder + 'minor_note_distribution.png')
    # plot_distribution(distributions['rhythm_distribution'],
    #                   title='Rhythm Distribution',
    #                   outfile=plt_folder + 'rhythm_distribution.png')
    # plot_distribution(distributions['directed_interval_distribution'],
    #                   title='Directed Interval Distribution',
    #                   outfile=plt_folder + 'directed_interval_distribution.png',
    #                   prob_threshold=0.0025)
    # plot_distribution(distributions['parallel_error_distribution'],
    #                   title='Parallel Error Distribution',
    #                   outfile=plt_folder + 'parallel_error_distribution.png')
    # plot_distribution(distributions['error_distribution'],
    #                   title='Error Distribution',
    #                   outfile=plt_folder + 'error_distribution.png')


if __name__ == '__main__':
    main()
