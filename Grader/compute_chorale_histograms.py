"""
functions for computing the distribution for a given chorale
"""

from collections import Counter, defaultdict
import music21
import numpy as np
import time

from Grader.voice_leading_helpers import *
from Grader.find_repeated_sequences import get_correlative_matrix, get_candidate_set, candidate_set_to_repeated_sequence_histogram
from Grader.find_self_similarity import get_self_similarity, BINS
from transformer_bach.constraint_helpers import score_to_hold_representation


def get_note_histogram(chorale, key):
    """
    Arguments
        chorale: a music21 Stream object
        key: music21.key.Key

    Returns a note histogram as a collections.Counter object for input chorale
        Counter key: (scale degree, accidental) or 'Rest'
        Counter value: count
    """
    nh = Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        if note_or_rest.isNote:
            sd = key.getScaleDegreeAndAccidentalFromPitch(note_or_rest.pitch)
            # nh[sd] += 1 # note onsets. BEWARE IF YOU UNCOMMENT!
            nh[sd] += int(note_or_rest.duration.quarterLength / 0.25) # note ticks
        else:
            # nh['Rest'] += 1 # note onsets. BEWARE IF YOU UNCOMMENT!
            nh['Rest'] += int(note_or_rest.duration.quarterLength / 0.25) # note ticks

    return nh


def get_rhythm_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns a rhythm histogram as a collections.Counter object for input chorale
        Counter key: float or 'Rest', notating a rhythm in terms of a quarter-note's length
            (i.e. 1.0: quarter-note, 0.5: eighth note, 2.0: half note, etc.)
        Counter value: count
    """
    rh = Counter()
    for note_or_rest in chorale.flat.notesAndRests:
        rh[note_or_rest.duration.quarterLength] += 1

    return rh


def get_harmonic_quality_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object

    Returns a harmonic quality (i.e. ignores root) histogram as calculated by music21.harmony.chordSymbolFigureFromChord()
    """

    hqh = Counter()       # harmony qualities
    
    chfy = chorale.chordify()
    for c in chfy.flat.getElementsByClass(music21.chord.Chord):
        csf = music21.harmony.chordSymbolFigureFromChord(c, True)
        if csf[0] == 'Chord Symbol Cannot Be Identified':
            hqh['unidentifiable'] += 1
        else:
            hqh[csf[1]] += 1

    return hqh


def get_interval_histogram(chorale, directed=True):
    """
    Arguments
        chorale: a music21 Stream object
        directed: True or False

    Returns two interval histograms, one directed and one undirected,
    as collections.Counter objects for input chorale
    """
    ih = Counter()

    for part in chorale.parts:
        intervals = part.melodicIntervals()[1:]  # all but first meaningless result
        for interval in intervals:
            ival = interval.directedName if directed else interval.name
            ih[ival] += 1

    return ih


def get_SATB_interval_histogram(chorale, voice, directed=True):
    """
    Arguments
        chorale: a music21 Stream object
        voice: 0 for soprano, 1 for alto, 2 for tenor, 3 for bass
        directed: True or False

    Returns two interval histograms, one directed and one undirected for the given voice
    as collections.Counter objects for input chorale
    """
    assert len(chorale.parts) == 4
    assert voice in range(4)

    voice_ih = Counter()

    intervals = chorale.parts[voice].melodicIntervals()[1:]
    for interval in intervals:
        ival = interval.directedName if directed else interval.name
        voice_ih[ival] += 1

    return voice_ih
    

def get_error_histogram(chorale, voice_ranges):
    """
    Arguments
        chorale: a music21 Stream object; must have 4 parts once chorale gets passed into its helper functions

    Returns a count of voice leading errors founding in keys
        Counter
    """
    possible_errors = ['H-8ve', 'H-5th', 'Overlap', 'Crossing', 'Spacing', 'Range']

    # initialize counts to 0
    error_histogram = Counter()
    error_histogram += find_voice_leading_errors(chorale) + find_voice_crossing_and_spacing_errors(
        chorale) + find_voice_range_errors(chorale, voice_ranges)
    error_histogram.update({error: 0 for error in possible_errors})  # doesn't over-write

    return error_histogram

def get_parallel_error_histogram(chorale):
    """
    Arguments
        chorale: a music21 Stream object; must have 4 parts once chorale gets passed into its helper functions

    Returns a count of voice leading errors founding in keys
        Counter
    """
    possible_errors = ['Prl-8ve', 'Prl-5th']

    # initialize counts to 0
    error_histogram = Counter()
    error_histogram += find_parallel_8ve_5th_errors(chorale)[0]
    error_histogram.update({error: 0 for error in possible_errors})  # doesn't over-write

    return error_histogram

def get_repeated_sequence_histogram(chorale):
    """
    sequences are calculated for each voice, and summer across voices
    """
    chorale = np.array(score_to_hold_representation(chorale))

    all_voices_candidate_set = defaultdict(lambda: [0,0])
    for part_idx in range(4):
        voice = list(chorale[part_idx])
        A = get_correlative_matrix(voice)
        candidate_set = get_candidate_set(A, voice)
        all_voices_candidate_set.update(candidate_set)
    repeated_sequence_histogram = candidate_set_to_repeated_sequence_histogram(all_voices_candidate_set)
    return repeated_sequence_histogram
