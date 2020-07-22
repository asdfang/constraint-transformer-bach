import numpy as np
import music21
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from transformer_bach.utils import ensure_dir
from transformer_bach.constraint_helpers import score_to_hold_representation


def get_correlative_matrix(voice):
    """
    Arguments
        voice: a list of notes
    
    return a 2D numpy array representing the correlative matrix for the given voice
    """
    n = len(voice)
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i+1, n):
            # do not start sequences with a hold symbol
            if (i == 0 or A[i-1, j-1] == 0) and np.all(voice[i] == '__'):
                continue
            # otherwise, check for tick equality
            if np.all(voice[i] == voice[j]):
                if i == 0:
                    A[i, j] = 1
                else:
                    A[i, j] = A[i-1, j-1] + 1
    return A

def get_candidate_set(A, voice):
    """
    Arguments
        A: the correlative matrix (2D numpy array) for the given voice
        voice: a list of notes
    
    return candidate_set, a dictionary with value as sequence, key as (rep_count, sub_count)
    """
    n = len(voice)
    candidate_set = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] == 0:
                continue

            # we reach the end of a substring of length 1
            if A[i,j] == 1 and (j+1 == n or A[i+1, j+1] == 0):
                seq = sequence_to_tuple(voice[i])
                if is_valid_sequence(seq):
                    candidate_set[seq] += 1
            
            # we are in a substring of length 1
            elif A[i,j] == 1 and A[i+1, j+1] != 0:
                seq = sequence_to_tuple(voice[i])
                if is_valid_sequence(seq):
                    candidate_set[seq] += 1
            
            # we reach the end of a substring of length longer than 1
            elif A[i,j] > 1 and (j+1 == n or A[i+1, j+1] == 0):
                ln = A[i,j]
                # for every substring ending with the last note 
                for k in range(ln - 1):
                    subseq = sequence_to_tuple(voice[i-k:i+1])
                    if is_valid_sequence(subseq):
                        candidate_set[subseq] += 1
                seq = sequence_to_tuple(voice[i-(ln-1):i+1])
                if is_valid_sequence(seq):
                    candidate_set[seq] += 1
            
            # we are in a substring of length longer than 1
            elif A[i,j] > 1 and A[i+1, j+1] != 0:
                ln = A[i,j]
                for k in range(A[i,j]):
                    subseq = sequence_to_tuple(voice[i-k:i+1])
                    if is_valid_sequence(subseq):
                        candidate_set[subseq] += 1

    a_time = time.time()
    candidate_set = pop_subsequences(candidate_set)
    b_time = time.time()
    candidate_set = calculate_true_frequencies(candidate_set)
    return candidate_set

def candidate_set_to_repeated_sequence_histogram(candidate_set):
    """
    convert candidate set to sequence_table, a Counter with key as sequence length, 
    value as count of occurrences of sequence with that length
    """
    repeated_sequence_histogram = Counter()
    for seq in candidate_set:
        ln = len(seq)
        freq = candidate_set[seq]
        repeated_sequence_histogram[ln] += freq
    
    return repeated_sequence_histogram

def calculate_true_frequencies(candidate_set):
    for seq in candidate_set:
        rep_count = candidate_set[seq]
        f = int((1 + np.sqrt(1 + 8 * rep_count))/2)
        candidate_set[seq] = f
    
    return candidate_set

def pop_subsequences(candidate_set):
    """
    remove redundant subsequences from the candidate set (python Dictionary)
    candidate_set has value as sequence, key as rep_count

    e.g. candidate_set = {('A4', '__', 'G4', '__'): 4, ('B5','G4'): 6, ('B5','G4','F4'): 6} – it will remove ('B5','G4')
    """ 
    ckeys = sorted(candidate_set.keys(), key=len)[::-1]
    last = len(ckeys)
    i = 0

    while i < last:
        j = i+1
        while j < last:
            seq1 = ckeys[i]
            seq2 = ckeys[j]
            if len(seq2) < len(seq1) and is_subset(seq1, seq2) and candidate_set[seq1] == candidate_set[seq2]:
                candidate_set.pop(seq2)
                del ckeys[j]
                last -= 1
            else:
                j += 1
        i += 1

    return candidate_set

"""
plotting
"""

def plot_repeated_sequence_histogram(repeated_sequence_counter, plt_dir, plt_name):
    """
    Arguments
        repeated_sequence_histogram: sequence_table representing the chorale
        plt_dir: output directory
        plt_name: file name
    
    plot the distribution/histogram for a chorale (can be a "mean" chorale)
    """
    plt.figure()
    ensure_dir(plt_dir)
    plt.bar(repeated_sequence_counter.keys(), repeated_sequence_counter.values())
    if len(list(repeated_sequence_counter.keys())) == 0:
        max_seq = 0
    else:
        max_seq = np.max(list(repeated_sequence_counter.keys()))
    xticks = range(1,(max_seq+4)//4,1)
    plt.xticks(ticks=[tick*4 for tick in xticks], labels=xticks)
    plt.xlim([0, max_seq+1])
    plt.xlabel('Sequence length (beats)')
    if np.sum(repeated_sequence_counter.values()) in [1-1e-8, 1+1e-8]:
        plt.ylabel('Proportion of repeated sequences')
    else:
        plt.ylabel('Count of repeated sequences')
    plt.savefig(f'{plt_dir}/{plt_name}.png')
    plt.close()

"""
HELPERS WHEN EACH INDIVIDUAL VOICE IS CONSIDERED SEPARATELY
"""

def sequence_to_tuple(seq):
    """
    convert a list of notes to a tuple of notes
    """
    if isinstance(seq, list):
        return tuple(seq)
    else:
        return tuple([seq])

def is_valid_sequence(sequence):
    """
    return true if the given sequence of notes begins with a note or a rest 
    AND contains at least two notes
    """
    num_notes = len([tick for tick in sequence if tick != '__'])
    return sequence[0] != '__' and num_notes > 1


def is_subset(seq1, seq2):
    """
    Arguments
        seq1, seq2: a sequence (tuple) of notes

    check if seq2 is a strict subsequence of seq1
    """
    s1 = list(seq1)
    s2 = list(seq2)
    assert len(s2) < len(s1)
    ln = len(seq2)
    return any((all(s2[j] == s1[i + j] for j in range(ln)) for i in range(len(s1) - ln + 1)))
