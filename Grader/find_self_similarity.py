import sys
sys.path[0] += '/../'

import music21
import numpy as np

BINS = np.arange(0, 4.10, 0.05)

def get_self_similarity(chorale):
    """
    a chorale is a list of lists with 4 elements, e.g. [[88,70,45,35], [89,60,55,45]]

    return the offset vector (np.ndarray), where each value corresponds to the self-similarity at that offset
    """
    n = len(chorale)
    a = [0] * n

    # for every possible offset
    for j in range(n):
        a[j] = np.sum([timestamp_match(chorale[i], chorale[i+j]) for i in range(n - j)])/(n - j)
    
    return a


def timestamp_match(tick1, tick2):
    """
    tick1, tick2: two lists of 4 elements representing two different ticks in the chorale
    """
    return np.sum([1 for a, b in zip(tick1, tick2) if a == b])