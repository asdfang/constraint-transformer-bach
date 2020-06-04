from collections import Counter
import csv, os
import matplotlib.pyplot as plt, numpy as np
from transformer_bach.utils import *


def histogram_to_distribution(counter):
    """
    Arguments
        counter: collections.Counter

    Returns a normalized collections.Counter, converting a histogram to a probability distribution
    """
    total = sum(counter.values(), 0.0)
    if total == 0:
        return counter
    
    for key in counter:
        counter[key] /= total

    return counter


def distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution):
    """
    Arguments
        chorale_distribution: Counter – this is the distribution that needs to be compared against the baseline
        dataset_distribution: Counter – this is the baseline distribution to compare chorale_distribution against

    Returns a 4-tuple that preserves the order for scipy's Wasserstein.
    The 3rd and 4th elements are: two lists of the same length, with each element being the probability from the distributions in the order of dataset_distribution's keys.
    """
    # vals and weights as in scipy's Wasserstein
    ordered_keys = [x[0] for x in dataset_distribution.most_common()]
    ordered_dataset_distribution_weights = [x[1] for x in dataset_distribution.most_common()]
    ordered_chorale_distribution_weights = []
    chorale_distribution_extras = Counter()

    for okey in ordered_keys:  # make chorale_distribution the same order as dataset_distribution
        ordered_chorale_distribution_weights.append(chorale_distribution[okey])

    for key in chorale_distribution:  # get leftover vals from chorale_distribution in ascending order
        if key not in dataset_distribution:
            chorale_distribution_extras[key] += chorale_distribution[key]
    chorale_distribution_extras = [x[1] for x in chorale_distribution_extras.most_common()]
    chorale_distribution_extras.reverse()

    # add the leftovers
    ordered_dataset_distribution_weights.extend([0] * len(chorale_distribution_extras))
    ordered_chorale_distribution_weights.extend(chorale_distribution_extras)

    if len(dataset_distribution) < len(chorale_distribution):
        assert ordered_dataset_distribution_weights[-1] == 0
    assert len(ordered_dataset_distribution_weights) == len(ordered_chorale_distribution_weights)

    # make inputs for scipy's wasserstein
    chorale_vals, distribution_vals = [i for i in range(len(ordered_chorale_distribution_weights))], [j for j in range(len(ordered_dataset_distribution_weights))]

    return chorale_vals, distribution_vals, ordered_chorale_distribution_weights, ordered_dataset_distribution_weights