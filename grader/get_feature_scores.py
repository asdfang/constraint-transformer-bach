"""
score functions for a chorale with reference to a dataset
"""
from scipy.stats import wasserstein_distance

from grader.distribution_helpers import *
from grader.compute_chorale_histograms import *


def get_error_score(chorale, dataset):
    num_notes = len(chorale.flat.notes)
    chorale_histogram = get_error_histogram(chorale, dataset.voice_ranges)

    num_errors = sum(chorale_histogram.values())
    chorale_distribution = histogram_to_distribution(chorale_histogram)
    dataset_distribution = dataset.distributions['error_distribution']
    error_note_ratio = num_errors / num_notes

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution)) * (error_note_ratio / dataset.error_note_ratio)

def get_parallel_error_score(chorale, dataset):
    num_notes = len(chorale.flat.notes)
    chorale_histogram = get_parallel_error_histogram(chorale)

    num_parallel_errors = sum(chorale_histogram.values())
    chorale_distribution = histogram_to_distribution(chorale_histogram)
    dataset_distribution = dataset.distributions['parallel_error_distribution']
    parallel_error_note_ratio = num_parallel_errors / num_notes

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution)) * (parallel_error_note_ratio / dataset.parallel_error_note_ratio)


def get_note_score(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    Returns Wasserstein distance between normalized chorale note distribution and normalized dataset note distribution
    """
    key = chorale.analyze('key')
    chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
    dataset_distribution = dataset.distributions[f'{key.mode}_note_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_rhythm_score(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object

    Returns Wasserstein distance between normalized chorale rhythm distribution and normalized dataset rhythm distribution
    """
    chorale_distribution = histogram_to_distribution(get_rhythm_histogram(chorale))
    dataset_distribution = dataset.distributions['rhythm_distribution']

    chorale_list, dataset_list = distribution_to_list(chorale_distribution, dataset_distribution)

    return wasserstein_distance(chorale_list, dataset_list)


def get_harmonic_quality_score(chorale, dataset):
    key = chorale.analyze('key')
    chorale_distribution = histogram_to_distribution(get_harmonic_quality_histogram(chorale))
    dataset_distribution = dataset.distributions[f'{key.mode}_harmonic_quality_distribution']

    chorale_list, dataset_list = distribution_to_list(chorale_distribution, dataset_distribution)
    
    return wasserstein_distance(chorale_list, dataset_list)


def get_directed_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=True))
    dataset_distribution = dataset.distributions[f'{key.mode}_directed_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_undirected_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=False))
    dataset_distribution = dataset.distributions[f'{key.mode}_undirected_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_S_directed_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=True)
    
    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_S_directed_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_S_undirected_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=False)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_S_undirected_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_A_directed_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=True)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_A_directed_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_A_undirected_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=False)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_A_undirected_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_T_directed_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=True)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_T_directed_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_T_undirected_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=False)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_T_undirected_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_B_directed_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=True)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_B_directed_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))


def get_B_undirected_interval_score(chorale, dataset):
    key = chorale.analyze('key')
    voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=False)

    chorale_distribution = histogram_to_distribution(voice_ih)
    dataset_distribution = dataset.distributions[f'{key.mode}_B_undirected_interval_distribution']

    return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))