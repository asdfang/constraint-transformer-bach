"""
class for grading a given chorale compared to Bach chorales
"""

from collections import defaultdict
from music21 import converter
from scipy.stats import wasserstein_distance

from Grader.distribution_helpers import *
from Grader.compute_chorale_histograms import *


FEATURES = ['error', 'parallel_error', 'note', 'rhythm', 'harmonic_quality', 
            'directed_interval', 'S_directed_interval', 'A_directed_interval', 
            'T_directed_interval', 'B_directed_interval']
            

class Grader:         
    def __init__(self, dataset, features):
        """
        :param dataset: the dataset to compare against
        :param features: the features to use
        """
        self.dataset = dataset
        self.features = features
    
    def grade_chorale(self, chorale):
        try:
            chorale_vector = self.get_feature_vector(chorale)
        except:     # sometimes the grading function fails
            return float('-inf'), []
        
        gm = self.dataset.gaussian
        grade = gm.score([chorale_vector])
        return grade, chorale_vector
    
    
    def get_feature_vector(self, chorale):
        assert self.dataset.distributions is not None
        chorale_vector = []
        for feature in self.features:
            method_name = f'get_{feature}_grade'
            feature_grade = getattr(self, method_name)(chorale)
            chorale_vector.append(feature_grade)
        
        return chorale_vector
    
    def get_error_grade(self, chorale):
        num_notes = len(chorale.flat.notes)
        chorale_histogram = get_error_histogram(chorale, self.dataset.voice_ranges)

        num_errors = sum(chorale_histogram.values())
        chorale_distribution = histogram_to_distribution(chorale_histogram)
        dataset_distribution = self.dataset.distributions['error_distribution']
        error_note_ratio = num_errors / num_notes

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution)) * (error_note_ratio / self.dataset.error_note_ratio)

    def get_parallel_error_grade(self, chorale):
        num_notes = len(chorale.flat.notes)
        chorale_histogram = get_parallel_error_histogram(chorale)

        num_parallel_errors = sum(chorale_histogram.values())
        chorale_distribution = histogram_to_distribution(chorale_histogram)
        dataset_distribution = self.dataset.distributions['parallel_error_distribution']
        parallel_error_note_ratio = num_parallel_errors / num_notes

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution)) * (parallel_error_note_ratio / self.dataset.parallel_error_note_ratio)

    def get_note_grade(self, chorale):
        """
        Arguments
            chorale: music21.stream.Stream

        Returns Wasserstein distance between normalized chorale note distribution and normalized dataset note distribution
        """
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
        dataset_distribution = self.dataset.distributions[f'{key.mode}_note_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_rhythm_grade(self, chorale):
        """
        Arguments
            chorale: music21.stream.Stream

        Returns Wasserstein distance between normalized chorale rhythm distribution and normalized dataset rhythm distribution
        """
        chorale_distribution = histogram_to_distribution(get_rhythm_histogram(chorale))
        dataset_distribution = self.dataset.distributions['rhythm_distribution']

        chorale_list, dataset_list = distribution_to_list(chorale_distribution, dataset_distribution)

        return wasserstein_distance(chorale_list, dataset_list)

    def get_harmonic_quality_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_harmonic_quality_histogram(chorale))
        dataset_distribution = self.dataset.distributions[f'{key.mode}_harmonic_quality_distribution']

        chorale_list, dataset_list = distribution_to_list(chorale_distribution, dataset_distribution)
        
        return wasserstein_distance(chorale_list, dataset_list)

    def get_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=True))
        dataset_distribution = self.dataset.distributions[f'{key.mode}_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=False))
        dataset_distribution = self.dataset.distributions[f'{key.mode}_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_S_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=True)
        
        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_S_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_S_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_S_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_A_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_A_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_A_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_A_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_T_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_T_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_T_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_T_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_B_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_B_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_B_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.dataset.distributions[f'{key.mode}_B_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))
