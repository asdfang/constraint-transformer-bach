"""
class for grading a given chorale compared to Bach chorales
"""

from collections import defaultdict
from music21 import converter
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import pickle
import os
from collections import Counter

from Grader.distribution_helpers import *
from Grader.compute_chorale_histograms import *


FEATURES = ['error', 'parallel_error', 'note', 'rhythm', 'harmonic_quality', 
            'directed_interval', 'S_directed_interval', 'A_directed_interval', 
            'T_directed_interval', 'B_directed_interval']
            

class Grader:         
    def __init__(self, features, iterator):
        """
        :param features: a list of features to use in grading
        :param iterator: an iterator of chorales to grade against
        """
        self.features = features
        self.iterator = iterator
        self.distributions = None
        self.error_note_ratio = None
        self.parallel_error_note_ratio = None
        self.gaussian = None
        self.voice_ranges = self.compute_voice_ranges(self.iterator, 4)
        self.load_or_pickle_distributions()
    
    def compute_voice_ranges(self, iterator, num_voices):
        voice_ranges = []
        for _ in range(num_voices):
            voice_ranges.append([float('inf'), float('-inf')])
        
        for chorale in iterator:
            assert len(chorale.parts) == num_voices
            for voice_index in range(num_voices):
                for note in chorale.parts[voice_index].flat.notes:
                    midi = note.pitch.midi
                    voice_ranges[voice_index][0] = min(midi, voice_ranges[voice_index][0])
                    voice_ranges[voice_index][1] = max(midi, voice_ranges[voice_index][1])

        return [tuple(e) for e in voice_ranges]

    def load_or_pickle_distributions(self):
        pickles_dir = f'{os.path.expanduser("~")}/transformer-bach/Grader/pickles/'
        distributions_file = os.path.join(pickles_dir, 'bach_distributions.txt')
        error_note_ratio_file =  os.path.join(pickles_dir, 'error_note_ratio.txt')
        parallel_error_note_ratio_file =  os.path.join(pickles_dir, 'parallel_error_note_ratio.txt')
        gaussian_file = os.path.join(pickles_dir, 'gaussian.txt')

        if os.path.exists(distributions_file) and os.path.exists(error_note_ratio_file) and os.path.exists(
                parallel_error_note_ratio_file) and os.path.exists(gaussian_file):
            print('Loading Bach chorale distributions')
            with open(distributions_file, 'rb') as fin:
                self.distributions = pickle.load(fin)
            with open(error_note_ratio_file, 'rb') as fin:
                self.error_note_ratio = pickle.load(fin)
            with open(parallel_error_note_ratio_file, 'rb') as fin:
                self.parallel_error_note_ratio = pickle.load(fin)
            with open(gaussian_file, 'rb') as fin:
                self.gaussian = pickle.load(fin)
        else:
            self.calculate_distributions()
            with open(distributions_file, 'wb') as fo:
                pickle.dump(self.distributions, fo)
            with open(error_note_ratio_file, 'wb') as fo:
                pickle.dump(self.error_note_ratio, fo)
            with open(parallel_error_note_ratio_file, 'wb') as fo:
                pickle.dump(self.parallel_error_note_ratio, fo)
            with open(gaussian_file, 'wb') as fo:
                pickle.dump(self.gaussian, fo)
    
    def calculate_distributions(self):
        print('Calculating ground-truth distributions over Bach chorales')

        major_nh = Counter()            # notes (for chorales in major)
        minor_nh = Counter()            # notes (for chorales in minor)
        rh = Counter()                  # rhythm
        major_hqh = Counter()           # harmonic quality
        minor_hqh = Counter()
        major_directed_ih = Counter()         # directed intervals for whole chorale
        minor_directed_ih = Counter()
        major_S_directed_ih = Counter()       # ... for soprano
        minor_S_directed_ih = Counter()
        major_A_directed_ih = Counter()       # ... for alto
        minor_A_directed_ih = Counter()
        major_T_directed_ih = Counter()       # ... for tenor
        minor_T_directed_ih = Counter()
        major_B_directed_ih = Counter()       # ... for bass
        minor_B_directed_ih = Counter()
        major_undirected_ih = Counter()       # undirected intervals for whole chorale
        minor_undirected_ih = Counter()
        major_S_undirected_ih = Counter()     # ... for soprano
        minor_S_undirected_ih = Counter()
        major_A_undirected_ih = Counter()     # ... for alto
        minor_A_undirected_ih = Counter()
        major_T_undirected_ih = Counter()     # ... for tenor
        minor_T_undirected_ih = Counter()
        major_B_undirected_ih = Counter()     # ... for bass
        minor_B_undirected_ih = Counter()
        eh = Counter()                  # errors (not including parallelism)
        peh = Counter()                 # parallel errors (octaves and fifths)
        num_notes = 0                   # number of notes

        print('Calculating feature distributions for Bach chorales')
        for chorale in tqdm(self.iterator):
            key = chorale.analyze('key')
            chorale_nh = get_note_histogram(chorale, key)
            if key.mode == 'major':
                # note histogram
                major_nh += chorale_nh
                # harmonic quality histogram
                major_hqh += get_harmonic_quality_histogram(chorale)
                # interval histograms
                major_directed_ih += get_interval_histogram(chorale, directed=True)
                major_S_directed_ih += get_SATB_interval_histogram(chorale, voice=0, directed=True)
                major_A_directed_ih += get_SATB_interval_histogram(chorale, voice=1, directed=True)
                major_T_directed_ih += get_SATB_interval_histogram(chorale, voice=2, directed=True)
                major_B_directed_ih += get_SATB_interval_histogram(chorale, voice=3, directed=True)
                major_undirected_ih += get_interval_histogram(chorale, directed=False)
                major_S_undirected_ih += get_SATB_interval_histogram(chorale, voice=0, directed=False)
                major_A_undirected_ih += get_SATB_interval_histogram(chorale, voice=1, directed=False)
                major_T_undirected_ih += get_SATB_interval_histogram(chorale, voice=2, directed=False)
                major_B_undirected_ih += get_SATB_interval_histogram(chorale, voice=3, directed=False)
            else:
                # note histogram
                minor_nh += chorale_nh
                # harmonic quality histogram
                minor_hqh += get_harmonic_quality_histogram(chorale)
                # interval histograms
                minor_directed_ih += get_interval_histogram(chorale, directed=True)
                minor_S_directed_ih += get_SATB_interval_histogram(chorale, voice=0, directed=True)
                minor_A_directed_ih += get_SATB_interval_histogram(chorale, voice=1, directed=True)
                minor_T_directed_ih += get_SATB_interval_histogram(chorale, voice=2, directed=True)
                minor_B_directed_ih += get_SATB_interval_histogram(chorale, voice=3, directed=True)
                minor_undirected_ih += get_interval_histogram(chorale, directed=False)
                minor_S_undirected_ih += get_SATB_interval_histogram(chorale, voice=0, directed=False)
                minor_A_undirected_ih += get_SATB_interval_histogram(chorale, voice=1, directed=False)
                minor_T_undirected_ih += get_SATB_interval_histogram(chorale, voice=2, directed=False)
                minor_B_undirected_ih += get_SATB_interval_histogram(chorale, voice=3, directed=False)

            # rhythm histogram
            rh += get_rhythm_histogram(chorale)
            
            # error histogram
            eh += get_error_histogram(chorale, self.voice_ranges)
            # parallel error histogram
            peh += get_parallel_error_histogram(chorale)
            # number of notes
            num_notes += len(chorale.flat.notes)

        # proportion of errors to notes
        error_note_ratio = sum(eh.values()) / num_notes

        # proportion of parallel errors to notes
        parallel_error_note_ratio = sum(peh.values()) / num_notes

        # convert histograms to distributions by normalizing
        distributions = {'major_note_distribution': major_nh,
                         'minor_note_distribution': minor_nh,
                         'rhythm_distribution': rh,
                         'major_harmonic_quality_distribution': major_hqh,
                         'minor_harmonic_quality_distribution': minor_hqh,
                         'major_directed_interval_distribution': major_directed_ih,
                         'minor_directed_interval_distribution': minor_directed_ih,
                         'major_S_directed_interval_distribution': major_S_directed_ih,
                         'minor_S_directed_interval_distribution': minor_S_directed_ih,
                         'major_A_directed_interval_distribution': major_A_directed_ih,
                         'minor_A_directed_interval_distribution': minor_A_directed_ih,
                         'major_T_directed_interval_distribution': major_T_directed_ih,
                         'minor_T_directed_interval_distribution': minor_T_directed_ih,
                         'major_B_directed_interval_distribution': major_B_directed_ih,
                         'minor_B_directed_interval_distribution': minor_B_directed_ih,
                         'major_undirected_interval_distribution': major_undirected_ih,
                         'minor_undirected_interval_distribution': minor_undirected_ih,
                         'major_S_undirected_interval_distribution': major_S_undirected_ih,
                         'minor_S_undirected_interval_distribution': minor_S_undirected_ih,
                         'major_A_undirected_interval_distribution': major_A_undirected_ih,
                         'minor_A_undirected_interval_distribution': minor_A_undirected_ih,
                         'major_T_undirected_interval_distribution': major_T_undirected_ih,
                         'minor_T_undirected_interval_distribution': minor_T_undirected_ih,
                         'major_B_undirected_interval_distribution': major_B_undirected_ih,
                         'minor_B_undirected_interval_distribution': minor_B_undirected_ih,
                         'error_distribution': eh,
                         'parallel_error_distribution': peh}

        for dist in distributions:
            distributions[dist] = histogram_to_distribution(distributions[dist])

        self.error_note_ratio = error_note_ratio
        self.parallel_error_note_ratio = parallel_error_note_ratio
        self.distributions = distributions

        chorale_vectors = []
        print('Calculating Gaussian')
        for chorale in tqdm(self.iterator):
            chorale_vector = self.get_feature_vector(chorale)
            chorale_vectors.append(chorale_vector)

        gm = GaussianMixture()
        self.gaussian = gm.fit(chorale_vectors)

    def grade_chorale(self, chorale):
        try:
            chorale_vector = self.get_feature_vector(chorale)
        except:     # sometimes the grading function fails
            return float('-inf'), []
        
        gm = self.gaussian
        grade = gm.score([chorale_vector])
        return grade, chorale_vector
    
    
    def get_feature_vector(self, chorale):
        assert self.distributions is not None
        chorale_vector = []
        for feature in self.features:
            method_name = f'get_{feature}_grade'
            feature_grade = getattr(self, method_name)(chorale)
            chorale_vector.append(feature_grade)
        
        return chorale_vector
    
    def get_error_grade(self, chorale):
        num_notes = len(chorale.flat.notes)
        chorale_histogram = get_error_histogram(chorale, self.voice_ranges)

        num_errors = sum(chorale_histogram.values())
        chorale_distribution = histogram_to_distribution(chorale_histogram)
        dataset_distribution = self.distributions['error_distribution']
        error_note_ratio = num_errors / num_notes

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution)) * (error_note_ratio / self.error_note_ratio)

    def get_parallel_error_grade(self, chorale):
        num_notes = len(chorale.flat.notes)
        chorale_histogram = get_parallel_error_histogram(chorale)

        num_parallel_errors = sum(chorale_histogram.values())
        chorale_distribution = histogram_to_distribution(chorale_histogram)
        dataset_distribution = self.distributions['parallel_error_distribution']
        parallel_error_note_ratio = num_parallel_errors / num_notes

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution)) * (parallel_error_note_ratio / self.parallel_error_note_ratio)

    def get_note_grade(self, chorale):
        """
        Arguments
            chorale: music21.stream.Stream

        Returns Wasserstein distance between normalized chorale note distribution and normalized dataset note distribution
        """
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
        dataset_distribution = self.distributions[f'{key.mode}_note_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_rhythm_grade(self, chorale):
        """
        Arguments
            chorale: music21.stream.Stream

        Returns Wasserstein distance between normalized chorale rhythm distribution and normalized dataset rhythm distribution
        """
        chorale_distribution = histogram_to_distribution(get_rhythm_histogram(chorale))
        dataset_distribution = self.distributions['rhythm_distribution']

        chorale_list, dataset_list = distribution_to_list(chorale_distribution, dataset_distribution)

        return wasserstein_distance(chorale_list, dataset_list)

    def get_harmonic_quality_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_harmonic_quality_histogram(chorale))
        dataset_distribution = self.distributions[f'{key.mode}_harmonic_quality_distribution']

        chorale_list, dataset_list = distribution_to_list(chorale_distribution, dataset_distribution)
        
        return wasserstein_distance(chorale_list, dataset_list)

    def get_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=True))
        dataset_distribution = self.distributions[f'{key.mode}_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=False))
        dataset_distribution = self.distributions[f'{key.mode}_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_S_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=True)
        
        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_S_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_S_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_S_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_A_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_A_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_A_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_A_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_T_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_T_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_T_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_T_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_B_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_B_directed_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))

    def get_B_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_B_undirected_interval_distribution']

        return wasserstein_distance(*distribution_to_list(chorale_distribution, dataset_distribution))
