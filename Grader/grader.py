"""
class for grading a given chorale compared to Bach chorales
"""

from collections import defaultdict
from music21 import converter
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import pickle
import os
from collections import Counter
import numpy as np

from transformer_bach.utils import ensure_dir
from Grader.distribution_helpers import histogram_to_distribution, distributions_to_wasserstein_inputs
from Grader.compute_chorale_histograms import get_note_histogram, get_harmonic_quality_histogram, get_SATB_interval_histogram, get_interval_histogram, \
    get_harmonic_quality_histogram, get_rhythm_histogram, get_error_histogram, get_parallel_error_histogram, get_repeated_sequence_histogram, \
    get_repeated_sequence_histogram


FEATURES = ['note', 'rhythm', 'parallel_error', 'harmonic_quality',
            'S_directed_interval', 'A_directed_interval', 'T_directed_interval', 'B_directed_interval', 
            'repeated_sequence']

POSSIBLE_FEATURES = ['note', 'rhythm', 'parallel_error', 'error', 'harmonic_quality',
                     'directed_interval', 'S_directed_interval', 'A_directed_interval', 'T_directed_interval', 'B_directed_interval',
                     'undirected_interval', 'S_undirected_interval', 'A_undirected_interval', 'T_undirected_interval', 'B_undirected_interval'
                     'repeated_sequence_1', 'repeated_sequence_2', 'S_repeated_sequence', 'A_repeated_sequence', 'T_repeated_sequence', 
                     'B_repeated_sequence', 'self_similarity']

class Grader:         
    def __init__(self, features, iterator=None, pickle_dir=None):
        """
        Arguments:
            features: list of features to use in grading function
            iterator: an iterator of chorales to grade against (most likely Bach)
            pickle_dir: a folder containing pickles, or the folder to save pickles in if it does not exist
        """
        self.features = features
        self.iterator = iterator
        self.distributions = None
        self.error_note_ratio = None
        self.parallel_error_note_ratio = None
        self.gaussian = None
        self.voice_ranges = None
        if pickle_dir is None:
            pickle_dir = '.'
        self.pickle_dir = f'{os.path.expanduser("~")}/transformer-bach/Grader/pickles/{pickle_dir}'
        self.load_or_pickle_distributions()
    
    def grade_chorale(self, chorale):
        try:
            chorale_vector = self.get_feature_vector(chorale)
        except:     # catch grading function failures
            return float('inf'), []
        
        grade = np.sum(chorale_vector)
        return grade, chorale_vector
    
    def get_feature_vector(self, chorale):
        assert self.distributions is not None
        chorale_vector = []
        for feature in self.features:
            method_name = f'get_{feature}_grade'
            feature_grade = getattr(self, method_name)(chorale)
            chorale_vector.append(feature_grade)
        
        return chorale_vector

    def load_or_pickle_distributions(self):
        voice_ranges_file = 'voice_ranges.txt'
        distributions_file = 'bach_distributions.txt'
        error_note_ratio_file = 'error_note_ratio.txt'
        parallel_error_note_ratio_file = 'parallel_error_note_ratio.txt'
        dist_files = [distributions_file, error_note_ratio_file, parallel_error_note_ratio_file]
        gaussian_file = 'gaussian.txt'

        # create pickle_dir if it does not exists
        if not os.path.exists(self.pickle_dir):
            ensure_dir(self.pickle_dir)
            with open(f'{self.pickle_dir}/features.txt', 'w') as readme:
                readme.write('Features:\n')
                readme.write('\n'.join(self.features))
        # compute or load voice ranges
        if os.path.exists(f'{self.pickle_dir}/{voice_ranges_file}'):
            with open(f'{self.pickle_dir}/{voice_ranges_file}', 'rb') as fin:
                self.voice_ranges = pickle.load(fin)
        else:
            self.compute_voice_ranges(self.iterator, 4)
            with open(f'{self.pickle_dir}/{voice_ranges_file}', 'wb') as fo:
                pickle.dump(self.voice_ranges, fo)
        # compute or load distributions
        if np.all([os.path.exists(f'{self.pickle_dir}/{f}') for f in dist_files]):
            with open(f'{self.pickle_dir}/{distributions_file}', 'rb') as fin:
                self.distributions = pickle.load(fin)
            with open(f'{self.pickle_dir}/{error_note_ratio_file}', 'rb') as fin:
                self.error_note_ratio = pickle.load(fin)
            with open(f'{self.pickle_dir}/{parallel_error_note_ratio_file}', 'rb') as fin:
                self.parallel_error_note_ratio = pickle.load(fin)
        else:
            self.calculate_distributions()
            with open(f'{self.pickle_dir}/{distributions_file}', 'wb') as fo:
                pickle.dump(self.distributions, fo)
            with open(f'{self.pickle_dir}/{error_note_ratio_file}', 'wb') as fo:
                pickle.dump(self.error_note_ratio, fo)
            with open(f'{self.pickle_dir}/{parallel_error_note_ratio_file}', 'wb') as fo:
                pickle.dump(self.parallel_error_note_ratio, fo)
        # compute or load Gaussian
        if os.path.exists(f'{self.pickle_dir}/{gaussian_file}'):
            with open(f'{self.pickle_dir}/{gaussian_file}', 'rb') as fin:
                self.gaussian = pickle.load(fin)
        else:
            self.fit_gaussian()            
            with open(f'{self.pickle_dir}/{gaussian_file}', 'wb') as fo:
                pickle.dump(self.gaussian, fo)
    
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

        self.voice_ranges = [tuple(e) for e in voice_ranges]

    def calculate_distributions(self):
        print('Calculating ground-truth distributions over Bach chorales')

        # initialize all histograms
        major_nh, minor_nh = Counter(), Counter()            # notes (for chorales in major)
                                                             # notes (for chorales in minor)
        rh = Counter()                  # rhythm
        
        major_hqh, minor_hqh = Counter(), Counter()           # harmonic quality
        
        major_directed_ih, minor_directed_ih = Counter(), Counter()         # directed intervals for whole chorale
        major_S_directed_ih, minor_S_directed_ih = Counter(), Counter()     # ... for soprano
        major_A_directed_ih, minor_A_directed_ih = Counter(), Counter()     # ... for alto
        major_T_directed_ih, minor_T_directed_ih = Counter(), Counter()     # ... for tenor
        major_B_directed_ih, minor_B_directed_ih = Counter(), Counter()     # ... for bass
        
        major_undirected_ih, minor_undirected_ih = Counter(), Counter()       # undirected intervals for whole chorale
        major_S_undirected_ih, minor_S_undirected_ih = Counter(), Counter()   # ... for soprano
        major_A_undirected_ih, minor_A_undirected_ih = Counter(), Counter()   # ... for alto
        major_T_undirected_ih, minor_T_undirected_ih = Counter(), Counter()   # ... for tenor
        major_B_undirected_ih, minor_B_undirected_ih = Counter(), Counter()   # ... for bass
        eh, peh = Counter(), Counter()                  # errors (not including parallelism)
                                                        # parallel errors (octaves and fifths)
        sh_1, sh_2, S_sh, A_sh, T_sh, B_sh = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()     # repeated sequences
        ssh = Counter()           # self-similarity
        ssh.update({b: 0 for b in BINS[:-1]})
        num_notes = 0                   # number of notes

        # calculate histograms for all Bach chorales (for relevant features)
        for chorale in tqdm(self.iterator):
            key = chorale.analyze('key')
            if 'note' in self.features:
                chorale_nh = get_note_histogram(chorale, key)
                if key.mode == 'major':
                    major_nh += chorale_nh
                else:
                    minor_nh += chorale_nh

            if 'harmonic_quality' in self.features:
                if key.mode == 'major':
                    major_hqh += get_harmonic_quality_histogram(chorale)
                else:
                    minor_hqh += get_harmonic_quality_histogram(chorale)
            
            if 'directed_interval' in self.features:
                if key.mode == 'major':
                    major_directed_ih += get_interval_histogram(chorale, directed=True)
                else:
                    minor_directed_ih += get_interval_histogram(chorale, directed=True)
            
            if 'undirected_interval' in self.features:
                if key.mode == 'major':
                    major_undirected_ih += get_interval_histogram(chorale, directed=False)
                else:
                    minor_undirected_ih += get_interval_histogram(chorale, directed=False)
            
            if 'S_directed_interval' in self.features:
                if key.mode == 'major':
                    major_S_directed_ih += get_SATB_interval_histogram(chorale, voice=0, directed=True)
                else:
                    minor_S_directed_ih += get_SATB_interval_histogram(chorale, voice=0, directed=True)
            
            if 'A_directed_interval' in self.features:
                if key.mode == 'major':
                    major_A_directed_ih += get_SATB_interval_histogram(chorale, voice=1, directed=True)
                else:
                    minor_A_directed_ih += get_SATB_interval_histogram(chorale, voice=1, directed=True)
            
            if 'T_directed_interval' in self.features:
                if key.mode == 'major':
                    major_T_directed_ih += get_SATB_interval_histogram(chorale, voice=2, directed=True)
                else:
                    minor_T_directed_ih += get_SATB_interval_histogram(chorale, voice=2, directed=True)
            
            if 'B_directed_interval' in self.features:
                if key.mode == 'major':
                    major_B_directed_ih += get_SATB_interval_histogram(chorale, voice=3, directed=True)
                else:
                    minor_B_directed_ih += get_SATB_interval_histogram(chorale, voice=3, directed=True)
            
            if 'S_undirected_interval' in self.features:
                if key.mode == 'major':
                    major_S_undirected_ih += get_SATB_interval_histogram(chorale, voice=0, directed=False)
                else:
                    minor_S_undirected_ih += get_SATB_interval_histogram(chorale, voice=0, directed=False)
            
            if 'A_undirected_interval' in self.features:
                if key.mode == 'major':
                    major_A_undirected_ih += get_SATB_interval_histogram(chorale, voice=1, directed=False)
                else:
                    minor_A_undirected_ih += get_SATB_interval_histogram(chorale, voice=1, directed=False)
            
            if 'T_undirected_interval' in self.features:
                if key.mode == 'major':
                    major_T_undirected_ih += get_SATB_interval_histogram(chorale, voice=2, directed=False)
                else:
                    minor_T_undirected_ih += get_SATB_interval_histogram(chorale, voice=2, directed=False)
            
            if 'B_undirected_interval' in self.features:
                if key.mode == 'major':
                    major_B_undirected_ih += get_SATB_interval_histogram(chorale, voice=3, directed=False)
                else:
                    minor_B_undirected_ih += get_SATB_interval_histogram(chorale, voice=3, directed=False)

            if 'rhythm' in self.features:
                rh += get_rhythm_histogram(chorale)
            
            if 'error' in self.features:
                eh += get_error_histogram(chorale, self.voice_ranges)
            
            if 'parallel_error' in self.features:
                peh += get_parallel_error_histogram(chorale)
            
            if 'repeated_sequence_1' in self.features:
                sh_1 += get_repeated_sequence_histogram_1(chorale)

            if 'repeated_sequence_2' in self.features:
                sh_2 += get_repeated_sequence_histogram_2(chorale)
            
            if 'S_repeated_sequence' in self.features:
                sh += get_repeated_sequence_histogram(chorale, voice=0)
            
            if 'A_repeated_sequence' in self.features:
                sh += get_repeated_sequence_histogram(chorale, voice=1)
            
            if 'T_repeated_sequence' in self.features:
                sh += get_repeated_sequence_histogram(chorale, voice=2)
            
            if 'B_repeated_sequence' in self.features:
                sh += get_repeated_sequence_histogram(chorale, voice=3)

            if 'self_similarity' in self.features:
                ssh.update(get_self_similarity_histogram(chorale))
            
            # number of notes
            num_notes += len(chorale.flat.notes)

        print(f'len(ssh.keys()): {len(ssh.keys())}')

        # proportion of errors to notes
        error_note_ratio = sum(eh.values()) / num_notes

        # proportion of parallel errors to notes
        parallel_error_note_ratio = sum(peh.values()) / num_notes

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
                         'parallel_error_distribution': peh,
                         'repeated_sequence_1_distribution': sh_1,
                         'repeated_sequence_2_distribution': sh_2,
                         'S_repeated_sequence_distribution': S_sh,
                         'A_repeated_sequence_distribution': A_sh,
                         'T_repeated_sequence_distribution': T_sh,
                         'B_repeated_sequence_distribution': B_sh,
                         'self_similarity_distribution': ssh}

        # normalize each histogram by the sum of dictionary values, converting to distribution
        for dist in distributions:
            distributions[dist] = histogram_to_distribution(distributions[dist])

        self.error_note_ratio = error_note_ratio
        self.parallel_error_note_ratio = parallel_error_note_ratio
        self.distributions = distributions

    def fit_gaussian(self):
        print('Calculating Gaussian')
        chorale_vectors = []
        for chorale in tqdm(self.iterator):
            chorale_vector = self.get_feature_vector(chorale)
            chorale_vectors.append(chorale_vector)
        
        gm = GaussianMixture()
        self.gaussian = gm.fit(chorale_vectors)
    
    def get_error_grade(self, chorale):
        num_notes = len(chorale.flat.notes)
        chorale_histogram = get_error_histogram(chorale, self.voice_ranges)

        num_errors = sum(chorale_histogram.values())
        # if num_errors == 0, the coefficient will be 0, so we can sidestep the calculation of wasserstein 
        if num_errors == 0:
            return 0
        chorale_distribution = histogram_to_distribution(chorale_histogram)
        dataset_distribution = self.distributions['error_distribution']
        error_note_ratio = num_errors / num_notes

        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)
        return wasserstein_distance(u, v, u_weights, v_weights) * (error_note_ratio / self.error_note_ratio)

    def get_parallel_error_grade(self, chorale):
        num_notes = len(chorale.flat.notes)
        chorale_histogram = get_parallel_error_histogram(chorale)

        num_parallel_errors = sum(chorale_histogram.values())
        # if num_parallel_errors == 0, the coefficient will be 0, so we can sidestep the calculation of wasserstein 
        if num_parallel_errors == 0:
            return 0
        
        chorale_distribution = histogram_to_distribution(chorale_histogram)
        dataset_distribution = self.distributions['parallel_error_distribution']
        parallel_error_note_ratio = num_parallel_errors / num_notes
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)
        
        return wasserstein_distance(u, v, u_weights, v_weights) * (parallel_error_note_ratio / self.parallel_error_note_ratio)

    def get_note_grade(self, chorale):
        """
        Arguments
            chorale: music21.stream.Stream

        Returns Wasserstein distance between normalized chorale note distribution and normalized dataset note distribution
        """
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_note_histogram(chorale, key))
        dataset_distribution = self.distributions[f'{key.mode}_note_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_rhythm_grade(self, chorale):
        """
        Arguments
            chorale: music21.stream.Stream

        Returns Wasserstein distance between normalized chorale rhythm distribution and normalized dataset rhythm distribution
        """
        chorale_distribution = histogram_to_distribution(get_rhythm_histogram(chorale))
        dataset_distribution = self.distributions['rhythm_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_harmonic_quality_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_harmonic_quality_histogram(chorale))
        dataset_distribution = self.distributions[f'{key.mode}_harmonic_quality_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=True))
        dataset_distribution = self.distributions[f'{key.mode}_directed_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        chorale_distribution = histogram_to_distribution(get_interval_histogram(chorale, directed=False))
        dataset_distribution = self.distributions[f'{key.mode}_undirected_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_S_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=True)
        
        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_S_directed_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_S_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=0, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_S_undirected_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_A_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_A_directed_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_A_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=1, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_A_undirected_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_T_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_T_directed_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_T_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=2, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_T_undirected_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_B_directed_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=True)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_B_directed_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)

    def get_B_undirected_interval_grade(self, chorale):
        key = chorale.analyze('key')
        voice_ih = get_SATB_interval_histogram(chorale, voice=3, directed=False)

        chorale_distribution = histogram_to_distribution(voice_ih)
        dataset_distribution = self.distributions[f'{key.mode}_B_undirected_interval_distribution']
        u, v, u_weights, v_weights = distributions_to_wasserstein_inputs(chorale_distribution, dataset_distribution)

        return wasserstein_distance(u, v, u_weights, v_weights)
    
    def get_repeated_sequence_grade(self, chorale):
        sh = get_repeated_sequence_histogram(chorale)
        chorale_distribution = histogram_to_distribution(sh)
        dataset_distribution = self.distributions['repeated_sequence_2_distribution']
        max_seq = np.max(list(chorale_distribution.keys()) + list(dataset_distribution.keys())) # in ticks
        chorale_list, dataset_list = [0] * (max_seq + 1), [0] * (max_seq + 1)

        # populate chorale_list at the indices corresponding to keys in chorale_distribution
        for seq_len in chorale_distribution:
            chorale_list[seq_len] = chorale_distribution[seq_len]
        
        for seq_len in dataset_distribution:
            dataset_list[seq_len] = dataset_distribution[seq_len]
        
        if np.sum(chorale_list) == 0:
            chorale_list[0] = 1
        
        vals = [i for i in range(len(chorale_list))]
        return wasserstein_distance(vals, vals, chorale_list, dataset_list)
