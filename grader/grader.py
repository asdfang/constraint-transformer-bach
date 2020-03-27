"""
grade a chorale compared to a dataset of ground-truth chorales
"""

from collections import defaultdict
from music21 import converter

from grader.get_feature_scores import *


score_methods_dict = {'error': get_error_score,
                      'parallel_error': get_parallel_error_score,
                      'note': get_note_score,
                      'rhythm': get_rhythm_score,
                      'harmonic_quality': get_harmonic_quality_score,
                      'directed_interval': get_directed_interval_score,
                      'undirected_interval': get_undirected_interval_score,
                      'S_directed_interval': get_S_directed_interval_score,
                      'A_directed_interval': get_A_directed_interval_score,
                      'T_directed_interval': get_T_directed_interval_score,
                      'B_directed_interval': get_B_directed_interval_score,
                    #   'S_undirected_interval': get_S_undirected_interval_score,
                    #   'A_undirected_interval': get_A_undirected_interval_score,
                    #   'T_undirected_interval': get_T_undirected_interval_score,
                    #   'B_undirected_interval': get_B_undirected_interval_score,
                      }


# def score_chorale(chorale, dataset, weights=None):
#     """
#     Arguments
#         chorale: music21.stream.Stream
#         dataset: a ChoraleDataset object
#         scores: a dict of weights for each score component

#     return overall score and component scores
#     """

#     assert dataset.distributions is not None

#     if not weights:
#         weights = {feature: 1 for feature in score_methods_dict}

#     scores = defaultdict(int)
#     score = 0

#     for feature in weights:
#         feature_score = score_methods_dict[feature](chorale, dataset)
#         feature_weight = weights[feature]

#         scores[feature] = feature_weight*feature_score
#         score += scores[feature]

#     return 2-np.min([score, 2]), scores


def grade_chorale(chorale, dataset):
    """
    Arguments
        chorale: music21.stream.Stream
        dataset: a ChoraleDataset object
        scores: a dict of weights for each score component

    return overall score and component scores
    """
    chorale_vector = get_feature_vector(chorale, dataset)
    gm = dataset.gaussian
    grade = gm.score([chorale_vector])
    return grade, chorale_vector


def get_feature_vector(chorale, dataset):
    assert dataset.distributions is not None

    chorale_vector = []
    for feature in score_methods_dict:
        feature_score = score_methods_dict[feature](chorale, dataset)
        chorale_vector.append(feature_score)
    
    return chorale_vector




