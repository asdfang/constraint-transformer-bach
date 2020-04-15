from itertools import islice
import os
import pickle
import music21
from music21 import note, harmony, expressions
from torch.utils.data import TensorDataset

# constants
SLUR_SYMBOL = '__'          # hold symbol
START_SYMBOL = 'START'
END_SYMBOL = 'END'
REST_SYMBOL = 'rest'
OUT_OF_RANGE = 'OOR'
PAD_SYMBOL = 'XX'
BEAT_SYMBOL = 'b'
DOWNBEAT_SYMBOL = 'B'
DURATION_SYMBOL = 'DUR'
YES_SYMBOL = 'YES'
NO_SYMBOL = 'NO'
UNKNOWN_SYMBOL = 'UKN'
TIME_SHIFT = 'TS'
STOP_SYMBOL = 'STOP'
MAX_VELOCITY = 128


def standard_name(note_or_rest, voice_range=None):
    """
    Convert music21 objects to str
    :param note_or_rest: one of music21.note.Note, music21.note.Rest, str, music21.harmony.ChordSymbol, or music21.expressions.TextExpression object
    
    :return: string representation of note or rest
    """
    if isinstance(note_or_rest, note.Note):
        if voice_range is not None:
            min_pitch, max_pitch = voice_range
            pitch = note_or_rest.pitch.midi
            if pitch < min_pitch or pitch > max_pitch:
                return OUT_OF_RANGE
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, note.Rest):
        return note_or_rest.name  # == 'rest' := REST_SYMBOL
    if isinstance(note_or_rest, str):
        return note_or_rest

    if isinstance(note_or_rest, harmony.ChordSymbol):
        return note_or_rest.figure
    if isinstance(note_or_rest, expressions.TextExpression):
        return note_or_rest.content


def standard_note(note_or_rest_string):
    """
    :param note_or_rest_string: string representation of note or rest

    :return: music21.note.Note object
    """
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    elif note_or_rest_string == END_SYMBOL:
        return note.Note('D~3', quarterLength=1)
    elif note_or_rest_string == START_SYMBOL:
        return note.Note('C~3', quarterLength=1)
    elif note_or_rest_string == PAD_SYMBOL:
        return note.Note('E~3', quarterLength=1)
    elif note_or_rest_string == SLUR_SYMBOL:
        return note.Rest()
    elif note_or_rest_string == OUT_OF_RANGE:
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


class ChoralesIteratorGen:
    def __init__(self, picked_chorales):
        self.picked_chorales = picked_chorales

    def __call__(self):
        it = iter(self.picked_chorales)
        return it.__iter__()


class TensorDatasetIndexed(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors[:])

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        return ret, index


def load_or_pickle_distributions(dataset):
    pickles_dir = f'{os.path.expanduser("~")}/transformer-bach/Grader/pickles/'
    distributions_file = os.path.join(pickles_dir, 'bach_distributions.txt')
    error_note_ratio_file =  os.path.join(pickles_dir, 'error_note_ratio.txt')
    parallel_error_note_ratio_file =  os.path.join(pickles_dir, 'parallel_error_note_ratio.txt')
    gaussian_file = os.path.join(pickles_dir, 'gaussian.txt')

    if os.path.exists(distributions_file) and os.path.exists(error_note_ratio_file) and os.path.exists(
            parallel_error_note_ratio_file) and os.path.exists(gaussian_file):
        print('Loading Bach chorale distributions')
        with open(distributions_file, 'rb') as fin:
            dataset.distributions = pickle.load(fin)
        with open(error_note_ratio_file, 'rb') as fin:
            dataset.error_note_ratio = pickle.load(fin)
        with open(parallel_error_note_ratio_file, 'rb') as fin:
            dataset.parallel_error_note_ratio = pickle.load(fin)
        with open(gaussian_file, 'rb') as fin:
            dataset.gaussian = pickle.load(fin)
    else:
        dataset.calculate_distributions()
        with open(distributions_file, 'wb') as fo:
            pickle.dump(dataset.distributions, fo)
        with open(error_note_ratio_file, 'wb') as fo:
            pickle.dump(dataset.error_note_ratio, fo)
        with open(parallel_error_note_ratio_file, 'wb') as fo:
            pickle.dump(dataset.parallel_error_note_ratio, fo)
        with open(gaussian_file, 'wb') as fo:
            pickle.dump(dataset.gaussian, fo)
