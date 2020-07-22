# move to another file? chorale_dataset.py?
import sys
sys.path[0] += '/../'
import numpy as np
import music21
from transformer_bach.DatasetManager.helpers import SLUR_SYMBOL, standard_name


def _has_one_part(stream):
    """
    Arguments
        stream: music21.stream.Stream

    returns True if input has only one part.
    """
    return stream.hasPartLikeStreams() == False or len(stream.parts) == 1


def single_part_to_hold_representation(part):
    """
    Arguments
        part: music21.stream.Stream – must contain only one line/part/voice

    returns a list that represents the part in hold representation, with notes 
    """
    assert _has_one_part(part)
    hold_representation = []
    part = part.flat.notesAndRests
    for note in part:
        hold_representation.append(standard_name(note))
        num_holds = int(note.duration.quarterLength / 0.25 - 1.0)
        hold_representation.extend([SLUR_SYMBOL] * num_holds)

    return hold_representation

def single_part_to_midi_tick_representation(part):
    """
    Arguments
        part: music21.stream.Stream – must contain only one line/part/voice

    returns a list that represents the part in midi numbers at every sixteenth tick that it is sounding
    """
    assert _has_one_part(part)
    midi_tick_representation = []
    part = part.flat.notesAndRests
    for note_or_rest in part:
        symbol = note_or_rest.pitch.midi if isinstance(note_or_rest, music21.note.Note) else 'rest'
        midi_tick_representation.extend([symbol] * int(note_or_rest.duration.quarterLength / 0.25))
    
    return midi_tick_representation


def score_to_hold_representation_for_voice(score, voice=0):
    """
    Arguments
        score: music21.stream.Score – probably a Bach chorale.
        voice: which voice index to convert to hold representation; soprano by default.

    returns a list that represents the specified part given by the voice index in hold representation
    """
    return single_part_to_hold_representation(score.parts[voice])

def score_to_midi_tick_representation_for_voice(score, voice=0):
    """
    Arguments
        score: music21.stream.Score – probably a Bach chorale.
        voice: which voice index to convert to hold representation; soprano by default.

    returns a list that represents the specified part given by the voice index in midi tick representation
    """
    return single_part_to_midi_tick_representation(score.parts[voice])


def score_to_hold_representation(score):
    """
    Arguments
        score: music21.stream.Score – probably a Bach chorale.
    
    returns a 2-D, first axis represents which voice (0 is soprano), second axis has hold representation
    """
    full_hold_score = []

    for i in range(4):
        full_hold_score.append(score_to_hold_representation_for_voice(score, voice=i))
    
    return full_hold_score

def score_to_midi_tick_representation(score):
    """
    Arguments
        score: music21.stream.Score – probably a four-part chorale.

    returns a 2-D list, first axis represents which tick, second axis represents which voice (0 is soprano).
    """
    full_score = []

    for i in range(4):
        full_score.append(score_to_midi_tick_representation_for_voice(score, voice=i))
    np_score = np.array(full_score)

    return np_score.T.tolist()