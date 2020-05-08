# move to another file? chorale_dataset.py?

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

    returns a list that represents the part in hold representation
    """
    assert _has_one_part(part)
    hold_representation = []
    part = part.flat.notesAndRests
    for note in part:
        hold_representation.append(standard_name(note))
        num_holds = int(note.duration.quarterLength / 0.25 - 1.0)
        hold_representation.extend([SLUR_SYMBOL] * num_holds)

    return hold_representation


def score_to_hold_representation_for_voice(score, voice=0):
    """
    Arguments
        score: music21.stream.Score – probably a Bach chorale.
        voice: which voice index to convert to hold representation; soprano by default.

    returns a list that represents the specified part given by the voice index in hold representation
    """
    return single_part_to_hold_representation(score.parts[voice])

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