import music21

NUM_PAIRS_PER_TASK = 10
NULL_SYMBOL = '_'
# MOCK_MODELS = ['base', 'a-gen', 'a-bach']
MOCK_MODELS = ['base']
NUM_TASKS = 1000


def is_midi(fname):
    return True if fname[-4:] == '.mid' else False


def clean_midi(fname):
    """
    Given a file name, overwrites the same file with a cleaned midi.
    """
    score = music21.converter.parse(fname)
    cleaned_score = clean_score(score)
    cleaned_score.write('mid', fname)


def clean_score(score):
    """
    Arguments:
        score: music21.stream.Score

    Returns a simply cleaned version of the score. For a visual test. Does not handle pick-ups or time signatures other than 4/4.
        Also, adds a metronome marking of 72.
    """
    assert len(score.parts) == 4
    score = score.parts
    cleaned_score = music21.stream.Score()
    cleaned_score.append(music21.tempo.MetronomeMark(number=72))    # hard-coded
    
    for part_id, part in enumerate(score):
        cleaned_score.append(part.flat.notesAndRests.flat)
    for part_id, part in enumerate(cleaned_score):
        part.id = part_id
        
    return cleaned_score

'''
to clean a directory of .midi to a new directory,
copy this file to that directory,
modify what you're writing to (cleaned_score.write('...')
to a new directory

and uncomment the below lines, modifying what you need to sort the original directory
'''
# files = list(filter((lambda f: f[-4:] == '.mid'), os.listdir()))
# files.sort(key=lambda f: int(f.split('.')[0]))
# for f in files:
#     clean_midi(f)
#     print(f)