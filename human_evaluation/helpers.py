import music21

NUM_PAIRS_PER_TASK = 10
NUM_PRETEST_QUESTIONS = 5
MODELS = ['base', 'bach', 'aug-gen']
NUM_TASKS = 2000

MODEL_TO_IDX = {
    'bach': 0,
    'aug-gen': 1,
    'base': 2,
    'bad': 3,
}

IDX_TO_COMPARISON = {
    'a': ['bach', 'aug-gen'],
    'b': ['bach', 'base'],
    'c': ['aug-gen', 'base'],
    'd': ['bach', 'bad'],
}

CSV = {
    'tasks': 'human_evaluation/data/tasks.csv',
    'a_pairs': 'human_evaluation/data/bach_aug-gen_pairs.csv',
    'b_pairs': 'human_evaluation/data/bach_base_pairs.csv',
    'c_pairs': 'human_evaluation/data/aug-gen_base_pairs.csv',
    'd_pairs': 'human_evaluation/data/bach_bad_pairs.csv',
    'completed_tasks': 'human_evaluation/data/completed_tasks.csv',
    'completed_a_pairs': 'human_evaluation/data/completed_aug-gen_bach_pairs.csv',
    'completed_b_pairs': 'human_evaluation/data/completed_base_bach_pairs.csv',
    'completed_c_pairs': 'human_evaluation/data/completed_aug-gen_base_pairs.csv',
    'completed_d_pairs': 'human_evaluation/data/completed_bad_bach_pairs.csv',
}

AUDIO_DIRS = {
    # 'bach': f'http://cortex.cs.northwestern.edu:8200/static/audio/{MODEL_TO_IDX["bach"]}/',
    # 'aug-gen': f'http://cortex.cs.northwestern.edu:8200/static/audio/{MODEL_TO_IDX["aug-gen"]}/',
    # 'base': f'http://cortex.cs.northwestern.edu:8200/static/audio/{MODEL_TO_IDX["base"]}/',
    # 'bad': f'http://cortex.cs.northwestern.edu:8200/static/audio/{MODEL_TO_IDX["bad"]}/',
    # 'pretest': 'http://cortex.cs.northwestern.edu:8200/static/audio/pretest/',
    'bach': f'/upload/audio/{MODEL_TO_IDX["bach"]}/',
    'aug-gen': f'/upload/audio/{MODEL_TO_IDX["aug-gen"]}/',
    'base': f'/upload/audio/{MODEL_TO_IDX["base"]}/',
    'bad': f'/upload/audio/{MODEL_TO_IDX["bad"]}/',
    'pretest': '/upload/audio/pretest/',
}

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
# import os
# files = list(filter((lambda f: f[-4:] == '.mid'), os.listdir()))
# files.sort(key=lambda f: int(f.split('.')[0]))
# for f in files:
#     clean_midi(f)
#     print(f)