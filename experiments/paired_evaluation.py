import sys
sys.path.insert(0, '../')

from tqdm import tqdm
import random
import click
import os, shutil
import importlib
from datetime import datetime
import csv
import music21
import numpy as np

from transformer_bach.utils import ensure_dir, get_threshold
from transformer_bach.bach_dataloader import BachDataloaderGenerator
from transformer_bach.DatasetManager.helpers import load_or_pickle_distributions
from Grader.grader import Grader, FEATURES

BASE_DEEPBACH_GENERATIONS = '../../deepbach/generations/10/'
BASE_TRANSFORMER_GENERATIONS = '../models/bach_decoder_config_2020-03-24_20:19:31/generations/'
EVAL_DIR = '../chorales/paired_evaluation'

@click.command()
@click.option('--mock_dir', default=BASE_TRANSFORMER_GENERATIONS)
@click.option('--eval_dir', default=EVAL_DIR)
@click.option('--num_comparisons', default=10)
def main(mock_dir,
         eval_dir,
         num_comparisons):
    answer_key = get_pairs(mock_dir=mock_dir, 
                           eval_dir=eval_dir, 
                           num_comparisons=num_comparisons)
    answers = grade_pairs(experiment_dir=os.path.join('..', 'chorales', 'paired_evaluation', '4-2'))
    acc = get_accuracy(answer_key, answers)
    print(acc)


def get_pairs(mock_dir,
              eval_dir='../chorales/paired_evaluation/4-2', 
              num_comparisons=10,
              clean_scores=False):
    """
    Arguments:
        mock_dir: folder of generations to compare Bach chorales to
        eval_dir: folder for evaluation
        num_comparisons: number of generations for each model
        clean_scores: whether to clean the scores to remove visual artifacts in scores
    """
    bach_dir = '../chorales/bach_chorales/'
    assert num_comparisons < np.min([len(os.listdir(bach_dir)), len(os.listdir(mock_dir))])
    
    bach_files = random.sample([f for f in os.listdir(bach_dir)], num_comparisons)
    mock_files = random.sample([f for f in os.listdir(mock_dir)], num_comparisons)
    
    random.shuffle(bach_files)
    random.shuffle(mock_files)

    answer_key = {}
    for i in range(num_comparisons):
        labels = ['a','b']
        random.shuffle(labels)
        pair_dir = os.path.join(eval_dir, f'{i}')
        ensure_dir(pair_dir)
        bach_fname = os.path.join(pair_dir, f'{i}_{labels[0]}.mid')
        mock_fname = os.path.join(pair_dir, f'{i}_{labels[1]}.mid')
        shutil.copy(os.path.join(bach_dir, bach_files[i]), bach_fname)
        shutil.copy(os.path.join(mock_dir, mock_files[i]), mock_fname)
        clean_midi(bach_fname)
        clean_midi(mock_fname)
        answer_key[i] = f'real: {labels[0]}, fake: {labels[1]}'
    
    return answer_key


def grade_pairs(experiment_dir):
    dataloader_generator = BachDataloaderGenerator(
        sequences_size=24
    )

    load_or_pickle_distributions(dataloader_generator.dataset)
    grader = Grader(dataset=dataloader_generator.dataset,
                    features=FEATURES)
    
    answer_key = {}
    trial_ids = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]
    
    for trial_id in tqdm(trial_ids):
        a_score = music21.converter.parse(os.path.join(experiment_dir, trial_id, f'{trial_id}_a.mid'))
        b_score = music21.converter.parse(os.path.join(experiment_dir, trial_id, f'{trial_id}_b.mid'))
        a_grade = grader.grade_chorale(a_score)[0]
        b_grade = grader.grade_chorale(b_score)[0]

        if a_grade > b_grade:
            real_label = 'a'
            fake_label = 'b'
        else:
            real_label = 'b'
            fake_label = 'a'
        answer_key[int(trial_id)] = f'real: {real_label}, fake: {fake_label}'
    
    return answer_key


def get_accuracy(answers, answer_key):
    assert answers.keys() == answer_key.keys()
    n = len(answers)
    
    correct_ct = 0
    for i in range(n):
        if answer_key[i] == answers[i]:
            correct_ct += 1

    return correct_ct/n


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


if __name__ == '__main__':
    main()
