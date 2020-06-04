import csv
from tqdm import tqdm
import os
import numpy as np
import music21

from transformer_bach.decoder_relative import TransformerBach
from transformer_bach.utils import parse_xml
from Grader.grader import Grader, FEATURES
from transformer_bach.constraint_helpers import score_to_hold_representation_for_voice
from transformer_bach.utils import ensure_dir

max_batch_size = 25


def grade_folder(chorale_dir, grader, grades_csv=None):
    """
    Arguments:
        chorale_dir: directory of chorales named 0.xml, 1.xml, etc.
        grader: Grader object
        grades_csv: file to write grades to
    
    grade a folder of chorales 
    """
    print(f'Grading chorales in {chorale_dir}')
    grades = []
    num_chorales = int(np.sum([1 for fname in os.listdir(chorale_dir) if fname.endswith('.xml')]))
    
    for chorale_idx in tqdm(range(num_chorales)):
        chorale_xml = f'{chorale_dir}/{chorale_idx}.xml'
        score = parse_xml(chorale_xml)
        grade, chorale_vector = grader.grade_chorale(score)
        grades.append([grade, *chorale_vector])
    
    if grades_csv is None:
        grades_csv = f'{chorale_dir}/grades.csv'
    print(f'Writing data to {chorale_dir}/{grades_csv}')
    with open(f'{chorale_dir}/{grades_csv}', 'w') as chorale_file:
        writer = csv.writer(chorale_file)
        writer.writerow(['', 'grade'] + grader.features)
        for i, grades in enumerate(grades):
            writer.writerow([i, *grades])


def grade_unconstrained_mock(grader, 
                             transformer,
                             output_dir=None,
                             num_generations=1):
    """
    Arguments:
        grader: Grader object
        transformer: model for generation
        grades_csv: csv file to write grades to
        num_generations: number of generations
    
    Usage example:
        grade_unconstrained_mock(grader=grader,
                                 transformer=transformer,
                                 grades_csv='results/unconstrained_mock_grades.csv',
                                 num_generations=351)
    """
    print('Generating and grading unconstrained mock chorales')
    mock_grades = []
    
    # calculate batch sizes
    batch_sizes = [max_batch_size]*(num_generations // max_batch_size)
    if num_generations % max_batch_size != 0:
        batch_sizes += [num_generations % max_batch_size]
    
    mock_scores = []
    for i in tqdm(range(len(batch_sizes))):
        score_batch = transformer.generate(temperature=0.9,
                                           top_p=0.8,
                                           batch_size=batch_sizes[i])
        mock_scores.extend(score_batch)
    
    for i, score in enumerate(mock_scores):
        # write score to XML
        if output_dir is None:
            output_dir = f'{transformer.model_dir}/unconstrained_mocks/'
        ensure_dir(output_dir)
        score.write('xml', f'{output_dir}/{i}.xml')
        
        # grade chorale
        grade, chorale_vector = grader.grade_chorale(score)
        mock_grades.append([grade, *chorale_vector])
    
    print('Writing data to csv file')
    grades_file = open(f'{output_dir}/grades.csv', 'w')
    reader = csv.writer(grades_file)
    reader.writerow(['', 'grade'] + FEATURES)
    for i, grades in enumerate(mock_grades):
        reader.writerow([i, *grades])
        grades_file.flush()
    grades_file.close()


def grade_constrained_mock(grader,
                           transformer,
                           output_dir=None,
                           bach_iterator=None,
                           ):
    """
    Arguments:
        grader: Grader object
        transformer: model for generation
        grades_csv: csv file to write grades to
        bach_iterator: iterator containing Bach chorales
    
    Usage example:
        grade_constrained_mock(grader=grader,
                               transformer=transformer,
                               grades_csv='results/tmp.csv',
                               bach_iterator=islice(bach_dataloader_generator.dataset.iterator_gen(), 1),
                               output_dir='chorales/testing/')
    """
    print('Generating and grading constrained mock chorales')
    mock_grades = []

    for i, bach_score in tqdm(enumerate(bach_iterator)):
        bach_melody = score_to_hold_representation_for_voice(bach_score, voice=0)
        try:
            mock_score = transformer.generate(temperature=0.9,
                                              top_p=0.8,
                                              batch_size=1,
                                              melody_constraint=bach_melody,
                                              hard_constraint=True)[0]
        # IndexError: index 96 is out of bounds for dimension 1 with size 96 on line 504
        except IndexError:
            print(f'chorale {i} is problem')
            mock_grades.append([float('-inf')])
            continue
        
        # write mock_score to XML
        if output_dir is None:
            output_dir = f'{transformer.model_dir}/constrained_mocks/'
        ensure_dir(output_dir)
        mock_score.write('xml', f'{output_dir}/{i}.xml')
        
        # grade chorale
        grade, chorale_vector = grader.grade_chorale(mock_score)
        mock_grades.append([grade, *chorale_vector])

    print('Writing data to csv file')
    with open(f'{output_dir}/grades.csv', 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(mock_grades):
            reader.writerow([i, *grades])