import csv

from transformer_bach.decoder_relative import TransformerBach
from Grader.grader import Grader, FEATURES


def grade_bach(grader, 
               bach_iterator, 
               grades_csv='tmp.csv'):
    """
    grade Bach chorales
    
    Usage example:
        grade_bach(grader=grader, 
                   bach_iterator=bach_dataloader_generator.dataset.iterator_gen(), 
                   grades_csv='results/bach_grades.csv')
    """
    print('Grading Bach chorales')
    bach_grades = []

    for bach_score in tqdm(bach_iterator):
        grade, chorale_vector = grader.grade_chorale(bach_score)
        bach_grades.append([grade, *chorale_vector])

    print('Writing data to csv files')
    with open(f'chorales/cleaned_bach_chorales/{grades_csv}', 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(bach_grades):
            reader.writerow([i, *grades])


def grade_unconstrained_mock(grader, 
                             transformer,
                             grades_csv='tmp.csv',
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
    
    ## TODO: call multiple times with batch_size of up to 25
    for i in tqdm(range(num_generations)):
        mock_score = transformer.generate(temperature=0.9,
                                          top_p=0.8,
                                          batch_size=1)[0]
        # write mock_score to MIDI
        output_dir = f'{transformer.model_dir}/unconstrained_mocks/'
        ensure_dir(output_dir)
        mock_score.write('midi', f'{output_dir}/{i}.mid')
        
        # grade chorale
        grade, chorale_vector = grader.grade_chorale(mock_score)
        mock_grades.append([grade, *chorale_vector])
    
    print('Writing data to csv file')
    with open(f'{output_dir}/{grades_csv}', 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(mock_grades):
            reader.writerow([i, *grades])


def grade_constrained_mock(grader,
                           transformer,
                           grades_csv='tmp.csv',
                           bach_iterator=None
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
        
        # write mock_score to MIDI
        output_dir = f'{transformer.model_dir}/constrained_mocks/'
        ensure_dir(output_dir)
        mock_score.write('midi', f'{output_dir}/{i}.mid')
        
        # grade chorale
        grade, chorale_vector = grader.grade_chorale(mock_score)
        mock_grades.append([grade, *chorale_vector])

    print('Writing data to csv file')
    with open(f'{output_dir}/grades_csv', 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(mock_grades):
            reader.writerow([i, *grades])