"""
@author: Gaetan Hadjeres
"""
import importlib
import os
import shutil
from datetime import datetime
import csv
from itertools import islice
import click
import torch

from transformer_bach.bach_dataloader import BachDataloaderGenerator
from transformer_bach.small_bach_dataloader import SmallBachDataloaderGenerator
from transformer_bach.decoder_relative import TransformerBach
from transformer_bach.getters import get_data_processor
from transformer_bach.melodies import MARIO_MELODY, TETRIS_MELODY, LONG_TETRIS_MELODY
from transformer_bach.DatasetManager.helpers import load_or_pickle_distributions
from transformer_bach.constraint_helpers import score_to_hold_representation_for_voice
from transformer_bach.utils import ensure_dir, get_threshold
from Grader.grader import Grader, FEATURES
from tqdm import tqdm

import music21

@click.command()
@click.option('--train', is_flag=True)
@click.option('--load', is_flag=True)
@click.option('--update', is_flag=True,
              help='update the given model for update_iterations')
@click.option('--overfitted', is_flag=True, 
              help='whether to load the overfitted model')
@click.option('--config', type=click.Path(exists=True))
@click.option('--num_workers', type=int, default=0)
def main(train,
         load,
         update,
         overfitted,
         config,
         num_workers,
         ):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')

    # Load config
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config

    # compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        config['timestamp'] = timestamp

    # set model_dir
    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'

    # === Decoder ====
    dataloader_generator_kwargs = config['dataloader_generator_kwargs']
    bach_dataloader_generator = BachDataloaderGenerator(
        sequences_size=dataloader_generator_kwargs['sequences_size']
    )

    data_processor = get_data_processor(
        dataloader_generator=bach_dataloader_generator,
        data_processor_type=config['data_processor_type'],
        data_processor_kwargs=config['data_processor_kwargs']
    )

    decoder_kwargs = config['decoder_kwargs']
    num_channels = 4            # is this number of voices?
    num_events_grouped = 4
    num_events = dataloader_generator_kwargs['sequences_size'] * 4
    transformer = TransformerBach(
        model_dir=model_dir,
        dataloader_generator=bach_dataloader_generator,
        data_processor=data_processor,
        d_model=decoder_kwargs['d_model'],
        num_encoder_layers=decoder_kwargs['num_encoder_layers'],
        num_decoder_layers=decoder_kwargs['num_decoder_layers'],
        n_head=decoder_kwargs['n_head'],
        dim_feedforward=decoder_kwargs['dim_feedforward'],
        dropout=decoder_kwargs['dropout'],
        positional_embedding_size=decoder_kwargs['positional_embedding_size'],
        num_channels=num_channels,
        num_events=num_events,
        num_events_grouped=num_events_grouped
    )

    if load:
        if overfitted:
            transformer.load(early_stopped=False)
        else:
            transformer.load(early_stopped=True)
        transformer.to('cuda')

    if train or update:
        # Copy .py config file in the save directory before training
        if not load:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            shutil.copy(config_path, f'{model_dir}/config.py')
            transformer.to('cuda')
    
    if train:
        transformer.train_model(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            plot=True,
            num_workers=num_workers
        )

    load_or_pickle_distributions(bach_dataloader_generator.dataset)
    grader = Grader(dataset=bach_dataloader_generator.dataset,
                    features=FEATURES)

    # BASELINE EXPERIMENT
    if update:
        update_iterations = config['update_iterations']
        generations_per_iteration = config['generations_per_iteration']
        batch_size = 351 // update_iterations

        update_file = open(f'results/update_grades_over_bach_chorales.csv', 'w')
        reader = csv.writer(update_file)
        reader.writerow(['', 'iter', 'grade'] + FEATURES)

        for i in range(update_iterations):
            print(f'----------- Iteration {i} -----------')
            print(f'Update model on chorales {i * batch_size} through {(i+1) * batch_size}')
            print('Creating dataset')
            next_dataloader_generator = SmallBachDataloaderGenerator(
                sequences_size=dataloader_generator_kwargs['sequences_size'],
                start_idx = i * batch_size,
                end_idx = (i+1) * batch_size
            )
            
            transformer.dataloader_generator = next_dataloader_generator            
            print('Training model on dataset')
            transformer.train_model(batch_size=config['batch_size'],
                                    num_batches=config['num_batches'],
                                    num_epochs=config['num_epochs'],
                                    lr=config['lr'],
                                    num_workers=num_workers)
            
            print(f'Generate {generations_per_iteration} chorales')
            ensure_dir(f'{transformer.model_dir}/generations/{i}/')
            for j in tqdm(range(generations_per_iteration)):
                score = transformer.generate(temperature=0.9,
                                             top_p=0.8,
                                             batch_size=1,
                                             melody_constraint=None)[0]
                score.write('midi', f'{transformer.model_dir}/generations/{i}/c{j}.mid')
                
                grade, chorale_vector = grader.grade_chorale(score)
                reader.writerow([i, j, grade, *chorale_vector])      # iteration, generation #, grade
    
    # grade_bach(grader=grader, 
    #            bach_iterator=bach_dataloader_generator.dataset.iterator_gen(), 
    #            grades_csv='results/bach_grades.csv')
    
    # grade_unconstrained_mock(grader=grader,
    #                          transformer=transformer,
    #                          grades_csv='results/unconstrained_mock_grades.csv',
    #                          num_generations=351)
    
    grade_constrained_mock(grader=grader,
                           transformer=transformer,
                           grades_csv='results/tmp.csv',
                           bach_iterator=islice(bach_dataloader_generator.dataset.iterator_gen(), 1),
                           output_dir='chorales/testing/')



def grade_bach(grader, 
               bach_iterator, 
               grades_csv='results/tmp.csv'):
    """
    grade Bach chorales
    """
    print('Grading Bach chorales')
    bach_grades = []

    for bach_score in tqdm(bach_iterator):
        grade, chorale_vector = grader.grade_chorale(bach_score)
        bach_grades.append([grade, *chorale_vector])

    print('Writing data to csv files')
    with open(grades_csv, 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(bach_grades):
            reader.writerow([i, *grades])


def grade_unconstrained_mock(grader, 
                             transformer,
                             grades_csv='results/tmp.csv',
                             num_generations=1):
    """
    Arguments:
        grader: Grader object
        transformer: model for generation
        grades_csv: csv file to write grades to
        num_generations: number of generations
    """
    print('Generating and grading unconstrained mock chorales')
    mock_grades = []
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
    with open(grades_csv, 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(mock_grades):
            reader.writerow([i, *grades])


def grade_constrained_mock(grader,
                           transformer,
                           grades_csv='results/tmp.csv',
                           bach_iterator=None,
                           output_dir=None,
                           ):
    """
    Arguments:
        grader: Grader object
        transformer: model for generation
        grades_csv: csv file to write grades to
        bach_iterator: iterator containing Bach chorales
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
        if output_dir is None:
            output_dir = f'{transformer.model_dir}/constrained_mocks/'
        ensure_dir(output_dir)
        mock_score.write('midi', f'{output_dir}/{i}.mid')
        
        # grade chorale
        grade, chorale_vector = grader.grade_chorale(mock_score)
        mock_grades.append([grade, *chorale_vector])

    print('Writing data to csv file')
    with open(grades_csv, 'w') as chorale_file:
        reader = csv.writer(chorale_file)
        reader.writerow(['', 'grade'] + FEATURES)
        for i, grades in enumerate(mock_grades):
            reader.writerow([i, *grades])


if __name__ == '__main__':
    main()
