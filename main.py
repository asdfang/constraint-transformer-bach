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
import random

from transformer_bach.bach_dataloader import BachDataloaderGenerator
from transformer_bach.small_bach_dataloader import SmallBachDataloaderGenerator
from transformer_bach.decoder_relative import TransformerBach
from transformer_bach.getters import get_data_processor
from transformer_bach.melodies import MARIO_MELODY, TETRIS_MELODY, LONG_TETRIS_MELODY
from transformer_bach.DatasetManager.helpers import load_or_pickle_distributions
from transformer_bach.constraint_helpers import score_to_hold_representation_for_voice
from transformer_bach.utils import ensure_dir
from experiments.update_model import update_on_generations, update_on_bach
from experiments.generate_and_grade import grade_bach, grade_constrained_mock, grade_unconstrained_mock
from Grader.grader import Grader, FEATURES
from Grader.helpers import get_threshold
from tqdm import tqdm

import music21

@click.command()
@click.option('--train', is_flag=True)
@click.option('--load', is_flag=True)
@click.option('--update', is_flag=True,
              help='update the given model')
@click.option('--generate', is_flag=True,
              help='generate from the given model')
@click.option('--overfitted', is_flag=True, 
              help='whether to load the overfitted model')
@click.option('--config', type=click.Path(exists=True))
@click.option('--num_workers', type=int, default=0)
def main(train,
         load,
         update,
         generate,
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
        include_transpositions=dataloader_generator_kwargs['include_transpositions'],
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

    if update:
        update_on_generations(transformer=transformer, 
                              grader=grader, 
                              config=config)
    
    if generate:
        grade_unconstrained_mock(transformer=transformer, grader=grader)


if __name__ == '__main__':
    main()
