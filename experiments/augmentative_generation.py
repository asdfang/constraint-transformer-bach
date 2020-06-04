import csv
import random
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
import numpy as np
import time

from transformer_bach.decoder_relative import TransformerBach
from Grader.grader import Grader, FEATURES
from Grader.helpers import get_threshold
from transformer_bach.utils import ensure_dir
from transformer_bach.small_bach_dataloader import SmallBachDataloaderGenerator
from transformer_bach.constraint_helpers import score_to_hold_representation_for_voice


unharmonizable_chorales = [81, 188, 202, 240, 263, 296]
MAX_BATCH_SIZE = 25


def augmentative_generation(transformer, 
                            grader,
                            config,
                            num_workers,
                            bach_iterator,
                            threshold):
    """
    transformer: TransformerBach object to be trained
    grader: Grader object used for grading generations
    config: config dictionary
    num_workers: 
    bach_iterator: iterator of training chorales
    threshold: the threshold to use for including generations in the training dataset. if float('-inf'), 
        all generations are included. if float('inf'), no generations are included and 
        this is the setting for the base model trained only on Bach chorales.
    
    if continuing model training, comment out lines 48, 51, 54, turn all the 'w' into 'a', 
    change the range endpoints on line 65, and replace the val_loss with the best one seen in training
    """
    # config files
    num_epochs = config['num_epochs']
    generations_per_epoch = config['generations_per_epoch']
    dataloader_generator_kwargs = config['dataloader_generator_kwargs']
    generation_kwargs = config['generation_kwargs']

    # csv files
    grades_file = open(f'{transformer.model_dir}/grades.csv', 'w')
    grades_csv = csv.writer(grades_file)
    grades_csv.writerow(['epoch', 'gen_id', 'grade'] + FEATURES)
    dataset_size_file = open(f'{transformer.model_dir}/dataset_sizes.csv', 'w')
    dataset_size_csv = csv.writer(dataset_size_file)
    dataset_size_csv.writerow(['epoch', 'num_unique_generations', 'num_chorales', 'num_examples'])
    loss_file = open(f'{transformer.model_dir}/loss.csv', 'w')
    loss_csv = csv.writer(loss_file)
    loss_csv.writerow(['epoch', 'train_loss', 'val_loss'])

    # get threshold
    print(f'Selection threshold: {threshold}')

    # initialize grades_seen
    with open('chorales/cleaned_bach_chorales/grades.csv', 'r') as fin:
        bach_df = pd.read_csv(fin)
        bach_grades = list(bach_df['grade'])
    grades_seen = bach_grades

    # augmentive generation main loop
    best_val = 1e8
    bach_and_good_mock = bach_iterator
    for epoch in range(num_epochs):
        print(f'----------- Epoch {epoch} -----------')
        print(f'Generate {generations_per_epoch} chorales')
        ensure_dir(f'{transformer.model_dir}/generations/{epoch}/')
        
        # calculate batch sizes
        batch_sizes = [MAX_BATCH_SIZE] * (generations_per_epoch // MAX_BATCH_SIZE)
        if generations_per_epoch % MAX_BATCH_SIZE != 0:
            batch_sizes += [generations_per_epoch % MAX_BATCH_SIZE]
        
        # generate scores
        scores = []
        start = time.time()
        gen_id = 0
        good_ct = 0
        for batch_size in batch_sizes:
            score_batch = transformer.generate(
                temperature=generation_kwargs['temperature'],
                top_p=generation_kwargs['top_p'],
                batch_size=batch_size,
                melody_constraint=None,
            )
            for score in score_batch:
                score.write('musicxml', f'{transformer.model_dir}/generations/{epoch}/{gen_id}.xml')
                score = score.stripTies(retainContainers=False)
                grade, chorale_vector = grader.grade_chorale(score)
                grades_csv.writerow([epoch, gen_id, grade, *chorale_vector])
                grades_file.flush()

                # avoid adding duplicate chorales
                if grade not in grades_seen:
                    scores.append(score)
                    grades_seen.append(grade)
                    if grade < threshold:
                        print(f'Picked chorale {gen_id} with grade {grade}')
                        bach_and_good_mock.append(score)
                        good_ct += 1
                gen_id += 1

        print(f'Generation and grading time (min): {(time.time()-start)/60}')
        print(f'{good_ct} chorales passed the threshold in this epoch')
        
        # only aug_gen creates new datasets
        np.random.shuffle(bach_and_good_mock)
        if good_ct != 0:
            print('Creating dataset of Bach and good mock chorales')
            next_dataloader_generator = SmallBachDataloaderGenerator(
                dataset_name=f'bach_and_good_mock_{epoch}',
                chorales=bach_and_good_mock,
                sequences_size=dataloader_generator_kwargs['sequences_size'],
                include_transpositions=True,
            )
        else:
            next_dataloader_generator = transformer.train_dataloader_generator
        
        # record size of dataset
        num_unique_generations = len(scores)
        num_chorales = len(bach_and_good_mock)
        num_examples = len(next_dataloader_generator.dataset.get_tensor_dataset(
            next_dataloader_generator.dataset.cache_dir))
        dataset_size_csv.writerow([epoch, num_unique_generations, num_chorales, num_examples])    
        dataset_size_file.flush()
        # update training dataloader generator
        transformer.train_dataloader_generator = next_dataloader_generator
            
        print('Training model on dataset')
        train_loss, val_loss = transformer.train_model(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_epochs=1,
            lr=config['lr'],
            num_workers=num_workers
        )

        # write loss to file
        loss_csv.writerow([epoch, train_loss, val_loss])
        loss_file.flush()

        # save model
        if epoch >= 10:
            transformer.save(early_stopped=False, epoch=epoch)
        else:
            transformer.save(early_stopped=False)
        if val_loss < best_val:   
            transformer.save(early_stopped=True)
            best_val = val_loss
    
    grades_file.close()
    dataset_size_file.close()
    loss_file.close()
