import csv
import random
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
import numpy as np

from transformer_bach.decoder_relative import TransformerBach
from Grader.grader import Grader, FEATURES
from Grader.helpers import get_threshold
from transformer_bach.utils import ensure_dir
from transformer_bach.small_bach_dataloader import SmallBachDataloaderGenerator
from transformer_bach.constraint_helpers import score_to_hold_representation_for_voice


unharmonizable_chorales = [81, 188, 202, 240, 263, 296]
max_batch_size = 10


def update_on_bach(transformer,
                   grader,
                   config,
                   selections_per_iteration,
                   bach_iterator,
                   num_workers):    
    # convert bach_iterator to list
    bach_chorales = []
    for _ in range(351):   
        bach_chorales.append(next(bach_iterator))

    update_iterations = config['update_iterations']
    generations_per_iteration = config['generations_per_iteration']
    dataloader_generator_kwargs = config['dataloader_generator_kwargs']

    update_file = open(f'{transformer.model_dir}/update_grades.csv', 'w')
    reader = csv.writer(update_file)
    reader.writerow(['iter', 'gen_id', 'grade'] + FEATURES)

    thres = get_threshold()
    print(f'Selection threshold: {thres}')

    for i in range(update_iterations):
        print(f'----------- Iteration {i} -----------')
        print(f'Generate {generations_per_iteration} chorales')
        ensure_dir(f'{transformer.model_dir}/update_generations/{i}/')
        picked_chorales = []
        scores = []
        for _ in range(5):
            scores_batch = transformer.generate(temperature=0.9,
                                                top_p=0.8,
                                                batch_size=generations_per_iteration // 5,
                                                melody_constraint=None)
            scores.extend(scores_batch)
        
        for j, score in enumerate(scores):
            score.write('midi', f'{transformer.model_dir}/update_generations/{i}/c{j}.mid')
            grade, chorale_vector = grader.grade_chorale(score)
            reader.writerow([i, j, grade, *chorale_vector])      # iteration, generation #, grade
            if grade > thres:
                print(f'Picked chorale {j} with grade {grade}')
                picked_chorales.append(score)
        
        print(f'{len(picked_chorales)} chorales passed the threshold')
        
        print(f'Pick {selections_per_iteration[i]} Bach chorales to train on')
        if selections_per_iteration[i] == 0:
            continue
        
        # randomly select chorales
        random.shuffle(bach_chorales)
        picked_bach_chorales = bach_chorales[:selections_per_iteration[i]]

        print('Creating dataset of Bach chorales')
        next_dataloader_generator = SmallBachDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            dataset_name=f'picked_bach_{i}',
            chorales=picked_bach_chorales,
        )
        transformer.dataloader_generator = next_dataloader_generator   
        
        print('Training model on dataset')
        transformer.train_model(batch_size=config['batch_size'],
                                num_batches=config['num_batches'],
                                num_epochs=config['update_epochs'],
                                lr=config['update_lr'],
                                num_workers=num_workers)


def update_on_generations_method_1(transformer, 
                                   grader, 
                                   config,
                                   num_workers):
    update_iterations = config['update_iterations']
    generations_per_iteration = config['generations_per_iteration']
    dataloader_generator_kwargs = config['dataloader_generator_kwargs']

    update_file = open(f'{transformer.model_dir}/update_grades.csv', 'w')
    reader = csv.writer(update_file)
    reader.writerow(['iter', 'gen_id', 'grade'] + FEATURES)

    thres = get_threshold()
    print(f'Selection threshold: {thres}')

    with open('chorales/cleaned_bach_chorales/bach_grades.csv', 'r') as fin:
        bach_df = pd.read_csv(fin)
        bach_grades = list(bach_df['grade'])
    
    grades_seen = bach_grades

    for i in range(update_iterations):
        print(f'----------- Iteration {i} -----------')
        print(f'Generate {generations_per_iteration} chorales')
        ensure_dir(f'{transformer.model_dir}/update_generations/{i}/')
        picked_chorales = []
        scores = []

        loop_it = 0
        while len(scores) < generations_per_iteration:
            print(f'---- Times in while loop: {loop_it} ----')
            scores_batch = transformer.generate(temperature=0.9,
                                                top_p=0.8,
                                                batch_size=10,
                                                melody_constraint=None)
            for j, score in enumerate(scores_batch):
                grade, chorale_vector = grader.grade_chorale(score)
                if grade not in grades_seen:
                    scores.append(score)
                    grades_seen.append(grade)
            
            if loop_it > 50:
                print('okay, I am stuck and you should abandon this version of method (1)')
                return
            
            loop_it += 1
        
        # discard excess chorales in last batch
        scores = scores[:50]
        
        for j, score in enumerate(scores):
            score.write('midi', f'{transformer.model_dir}/update_generations/{i}/c{j}.mid')
            grade, chorale_vector = grader.grade_chorale(score)
            reader.writerow([i, j, grade, *chorale_vector])      # iteration, generation #, grade
            if grade > thres:
                print(f'Picked chorale {j} with grade {grade}')
                picked_chorales.append(score)
        
        print(f'Picked {len(picked_chorales)} chorales')

        if len(picked_chorales) == 0:
            continue
        
        print('Creating dataset of selected generations')
        next_dataloader_generator = SmallBachDataloaderGenerator(
            dataset_name=f'picked_mock_m1_{i}',
            chorales=picked_chorales,
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            include_transpositions=True,
        )
        transformer.dataloader_generator = next_dataloader_generator            
        
        print('Training model on dataset')
        transformer.train_model(batch_size=config['batch_size'],
                                num_batches=config['num_batches'],
                                num_epochs=config['update_epochs'],
                                lr=config['update_lr'],
                                num_workers=num_workers)


def update_on_generations_method_2(transformer, 
                                   grader,
                                   config,
                                   num_workers,
                                   bach_iterator):
    # convert bach_iterator to list
    bach_melodies = {}
    for idx, bach_score in enumerate(bach_iterator):
        if idx not in unharmonizable_chorales:
            bach_melodies[idx] = score_to_hold_representation_for_voice(bach_score, voice=0)
    
    update_iterations = config['update_iterations']
    generations_per_iteration = config['generations_per_iteration']
    dataloader_generator_kwargs = config['dataloader_generator_kwargs']

    update_file = open(f'{transformer.model_dir}/update_grades.csv', 'w')
    reader = csv.writer(update_file)
    reader.writerow(['iter', 'gen_id', 'bach_id', 'grade'] + FEATURES)

    thres = get_threshold()
    print(f'Selection threshold: {thres}')

    for i in range(update_iterations):
        print(f'----------- Iteration {i} -----------')
        print(f'Generate {generations_per_iteration} chorales')
        ensure_dir(f'{transformer.model_dir}/update_generations/{i}/')
        picked_bach_idx = random.sample(bach_melodies.keys(), generations_per_iteration)

        scores = []
        for j in tqdm(range(generations_per_iteration)):
            idx = picked_bach_idx[j]
            melody = bach_melodies[idx]
            try:
                score = transformer.generate(temperature=0.9,
                                             top_p=0.8,
                                             batch_size=1,
                                             melody_constraint=melody,
                                             hard_constraint=True)[0]
                scores.append(score)
            except:
                score.write('midi', f'{transformer.model_dir}/bad_chorale.mid')
        
        picked_chorales = []
        for j, (idx, score) in enumerate(zip(picked_bach_idx, scores)):
            score.write('midi', f'{transformer.model_dir}/update_generations/{i}/c{j}.mid')
            grade, chorale_vector = grader.grade_chorale(score)
            reader.writerow([i, j, idx, grade, *chorale_vector])      # iteration, generation #, grade
            if grade > thres:
                print(f'Picked chorale {j} with grade {grade}')
                picked_chorales.append(score)
        
        print(f'Picked {len(picked_chorales)} chorales')

        if len(picked_chorales) == 0:
            continue
        
        print('Creating dataset of selected generations')
        next_dataloader_generator = SmallBachDataloaderGenerator(
            dataset_name=f'picked_mock_m2_{i}',
            chorales=picked_chorales,
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            include_transpositions=True,
        )
        transformer.dataloader_generator = next_dataloader_generator            
        
        print('Training model on dataset')
        transformer.train_model(batch_size=config['batch_size'],
                                num_batches=config['num_batches'],
                                num_epochs=config['update_epochs'],
                                lr=config['update_lr'],
                                num_workers=num_workers)


def update_on_generations_method_5(transformer, 
                                   grader,
                                   config,
                                   num_workers,
                                   bach_iterator,
                                   update_batch_size):
    # convert bach_iterator to list
    bach_and_good_mock = []
    for score in bach_iterator:
        grade, chorale_vector = grader.grade_chorale(score)
        bach_and_good_mock.append((score, grade))
    
    update_iterations = config['update_iterations']
    generations_per_iteration = config['generations_per_iteration']
    dataloader_generator_kwargs = config['dataloader_generator_kwargs']

    update_file = open(f'{transformer.model_dir}/update_grades.csv', 'w')
    reader = csv.writer(update_file)
    reader.writerow(['iter', 'gen_id', 'grade'] + FEATURES)

    thres = get_threshold()
    print(f'Selection threshold: {thres}')

    for i in range(update_iterations):
        print(f'----------- Iteration {i} -----------')
        print(f'Generate {generations_per_iteration} chorales')
        ensure_dir(f'{transformer.model_dir}/update_generations/{i}/')
        
        # calculate batch sizes
        batch_sizes = [max_batch_size]*(generations_per_iteration//max_batch_size)
        if generations_per_iteration % max_batch_size != 0:
            batch_sizes += [generations_per_iteration % max_batch_size]
        
        # generate scores
        scores = []
        for j in range(len(batch_sizes)):
            score_batch = transformer.generate(temperature=0.9,
                                               top_p=0.8,
                                               batch_size=batch_sizes[j],
                                               melody_constraint=None)
            scores.extend(score_batch)

        # select scores based on grade
        good_ct = 0
        for j, score in enumerate(scores):
            score.write('midi', f'{transformer.model_dir}/update_generations/{i}/c{j}.mid')
            grade, chorale_vector = grader.grade_chorale(score)
            reader.writerow([i, j, grade, *chorale_vector])      # iteration, generation #, grade
            if grade > thres:
                print(f'Picked chorale {j} with grade {grade}')
                bach_and_good_mock.append((score, grade))
                good_ct += 1
        
        print(f'{good_ct} chorales passed the threshold in this iteration')
        
        # sample from bach_and_good_mock based on grade
        print('Creating dataset of sampled Bach and mock chorales')
        p = softmax([grade for score, grade in bach_and_good_mock])
        chorales = [score for score, grade in bach_and_good_mock]
        picked_chorales = np.random.choice(chorales, size=update_batch_size, p=p)
        next_dataloader_generator = SmallBachDataloaderGenerator(
            dataset_name=f'picked_mock_m5_{i}',
            chorales=picked_chorales,
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            include_transpositions=True,
        )
        transformer.dataloader_generator = next_dataloader_generator            
        
        print('Training model on dataset')
        transformer.train_model(batch_size=config['batch_size'],
                                num_batches=config['num_batches'],
                                num_epochs=config['update_epochs'],
                                lr=config['update_lr'],
                                num_workers=num_workers)
