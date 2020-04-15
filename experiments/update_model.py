import csv

from transformer_bach.decoder_relative import TransformerBach
from Grader.grader import Grader, FEATURES
from Grader.helpers import get_threshold
from transformer_bach.utils import ensure_dir
from transformer_bach.small_bach_dataloader import SmallBachDataloaderGenerator


def update_on_bach(transformer,
                   grader,
                   update_iterations,
                   generations_per_iteration,
                   selections_per_iteration):
    
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
        for _ in range(2):
            scores_batch = transformer.generate(temperature=0.9,
                                                top_p=0.8,
                                                batch_size=generations_per_iteration // 2,
                                                melody_constraint=None)
            scores.extend(scores_batch)
        
        for j, score in enumerate(scores):
            score.write('midi', f'{transformer.model_dir}/generations/{i}/c{j}.mid')
            grade, chorale_vector = grader.grade_chorale(score)
            reader.writerow([i, j, grade, *chorale_vector])      # iteration, generation #, grade
            if grade > thres:
                print(f'Picked chorale {j} with grade {grade}')
                picked_chorales.append(score)
        
        print(f'{len(picked_chorales)} chorales passed the threshold')
        
        print(f'Pick {selections_per_iteration[i]} Bach chorales to train on')
        if selections_per_iteration[i] == 0:
            continue
        
        bach_chorales = []
        for _ in range(351):
            bach_chorales.append(next(bach_dataloader_generator.dataset.iterator_gen()))
        random.shuffle(bach_chorales)
        picked_bach_chorales = bach_chorales[:selections_per_iteration[i]]

        print('Creating dataset of Bach chorales')
        next_dataloader_generator = SmallBachDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            dataset_name=f'picked_generations_{i}',
            chorales=picked_bach_chorales,
        )
        transformer.dataloader_generator = next_dataloader_generator   
        
        print('Training model on dataset')
        transformer.train_model(batch_size=config['batch_size'],
                                num_batches=config['num_batches'],
                                num_epochs=config['num_epochs'],
                                lr=config['lr'],
                                num_workers=num_workers)


def update_on_generations(transformer, 
                          grader, 
                          config):
    update_iterations = config['update_iterations']
    generations_per_iteration = config['generations_per_iteration']
    num_epochs_per_iteration = config['num_epochs_per_iteration']

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
        for _ in range(2):
            scores_batch = transformer.generate(temperature=0.9,
                                                top_p=0.8,
                                                batch_size=generations_per_iteration // 2,
                                                melody_constraint=None)
            scores.extend(scores_batch)
        
        for j, score in enumerate(scores):
            score.write('midi', f'{transformer.model_dir}/generations/{i}/c{j}.mid')
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
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            dataset_name=f'picked_generations_{i}',
            chorales=picked_chorales,
        )
        transformer.dataloader_generator = next_dataloader_generator            
        
        print('Training model on dataset')
        transformer.train_model(batch_size=config['batch_size'],
                                num_batches=config['num_batches'],
                                num_epochs=config['num_epochs'],
                                lr=config['lr'],
                                num_workers=num_workers)