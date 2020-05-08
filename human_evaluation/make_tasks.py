"""
create tasks.csv in the format of

task_id,1,2,3,4,5,6,7,8,9,10

where the first column is a unique task_id, and the rest of the columns are populated by pair_ids referring to
pairs in answer_key.csv
"""

import sys
sys.path.insert(0, '../')

import csv
import pandas as pd
import numpy as np
import numpy.random as random
from tqdm import tqdm
from numpy import random
import pandas as pd
from itertools import combinations

from helpers import NUM_TASKS, CSV, is_midi, IDX_TO_COMPARISON

BACH_DIR = 'chorales/cleaned_bach_chorales'
BASE_DIR = 'models/base_05-06_19:15/unconstrained_mocks/'
AUG_GEN_DIR = 'models/aug-gen_05-06_19:15/unconstrained_mocks/'
BAD_DIR = 'models/aug-gen_05-06_19:15/unconstrained_mocks/'


def main():
    directories = {'aug-gen': AUG_GEN_DIR, 'base': BASE_DIR, 'bach': BACH_DIR, 'bad': BAD_DIR}
    make_pairs(directories)
    make_tasks()


def make_tasks():
    """
    creates tasks.csv, where each row is a task of randomly selected pair_ids
    corresponding to the pairs in *_pairs.csv
    """
    tasks = []

    for task_id in range(NUM_TASKS):
        # select pairs of each type
        a_pairs = [f'a_{idx}' for idx in random.choice(range(351), 3)]
        b_pairs = [f'b_{idx}' for idx in random.choice(range(351), 3)]
        c_pairs = [f'c_{idx}' for idx in random.choice(range(351), 3)]
        d_pair = f'd_{random.choice(range(351))}'
        # order the pairs
        d_pos = random.choice(range(3,6))
        part1 = a_pairs + b_pairs
        random.shuffle(part1)
        part1.insert(d_pos, d_pair)
        random.shuffle(c_pairs)
        pairs = part1 + c_pairs
        tasks.append([task_id, *pairs])

    with open(CSV['tasks'], 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(['task_id','1','2','3','4','5','6','7','8','9','10'])
        writer.writerows(tasks)


def make_pairs(dirs):
    """
    Arguments:
        directories: key is model, value is generation directory containing grades.csv
    
    creates a *_pairs.csv file for every type of comparison
    """
    labels = ['a', 'b']
    for label1, label2 in list(IDX_TO_COMPARISON.values()):
        # pairs df
        columns = ['pair_id', f'{label1}_id', f'{label2}_id', 
                   f'{label1}_grade', f'{label2}_grade',
                   f'{label1}_is', f'{label2}_is']
        pairs_df = pd.DataFrame(columns=columns)
        # randomly order ids 
        ids1 = [i for i in range(351)]
        ids2 = [i for i in range(351)]
        random.shuffle(ids1)
        random.shuffle(ids2)
        # read grades as dataframe
        grades_df1 = pd.read_csv(f'{dirs[label1]}/grades.csv')
        grades_df2 = pd.read_csv(f'{dirs[label2]}/grades.csv')
        # create pairs
        for pair_id, (id1, id2) in enumerate(zip(ids1, ids2)):
            random.shuffle(labels)
            pair = pd.DataFrame(
                [[pair_id, id1, id2, grades_df1.at[id1, 'grade'], grades_df2.at[id2, 'grade'], labels[0], labels[1]]], 
                columns=columns
            )
            pairs_df = pairs_df.append(pair)

        pairs_df.to_csv(f'human_evaluation/data/{label1}_{label2}_pairs.csv', index=False)


if __name__ == '__main__':
    main()
