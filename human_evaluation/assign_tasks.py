"""
create tasks.csv in the format of

task_id,1,2,3,4,5,6,7,8,9,10

where the first column is a unique task_id, and the rest of the columns are populated by pair_ids referring to
pairs in answer_key.csv
"""

import sys
sys.path.insert(0, '../')

import random
import csv
from human_evaluation.helpers import NUM_TASKS, NUM_PAIRS_PER_TASK, MOCK_MODELS

with open('tasks.csv', 'w') as fo:
    writer = csv.writer(fo)
    writer.writerow(['task_id','1','2','3','4','5','6','7','8','9','10'])
    pair_ids = [i for i in range(351*len(MOCK_MODELS))]
    task_id = 0

    for _ in range(NUM_TASKS // (len(pair_ids) // NUM_PAIRS_PER_TASK)):
        random.shuffle(pair_ids)
        pair_ids = pair_ids[:len(pair_ids) - len(pair_ids) % NUM_PAIRS_PER_TASK]        # trim pair_ids to be divisible by NUM_PAIRS_PER_TASK
        for idx in range(0, len(pair_ids), NUM_PAIRS_PER_TASK):
            writer.writerow([task_id, *pair_ids[idx:idx+NUM_PAIRS_PER_TASK]])
            task_id += 1
