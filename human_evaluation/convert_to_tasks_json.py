"""
From tasks, {a,b,c,d}_pairs, write a file called 'tasks.json' in human_evaluation/
"""
import sys
sys.path.insert(0, '../')

import pandas as pd
import json
from tqdm import tqdm
from helpers import NUM_PAIRS_PER_TASK, CSV, AUDIO_DIRS

pairs = {}

with open(CSV['a_pairs'], 'r') as fin:
    pairs['a'] = pd.read_csv(fin)
with open(CSV['b_pairs'], 'r') as fin:
    pairs['b'] = pd.read_csv(fin)
with open(CSV['c_pairs'], 'r') as fin:
    pairs['c'] = pd.read_csv(fin)
with open(CSV['d_pairs'], 'r') as fin:
    pairs['d'] = pd.read_csv(fin)
with open(CSV['tasks'], 'r') as fin:
    tasks = pd.read_csv(fin)

pairs['a'].set_index('pair_id')
pairs['b'].set_index('pair_id')
pairs['c'].set_index('pair_id')
pairs['d'].set_index('pair_id')
tasks.set_index('task_id')


all_tasks_json = {}

for index, task_row in tqdm(tasks.iterrows()): # for each task
    task_json = {}
    data = {}
    task_id = int(task_row['task_id'])

    for i in range(NUM_PAIRS_PER_TASK): # for each pair in the current task
        pair = task_row[i+1] # task_ids are 1-indexed
        pair_idx, pair_row_num = task_row[i+1].split('_')
        pair_row_num = int(pair_row_num)
        pair_row = pairs[pair_idx].iloc[pair_row_num] # get row in correct csv file

        # pretest (always the same)
        data['urlpt1a'] = AUDIO_DIRS['pretest'] + "q1-A.mp3"
        data['urlpt1b'] = AUDIO_DIRS['pretest'] + "q1-B.mp3"
        data['urlpt2'] = AUDIO_DIRS['pretest'] + "q2.mp3"
        data['urlpt3'] = AUDIO_DIRS['pretest'] + "q3.mp3"
        data['urlpt4a'] = AUDIO_DIRS['pretest'] + "q4-A.mp3"
        data['urlpt4b'] = AUDIO_DIRS['pretest'] + "q4-B.mp3"
        data['urlpt5'] = AUDIO_DIRS['pretest'] + "q5.mp3"

        # order A and B correctly...
        if pair_idx == 'a': # aug-gen vs. bach
            if pair_row['bach_is'] == 'a':
                data['url'+str(i)+'a'] = AUDIO_DIRS['bach'] + str(pair_row['bach_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['aug-gen'] + str(pair_row['aug-gen_id']) + '.mp3'
            else:
                data['url'+str(i)+'a'] = AUDIO_DIRS['aug-gen'] + str(pair_row['aug-gen_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['bach'] + str(pair_row['bach_id']) + '.mp3'
        elif pair_idx == 'b': # base vs bach
            if pair_row['bach_is'] == 'a':
                data['url'+str(i)+'a'] = AUDIO_DIRS['bach'] + str(pair_row['bach_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['base'] + str(pair_row['base_id']) + '.mp3'
            else:
                data['url'+str(i)+'a'] = AUDIO_DIRS['base'] + str(pair_row['base_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['bach'] + str(pair_row['bach_id']) + '.mp3'
        elif pair_idx == 'c': # aug-gen vs. base
            if pair_row['aug-gen_is'] == 'a':
                data['url'+str(i)+'a'] = AUDIO_DIRS['aug-gen'] + str(pair_row['aug-gen_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['base'] + str(pair_row['base_id']) + '.mp3'
            else:
                data['url'+str(i)+'a'] = AUDIO_DIRS['base'] + str(pair_row['base_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['aug-gen'] + str(pair_row['aug-gen_id']) + '.mp3'
        else: # bad vs. bach
            if pair_row['bach_is'] == 'a':
                data['url'+str(i)+'a'] = AUDIO_DIRS['bach'] + str(pair_row['bach_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['bad'] + str(pair_row['bad_id']) + '.mp3'
            else:
                data['url'+str(i)+'a'] = AUDIO_DIRS['bad'] + str(pair_row['bad_id']) + '.mp3'
                data['url'+str(i)+'b'] = AUDIO_DIRS['bach'] + str(pair_row['bach_id']) + '.mp3'
    
    task_json['id'] = task_id
    task_json['data'] = data
    all_tasks_json[str(task_id)] = task_json


with open('human_evaluation/tasks.json', 'w') as f:
    s = json.dumps(all_tasks_json, indent=4)
    f.write(s)
    f.close()