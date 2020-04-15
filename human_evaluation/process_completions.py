import json
import os
import pandas as pd
import csv
import numpy as np

from helpers import NUM_PAIRS_PER_TASK, NULL_SYMBOL


def parse_completion(completion_json_path):
    """
    Arguments
        completion_json_path: path to json containing one completion

    Returns a dictionary of relevant information for a single completion
    ex: parse_completion('label-studio/label_studio/tbach-eval/completions/1.json')
    """
    with open(completion_json_path) as f:
        c = json.load(f)
    # init
    completion = {}
    completion['task_id'] = c['id']
    completion['background'] = NULL_SYMBOL
    for i in range(1, NUM_PAIRS_PER_TASK + 1):
        completion[str(i)] = NULL_SYMBOL
    
    # fill
    assert len(c['completions']) == 1
    results = c['completions'][0]['result']
    for res in results:
        completion[res['from_name']] = res['value']['choices'][0][0]

    return completion

#TODO: for alisa :)
# completions_dir = 'label-studio/label_studio/tbach-eval/completions'
# list of files in the completions_dir in increasing numerical order:
# completions = sorted(os.listdir(completions_dir), key=lambda f: int(f.split('.')[0]))

def update_completed_tasks(completion):
    """
    Arguments
        completion: a dictionary representing one completion
    
    Update completed_tasks.csv with the given completion
    """

    with open('../human_evaluation/tasks.csv', 'r') as tasks:
        tasks_df = pd.read_csv(tasks)
    with open('../human_evaluation/answer_key.csv', 'r') as answer_key:
        answer_key_df = pd.read_csv(answer_key)
    with open('../human_evaluation/completed_tasks.csv', 'r') as completion_tasks:
        completion_tasks_df = pd.read_csv(completion_tasks)
        assert completion['task_id'] not in completion_tasks_df['task_id']
        
    with open('../human_evaluation/completed_tasks.csv', 'a') as completed_tasks: 
        completed_tasks_writer = csv.writer(completed_tasks)

        picks = [completion[str(i)] for i in range(1, NUM_PAIRS_PER_TASK + 1)]
        task_id = completion['task_id']
        answers = []
        for i in range(1, NUM_PAIRS_PER_TASK + 1):
            pair_id = tasks_df.at[task_id, str(i)]
            bach_is = answer_key_df.at[pair_id, 'bach_is']
            answers.append(bach_is)
        acc = np.sum([1 for pick, answer in zip(picks, answers) if pick == answer])/NUM_PAIRS_PER_TASK
        completed_tasks_writer.writerow([completion['task_id'], completion['background'], *picks, *answers, acc])


completion = parse_completion('../../label-studio/label_studio/tbach-eval/completions/2.json')
update_completed_tasks(completion)