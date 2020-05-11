"""
Overwrites human_evaluation/data/completed_tasks.csv

USAGE:
python human_evaluation/process_completions.py

This file and web server are on different servers, so we'll have to copy over the completions dir to this server.
"""
import sys
sys.path.insert(0, '../')

import json
import os
import pandas as pd
import csv
import numpy as np

from helpers import NUM_PAIRS_PER_TASK, CSV, IDX_TO_COMPARISON, NUM_PRETEST_QUESTIONS


def main():
    completions_dir = 'human_evaluation/completions'
    completions = sorted(os.listdir(completions_dir), key=lambda f: int(f.split('.')[0]))
    rows = []
    for json_file in completions:
        completion = parse_completion(f'{completions_dir}/{json_file}')
        row = completion_dict_to_row(completion)
        rows.append(row)

    with open(CSV['completed_tasks'], 'w') as completed_tasks: 
        completed_tasks_writer = csv.writer(completed_tasks)
        ptq_labels = [f'ptq{i}' for i in range(1, NUM_PRETEST_QUESTIONS + 1)]
        q_labels = [f'q{i}' for i in range(1, NUM_PAIRS_PER_TASK + 1)]
        p1_answers = [f'bach_is{i}' for i in range(1, 8)]
        p2_answers = [f'aug-gen_is{i}' for i in range(8,11)]
        headers = ptq_labels + q_labels + p1_answers + p2_answers
        completed_tasks_writer.writerow(['task_id','background',*headers,
                                         'listening','part1_accuracy','time'])
        completed_tasks_writer.writerows(rows)


def _grade_pretest(completion_dict):
    """
    completion_dict must have keys 'ptq{1-5}'
    """
    key = pd.read_csv('human_evaluation/data/pre-test_key.csv')['answer']
    
    if np.sum([completion_dict[f'ptq{i}'] == key[i-1] for i in [1, 2, 3]]) < 3:
        return 1
    elif np.sum([completion_dict[f'ptq{i}'] == key[i-1] for i in [4, 5]]) < 2:
        return 2
    else: 
        return 3


def parse_completion(completion_json_path):
    """
    Arguments
        completion_json_path: path to json containing one completion

    Returns a dictionary of relevant information for a single completion with keys
    task_id,points_on_pretest,background,ptq{1-5},{1-10},time
    ex: parse_completion('human_evaluation/completions/1.json')
    """
    with open(completion_json_path) as f:
        c = json.load(f)

    # init
    completion = {}
    completion['task_id'] = c['id']
    
    # fill
    # assert len(c['completions']) == 1
    completion['time'] = c['completions'][0]['lead_time']
    results = c['completions'][0]['result']

    for res in results:
        if res['from_name'] == 'consent':
            continue
        completion[res['from_name']] = res['value']['choices'][0][0]

    completion['background'] = _grade_pretest(completion)

    return completion


def completion_dict_to_row(completion):
    """
    Arguments
        completion: a dictionary representing one completion
    
    Update completed_tasks.csv with the given completion
    """

    tasks_df = pd.read_csv(CSV['tasks'])
    pairs_df_dict = {}

    for x in ['a', 'b', 'c', 'd']:
        pairs_df_dict[x] = pd.read_csv(CSV[f'{x}_pairs'])
    
    task_id = completion['task_id']
    correct_ct = 0
    keys = []
    for question_id in range(1, NUM_PAIRS_PER_TASK + 1):
        pick = completion[f'q{question_id}']
        pair_id = tasks_df.at[task_id, f'q{question_id}']
        comparison_idx = pair_id.split('_')[0]
        idx = int(pair_id.split('_')[1])
        pair_row = pairs_df_dict[comparison_idx].iloc[idx]
        labels = IDX_TO_COMPARISON[comparison_idx]
        key = pair_row[f'{labels[0]}_is']
        keys.append(key)
        correct = True if pick == key else False
        if comparison_idx == 'd':
            listening = correct
        if comparison_idx in ['a', 'b', 'd'] and correct:
            correct_ct += 1
    
    acc = correct_ct / 7
    pretest_answers = [completion[f'ptq{i}'] for i in range(1, NUM_PRETEST_QUESTIONS + 1)]
    test_answers = [completion[f'q{i}'] for i in range(1, NUM_PAIRS_PER_TASK + 1)]
    row = [task_id, completion['background'], *pretest_answers, *test_answers, 
           *keys, listening, acc, completion['time']]

    return row


if __name__ == '__main__':
    main()