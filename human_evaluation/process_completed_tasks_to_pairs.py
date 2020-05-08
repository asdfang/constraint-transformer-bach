import sys
sys.path.insert(0, '../')

import pandas as pd
from helpers import CSV, IDX_TO_COMPARISON

tasks_df = pd.read_csv(CSV['tasks'])
completed_tasks_df = pd.read_csv(CSV['completed_tasks'])

pairs_df_dict = {}
completed_pairs_df_dict = {}
common_columns = ['task_id', 'question_id', 'background', 'pair_id']
for x in ['a', 'b', 'c', 'd']:
    pairs_df_dict[x] = pd.read_csv(CSV[f'{x}_pairs'])
    labels = IDX_TO_COMPARISON[x]
    completed_pairs_df_dict[x] = pd.DataFrame(columns=common_columns + [f'{labels[0]}_id', f'{labels[1]}_id', 
                                              f'{labels[0]}_grade', f'{labels[1]}_grade', 'correct'])

for i, task_row in completed_tasks_df.iterrows():
    task_id = int(task_row['task_id'])
    for question_id in range(1, 11):
        pick = task_row[question_id]
        pair_id = tasks_df.at[task_id, str(question_id)]
        comparison_idx = pair_id.split('_')[0]       # comparison type
        pair_row = pairs_df_dict[comparison_idx].iloc[int(pair_id.split('_')[1])]
        labels = IDX_TO_COMPARISON[comparison_idx]
        key = task_row[f'{labels[0]}_is{question_id}']
        correct = True if pick == key else False
        row = pd.DataFrame(
            [[task_id, question_id, task_row['background'], pair_id, pair_row[f'{labels[0]}_id'], 
              pair_row[f'{labels[1]}_id'], pair_row[f'{labels[0]}_grade'], pair_row[f'{labels[1]}_grade'], correct]],
            columns=completed_pairs_df_dict[comparison_idx].columns
        )
        completed_pairs_df_dict[comparison_idx] = completed_pairs_df_dict[comparison_idx].append(row)

for comparison_idx in completed_pairs_df_dict:
    completed_pairs_df_dict[comparison_idx].to_csv(CSV[f'completed_{comparison_idx}_pairs'], index=False)
