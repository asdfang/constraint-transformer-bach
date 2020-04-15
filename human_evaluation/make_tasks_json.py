import pandas as pd
import json
from tqdm import tqdm
from helpers import NUM_PAIRS_PER_TASK

# TODO: obfuscate these directory names
BACH_DIR = 'http://cortex.cs.northwestern.edu:8200/static/audio/bach_chorales_piano/'
MOCK_DIRS = {
    'base': 'http://cortex.cs.northwestern.edu:8200/static/audio/base_mocks_piano/',
    'a-gen': '',
    'a-bach': '',
}
TASKS_JSON_PATH = '../../label-studio/label_studio/tbach-eval/tasks.json'

with open('answer_key.csv', 'r') as fin:
    answer_key = pd.read_csv(fin)
answer_key.set_index('pair_id')

with open ('tasks.csv', 'r') as fin:
    tasks = pd.read_csv(fin)
tasks.set_index('task_id')


all_tasks_json = {}
for index, task_row in tqdm(tasks.iterrows()):
    task_json = {}
    data = {}
    task_id = int(task_row['task_id'])

    for i in range(NUM_PAIRS_PER_TASK):
        pair_row = answer_key.iloc[task_row[i+1]]
        if pair_row['bach_is'] == 'a':
            data['url'+str(i)+'a'] = BACH_DIR + str(pair_row['bach_id']) + '.mp3'
            data['url'+str(i)+'b'] = MOCK_DIRS[pair_row['mock_model']] + str(pair_row['mock_id']) + '.mp3'
        else:
            data['url'+str(i)+'a'] = MOCK_DIRS[pair_row['mock_model']] + str(pair_row['mock_id']) + '.mp3'
            data['url'+str(i)+'b'] = BACH_DIR + str(pair_row['bach_id']) + '.mp3'
    
    task_json['id'] = task_id
    task_json['data'] = data
    all_tasks_json[str(task_id)] = task_json

with open(TASKS_JSON_PATH, 'w') as f:
    s = json.dumps(all_tasks_json, indent=4)
    f.write(s)
    f.close()