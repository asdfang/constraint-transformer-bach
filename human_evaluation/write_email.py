import pandas as pd
from helpers import CSV
import click

@click.command()
@click.option('--task_id', type=int, default=0)
def main(task_id):
    df = pd.read_csv(CSV['completed_tasks'])
    task_row = df[df['task_id'] == task_id]
    correct_ct = int(task_row['part1_accuracy'].values[0]*7)
    missed_questions = []
    for i in range(1, 8):
        if task_row[f'q{i}'].values[0] != task_row[f'bach_is{i}'].values[0]:
            missed_questions.append(i)
    missed_questions_str = ''
    for i, q in enumerate(missed_questions):
        if i == 0:
            missed_questions_str += f'{q}'
        else:
            missed_questions_str += f', {q}'
    email = f"Hi!\n\nYou received {correct_ct}/7. If you're curious, you can listen to the test again at http://cortex.cs.northwestern.edu:8200/?task_id={task_id}. You missed questions {missed_questions_str}.\n\nThanks!"
    
    print(email)


if __name__ == '__main__':
    main()