import sys
sys.path[0] += '/../'

import numpy as np
import pandas as pd
from human_evaluation.helpers import IDX_TO_COMPARISON, CSV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# PLOT_COLORS = {'bach': 'gray', 'base': 'lightskyblue'}
PLOT_LABELS = {'bach': 'Bach\nchorales', 'base': 'Generated\nchorales'}


def main():
    # count_participants()
    results = read_results()
    print(results)
    plot_results(results)
    # confusion()


def discard_waveform_completions(df):
    waveform_completions = list(pd.read_csv(CSV['waveform_completions'])['waveform_task_id'])
    df = df.loc[(~df['task_id'].isin(waveform_completions)) | (df['background'] == 3)]
    return df


def count_participants():
    df = pd.read_csv(CSV['new_completed_pairs'])
    df = discard_waveform_completions(df)
    for background in [1, 2, 3]:
        print(f'Participants with background {background}: {len(df.loc[df["background"] == background].index)//3}')


def read_results():
    """
    return a table with columns 1, 2, 3 grading_function, and rows a, b, c, showing the accuracy of each group
    for each type of comparison. accuracy is defined relative to Bach for comparisons a and b, and relative to
    aug gen for comparison c. 
    """
    results = {k: 0 for k in [1, 2, 3, 'grading_function']}
    df = pd.read_csv(CSV[f'new_completed_pairs'])
    df = discard_waveform_completions(df)
    df.to_csv('human_evaluation/new_grades_data/no_waveform_completed_pairs.csv', index=False)
    # calculate human accuracy for each background
    for background in [1, 2, 3]:
        small_df = df.loc[df['background'] == background]
        if len(small_df.index) != 0:
            acc = np.sum(small_df['correct'])/len(small_df.index)
            results[background] = acc
    # calculate grading function accuracy
    print(df['bach_grade'])
    grading_acc = len([1 for a,b in zip(df['bach_grade'], df['base_grade']) if a<b])/len(df.index)
    results['grading_function'] = grading_acc
    return results


def confusion():
    """
    calculate the confusion matrix for a particular comparison type
    """
    df = pd.read_csv(CSV[f'new_completed_pairs'])
    df = discard_waveform_completions(df)
    print(f'Bach vs. base')
    # True when grading function picks Bach over Mock
    grading_function_pick = [a<b for a,b in zip(df[f'bach_grade'], df[f'base_grade'])]
    # True when human picks Bach over Mock
    human_pick = df['correct']
    C = confusion_matrix(grading_function_pick, human_pick, labels=[True, False])
    print(C)


def plot_results(results):
    plt.figure()
    fig, ax = plt.subplots(figsize=(7,1.5))
    plt.style.use('seaborn-whitegrid')
    height = 0.05
    ypos = np.arange(4*height, 0, -height)

    # now do b
    plt.title('Proportion Identified as Bach')
    acc = list(results.values())
    ax.barh(
        ypos,
        acc,
        alpha=0.5,
        height=height-0.01,
        label=PLOT_LABELS['bach'],
    )
    ax.barh(
        ypos,
        [1-a for a in acc],
        left=acc,
        alpha=0.5,
        height=height-0.01,
        label=PLOT_LABELS['base'],
    )
    
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.axvline(x=0.5, ymin=0, ymax=1, dashes=(2,2))
    ax.set_yticks(ypos)
    ax.set_yticklabels(['Novice', 'Intermediate', 'Expert', 'Grading\nFunction'])
    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_xticklabels(['0.0', '0.25', '0.5\n(chance)','0.75','1.0'])

    fig.legend(loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, left=0.16, right=0.81, bottom=0.27)
    plt.savefig('plots/human_results.png')


if __name__ == '__main__':
    main()