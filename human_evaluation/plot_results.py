import sys
sys.path[0] += '/../'

import numpy as np
import pandas as pd
from human_evaluation.helpers import IDX_TO_COMPARISON, CSV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

PLOT_COLORS = {'bach': 'lightcoral', 'aug-gen': 'lightskyblue', 'base': 'darkgray'}
PLOT_LABELS = {'bach': 'Bach', 'aug-gen': 'Aug. Gen.', 'base': 'Base'}

INCLUDE_WAVEFORM=False


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


def count_participants(include_waveform=INCLUDE_WAVEFORM):
    df = pd.read_csv(CSV['completed_tasks'])
    if include_waveform is False:
        df = discard_waveform_completions(df)
    # keep only listening participants
    # df = df.loc[df['listening'] == True]
    for background in [1, 2, 3]:
        print(f'Participants with background {background}: {len(df.loc[df["background"] == background].index)}')


def read_results(include_waveform=INCLUDE_WAVEFORM):
    """
    return a table with columns 1, 2, 3 grading_function, and rows a, b, c, showing the accuracy of each group
    for each type of comparison. accuracy is defined relative to Bach for comparisons a and b, and relative to
    aug gen for comparison c. 
    """
    results = pd.DataFrame(columns=[1, 2, 3, 'grading_function'], index=['a', 'b', 'c'])
    # for comparison in ['a', 'b', 'c']:
    for comparison in ['b']:
        # df = pd.read_csv(CSV[f'completed_{comparison}_pairs'])
        df = pd.read_csv(CSV[f'new_completed_{comparison}_pairs'])
        if include_waveform is False:
            df = discard_waveform_completions(df)
        # calculate human accuracy for each background
        for background in [1, 2, 3]:
            small_df = df.loc[df['background'] == background]
            if len(small_df.index) != 0:
                acc = np.sum(small_df['correct'])/len(small_df.index)
                results.at[comparison, background] = acc
        # calculate grading function accuracy
        labels = IDX_TO_COMPARISON[comparison]
        grading_acc = len([1 for a,b in zip(df[f'{labels[0]}_grade'], df[f'{labels[1]}_grade']) if a>b])/len(df.index)
        results.at[comparison, 'grading_function'] = grading_acc
    return results


def confusion():
    for x in ['a', 'b', 'c']:
        _confusion(x)


def _confusion(comparison_type, include_waveform=INCLUDE_WAVEFORM):
    """
    calculate the confusion matrix for a particular comparison type
    """
    df = pd.read_csv(CSV[f'completed_{comparison_type}_pairs'])
    if include_waveform is False:
        df = discard_waveform_completions(df)
    labels = IDX_TO_COMPARISON[comparison_type]
    print(f'{labels[0]} vs. {labels[1]}')
    # True when grading function picks Bach over Mock
    grading_function_pick = [a>b for a,b in zip(df[f'{labels[0]}_grade'], df[f'{labels[1]}_grade'])]
    # True when human picks Bach over Mock
    human_pick = df['correct']
    C = confusion_matrix(grading_function_pick, human_pick, labels=[True, False])
    print(C)


def plot_results(results):
    plt.figure()
    fig, axes = plt.subplots(1, 3, figsize=(8,5))
    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.9, left=0.1, right=0.83, bottom=0.17)
    PLOT_COLORS = {'bach': 'lightcoral', 'aug-gen': 'lightskyblue', 'base': 'gray'}
    plot_labels = {'bach': 'Bach', 'aug-gen': 'Aug. Gen.', 'base': 'Base'}
    width = 0.2
    xpos = np.arange(0, 4*width, width)

    # we're gonna do a
    plt.sca(axes[0])
    labels = IDX_TO_COMPARISON['a']
    plt.title('Bach vs. Aug. Gen.')
    plt.ylabel('Proportion identified as Bach')
    for i, bg in enumerate(list(results.columns)):
        acc = results.loc['a', bg]
        good = plt.bar(
            xpos[i],
            acc, 
            color=PLOT_COLORS[labels[0]], 
            alpha=0.8, 
            width=width-0.1, 
            label=plot_labels[labels[0]] if i==0 else None
        )
        bad = plt.bar(
            xpos[i],
            1-acc, 
            bottom=acc, 
            color=PLOT_COLORS[labels[1]], 
            alpha=0.8, 
            width=width-0.1,
            label=plot_labels[labels[1]] if i==0 else None
        )
    # now do b
    plt.sca(axes[1])
    labels = IDX_TO_COMPARISON['b']
    plt.title('Bach vs. Base')
    plt.ylabel('Proportion identified as Bach')
    for i, bg in enumerate(list(results.columns)):
        acc = results.loc['b', bg]
        good = plt.bar(
            xpos[i],
            acc, 
            color=PLOT_COLORS[labels[0]], 
            alpha=0.8, 
            width=width-0.1, 
            label=None
        )
        bad = plt.bar(
            xpos[i],
            1-acc, 
            bottom=acc, 
            color=PLOT_COLORS[labels[1]], 
            alpha=0.8, 
            width=width-0.1, 
            label=plot_labels[labels[1]] if i==0 else None
        )
    # now do c
    plt.sca(axes[2])
    labels = IDX_TO_COMPARISON['c']
    plt.title('Aug. Gen. vs. Base')
    plt.ylabel('Proportion identified as more like Bach')
    for i, bg in enumerate(list(results.columns)):
        acc = results.loc['c', bg]
        good = plt.bar(
            xpos[i],
            acc, 
            color=PLOT_COLORS[labels[0]], 
            alpha=0.8, 
            width=width-0.1, 
            label=None
        )
        bad = plt.bar(
            xpos[i],
            1-acc, 
            bottom=acc, 
            color=PLOT_COLORS[labels[1]], 
            alpha=0.8, 
            width=width-0.1, 
            label=None
        )
    for i in range(3):
        axes[i].grid(False)
        axes[i].axhline(y=0.5, xmin=0, xmax=1, dashes=(2,2))
        axes[i].set_xticks(xpos)
        axes[i].set_xticklabels(['Novice', 'Intermediate', 'Expert', 'Grading\nFunction'], rotation=45)
        axes[i].set_yticks(np.arange(0, 1, 0.25))
        axes[i].set_yticklabels([str(i) for i in np.arange(0, 1, 0.25)])

    handles, labels = [(a + b + c) for a, b, c in zip(*[axes[i].get_legend_handles_labels() for i in range(3)])]
    fig.legend(handles, labels, loc='center right')
    plt.savefig('plots/human_results.png')


if __name__ == '__main__':
    main()