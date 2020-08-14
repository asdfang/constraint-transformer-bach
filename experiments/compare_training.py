import sys
sys.path[0] += '/../'

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from Grader.helpers import get_threshold
from experiments.helpers import read_training_data


PLOT_COLORS = {'aug-gen': 'lightskyblue', 'base': 'lightcoral', 'baseline': 'grey', 'bach': 'steelblue'}
PLOT_LABELS = {'aug-gen': 'Aug. Gen.', 'base': 'Base', 'baseline': 'Baseline'}

aug_gen_dir = 'models/aug-gen_06-04_23:10'
base_dir = 'models/base_06-02_06:55'
baseline_dir = 'models/base_06-03_00:00'

def main():
    dir_dict = {'aug-gen': aug_gen_dir, 'base': base_dir, 'baseline': baseline_dir}
    plot_median_grade_per_epoch(dir_dict, num_epochs=40)

def plot_median_grade_per_epoch(dir_dict, num_epochs):
    median_dict = defaultdict(lambda: [0]*num_epochs)
    for model_label, model_path in dir_dict.items():
        data_dict = read_training_data(data_file=f'{model_path}/grades.csv', feature='grade')
        for epoch, grades in data_dict.items():
            if epoch < num_epochs:
                median_dict[model_label][epoch] = np.median(grades)

    plt.figure()
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.grid(False)
    thres = get_threshold(
        data_file='experiments/ablations/reg_pe_no_oe/bach_grades.csv',
        column='grade',
        aggregate='75p',
    )
    plt.axhline(y=thres, dashes=(2,2), label='Lowest Bach\ngrade threshold', color=PLOT_COLORS['bach'])
    xlim = range(num_epochs)
    for model_label, median_grades in median_dict.items():
        plt.plot(xlim, median_grades[:num_epochs], label=PLOT_LABELS[model_label], color=PLOT_COLORS[model_label])
    plt.title('Median Grade of Generations During Training')
    ax.set_xticks([i+1 for i in xlim])
    ax.set_xticklabels([str(i) for i in xlim])
    for label in ax.get_xaxis().get_ticklabels()[1::2]:
        label.set_visible(False)
    # plt.legend(loc='right')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.2,0.5))
    plt.ylabel('Grade')
    plt.xlabel('Epoch')
    plt.savefig('plots/median_grades_per_epoch.png', bbox_inches='tight')


if __name__ == '__main__':
    main()