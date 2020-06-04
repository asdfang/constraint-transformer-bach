import sys
sys.path[0] += '/../'

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from Grader.helpers import get_threshold
from experiments.helpers import read_training_data


PLOT_COLORS = {'bach': 'lightcoral', 'aug-gen': 'lightskyblue', 'base': 'darkgray'}
PLOT_LABELS = {'bach': 'Bach', 'aug-gen': 'Aug. Gen.', 'base': 'Base'}

aug_gen_dir = 'models/aug-gen_05-09_06:09'
base_dir = 'models/base_05-10_07:20'

def main():
    dir_dict = {'aug-gen': aug_gen_dir, 'base': base_dir}
    plot_median_grade_per_epoch(dir_dict, num_epochs=20)

def plot_median_grade_per_epoch(dir_dict, num_epochs):
    median_dict = defaultdict(list)
    for model_label, model_path in dir_dict.items():
        data_dict = read_training_data(data_file=f'{model_path}/grades.csv', feature='grade', threshold=-1000)
        for epoch, grades in data_dict.items():
            median_dict[model_label].append(np.median(grades))

    plt.figure()
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    thres = get_threshold()
    plt.axhline(y=thres, dashes=(2,2), label='Lowest Bach\ngrade threshold', color=PLOT_COLORS['bach'])
    xlim = range(num_epochs)
    for model_label, median_grades in median_dict.items():
        plt.plot(xlim, median_grades[:num_epochs], label=PLOT_LABELS[model_label], color=PLOT_COLORS[model_label])
    plt.title('Median Grade of Generations During Training')
    plt.xticks(xlim)
    plt.xlim([0, num_epochs])
    plt.legend(loc='lower right')
    plt.ylabel('Grade')
    plt.xlabel('Epoch')
    plt.savefig('plots/median_grades_per_epoch.png')


if __name__ == '__main__':
    main()