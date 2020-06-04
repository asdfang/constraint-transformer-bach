import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import music21
from tqdm import tqdm

from Grader.helpers import get_threshold
from transformer_bach.utils import ensure_dir, parse_xml


def label_bars(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width()/2., 1.01*height,
            '%d' % int(height),
            ha='center', va='bottom'
        )


def read_training_data(data_file, feature='grade', threshold=None):
    """
    Returns dictionary with iteration as key and list of grades/distances as value
    """
    df = pd.read_csv(data_file)
    df = df.replace(float('-inf'), np.nan).dropna(subset=['grade'])
    num_epochs = np.max([int(it) for it in df['epoch']])
    data_dict = {}
    for it in range(num_epochs+1):
        grades = df.loc[df['epoch'] == it][feature]
        if threshold:
            grades = [x for x in grades if x > threshold]
        data_dict[it] = grades
    
    return data_dict


def get_good_mocks(model_dir):
    good_mock = []
    grades_df = pd.read_csv(f'{model_dir}/grades.csv')
    grades_df = grades_df.loc[grades_df['epoch'] <= 22]
    for row in grades_df.itertuples(index=False):
        if row.grade > get_threshold():
            epoch = row.epoch
            gen_id = row.gen_id
            score = parse_xml(f'{model_dir}/generations/{epoch}/{gen_id}.xml')
            good_mock.append(score)
    return good_mock
