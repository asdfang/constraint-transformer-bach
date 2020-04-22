import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def label_bars(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')


def read_update_data(data_file, feature, threshold=None):
    """
    Returns dictionary with iteration as key and list of grades/distances as value
    """
    df = pd.read_csv(data_file)
    df = df.replace(float('-inf'), np.nan).dropna(subset=['grade'])
    update_iterations = np.max([int(it) for it in df['iter']])
    data_dict = {}
    for it in range(update_iterations + 1):
        grades = df.loc[df['iter'] == it][feature]
        if threshold:
            grades = [x for x in grades if x > threshold]
        data_dict[it+1] = grades
    
    return data_dict