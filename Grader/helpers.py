import pandas as pd
import numpy as np


def get_threshold(data_file='results/bach_grades.csv', feature='grade'):
    data = pd.read_csv(data_file)[feature]
    return np.min(data) if feature == 'grade' else np.max(data)
