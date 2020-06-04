import pandas as pd
import numpy as np


def get_threshold(data_file, column='grade', aggregate='min'):
    """
    Arguments
        threshold: one of 'min', 'max', 'median', f'{n}p'
    returns threshold to be used for augmentative generation
    """
    data = pd.read_csv(data_file)[column]
    
    if aggregate == 'min':
        return np.min(data)
    elif aggregate == 'max': 
        return np.max(data)
    elif aggregate == 'median':
        return np.median(data)
    elif aggregate[-1] == 'p':
        p = int(aggregate[:-1])
        return np.percentile(data, p)
