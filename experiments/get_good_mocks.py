import sys
sys.path[0] += '/../'

import pandas as pd
import music21
from Grader.helpers import get_threshold

def main():
    aug_gen_model_dir = 'models/aug-gen_05-09_06:09'
    get_good_mocks(aug_gen_model_dir)


def get_good_mocks(model_dir):
    good_mock = []
    grades_df = pd.read_csv(f'{model_dir}/grades.csv')
    grades_df = grades_df.loc[grades_df['epoch'] <= 17]
    for row in grades_df.itertuples(index=False):
        if row.grade > get_threshold():
            epoch = row.epoch
            gen_id = row.gen_id
            score = music21.converter.parse(f'{model_dir}/generations/{epoch}/{gen_id}.mid')
            good_mock.append(score)
    return good_mock


if __name__ == '__main__':
    main()
