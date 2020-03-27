import sys
sys.path.insert(0, '../')

from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

import click
import random
from DeepBach.helpers import ensure_dir
import os, shutil
from tqdm import tqdm
import numpy as np
from collections import defaultdict


@click.command()
@click.option('--include_transpositions', is_flag=True,
              help='whether to include transpositions (for dataset creation, or for pointing to the right folder at generation time)')
def main(include_transpositions):
    get_pairs(folders=[1,2], num_comparisons=10)


def get_pairs(folders=None, eval_dir='../generations/paired_evaluation/iters_1_2', num_comparisons=10):
    """
    Arguments:
        dataset: dataset of real Bach chorales
        model_ids: list of model IDs we are comparing
        eval_dir: folder for evaluation
        num_comparisons_per_model: number of generations for each model
    """
    answer = {}
    for iter_id, chorale_id in enumerate(np.random.choice(range(50), size=num_comparisons, replace=False)):
        pair_dir = os.path.join(eval_dir, f'{iter_id}')
        ensure_dir(pair_dir)
        labels = ['a','b']
        np.random.shuffle(labels)
        for i, folder in enumerate(folders):
            input_chorale = f'../generations/21/{folder}/c{chorale_id}.mid'
            output_folder = f'{pair_dir}/{labels[i]}_{chorale_id}.mid'
            if folder == 2:
                answer[iter_id] = labels[i]
            shutil.copy(input_chorale, output_folder)

    print(answer)


if __name__ == '__main__':
    main()
