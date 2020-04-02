import os
import torch
import music21

from transformer_bach.DatasetManager.music_dataset import MusicDataset
from transformer_bach.DatasetManager.chorale_dataset import ChoraleDataset, ChoraleBeatsDataset
from transformer_bach.DatasetManager.helpers import NextChoralesIteratorGen

# to use an existing dataset, add an entry in the all_datasets variable
# and specify its base class and which music21 objects it uses
# by giving an iterator over music21 scores
all_datasets = {
    'bach_chorales':
        {
            'dataset_class_name': ChoraleDataset,
            'corpus_it_gen':      music21.corpus.chorales.Iterator
        },
    'bach_chorales_beats':
        {
            'dataset_class_name': ChoraleBeatsDataset,
            'corpus_it_gen':      music21.corpus.chorales.Iterator
        },
}

class DatasetManager:
    def __init__(self):
        self.cache_dir = f'{os.path.expanduser("~")}/transformer-bach/data/dataset_cache'
        # create cache dir if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        self.all_datasets = all_datasets

    def add_dataset(self, 
                    dataset_name, 
                    dataset_class_name, 
                    corpus_it_gen):
        self.all_datasets[dataset_name] = {'dataset_class_name': dataset_class_name,
                                           'corpus_it_gen': corpus_it_gen}
    
    def get_all_datasets(self):
        return self.all_datasets

    def get_dataset(self, name: str, **dataset_kwargs) -> MusicDataset:
        all_datasets = self.get_all_datasets()
        
        if name in all_datasets:
            return self.load_if_exists_or_initialize_and_save(
                name=name,
                **all_datasets[name],
                **dataset_kwargs
            )
        else:
            print(f'Dataset with name {name} is not registered in all_datasets variable')

    def load_if_exists_or_initialize_and_save(self,
                                              dataset_class_name,
                                              corpus_it_gen,
                                              name,
                                              **kwargs):
        """

        :param dataset_class_name:
        :param corpus_it_gen:
        :param name:
        :param kwargs: parameters specific to an implementation
        of MusicDataset (ChoraleDataset for instance)
        :return:
        """
        kwargs.update(
            {'name':          name,
             'corpus_it_gen': corpus_it_gen,
             })
        dataset = dataset_class_name(**kwargs)
        if os.path.exists(dataset.filepath(self.cache_dir)):
            print(f'Loading {dataset.__repr__()} from {dataset.filepath(self.cache_dir)}')
            dataset = torch.load(dataset.filepath(self.cache_dir))
            print(f'(the corresponding TensorDataset is not loaded)')
        else:
            print(f'Creating {dataset.__repr__()}, '
                  f'both tensor dataset and parameters')
            # initialize and force the computation of the tensor_dataset
            # first remove the cached data if it exists
            if os.path.exists(dataset.tensor_dataset_filepath(self.cache_dir)):
                os.remove(dataset.tensor_dataset_filepath(self.cache_dir))
            # recompute dataset parameters and tensor_dataset
            # this saves the tensor_dataset in dataset.tensor_dataset_filepath
            tensor_dataset = dataset.get_tensor_dataset(self.cache_dir)
            # save all dataset parameters EXCEPT the tensor dataset
            # which is stored elsewhere
            dataset.tensor_dataset = None
            torch.save(dataset, dataset.filepath(cache_dir=self.cache_dir))
            print(f'{dataset.__repr__()} saved in {dataset.filepath(cache_dir=self.cache_dir)}')
            dataset.tensor_dataset = tensor_dataset
        return dataset
