from transformer_bach.DatasetManager.chorale_dataset import ChoraleBeatsDataset
from transformer_bach.DatasetManager.dataset_manager import DatasetManager
from transformer_bach.DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
from transformer_bach.dataloader_generator import DataloaderGenerator

subdivision = 4
num_voices = 4
metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=subdivision),
    KeyMetadata()
]

class BachDataloaderGenerator(DataloaderGenerator):
    def __init__(self, include_transpositions, sequences_size):
        dataset_manager = DatasetManager()

        chorale_dataset_kwargs = {
            'voice_ids':      [0, 1, 2, 3],
            'metadatas':      metadatas,
            'sequences_size': sequences_size,
            'subdivision':    subdivision,
            'include_transpositions':   include_transpositions,
        }

        dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
            name='bach_chorales_beats',
            **chorale_dataset_kwargs
        )
        super(BachDataloaderGenerator, self).__init__(dataset=dataset)

    def dataloaders(self, batch_size, num_workers=0, split=[0.85, 0.10], shuffle_train=True,
                    shuffle_val=False):
        # discard metadata
        # and put num_channels (num_voices) at the last dimension
        return [({'x': t[0].transpose(1, 2)} for t in dataloader)
                    for dataloader
                    in self.dataset.data_loaders(batch_size, 
                                                 num_workers=num_workers,
                                                 split=split,
                                                 shuffle_train=shuffle_train,
                                                 shuffle_val=shuffle_val)]

    def write(self, x, path):
        """

        :param x: (batch_size, num_events, num_channels)
        :return: list of music21.Score
        """
        score = self.dataset.tensor_to_score(x.transpose(1, 0))
        score.write('xml', f'{path}.xml')
        return

    def to_score(self, tensor_scores):
        scores = [self.dataset.tensor_to_score(tensor_score.transpose(1, 0))
                  for tensor_score in tensor_scores]
        return scores
