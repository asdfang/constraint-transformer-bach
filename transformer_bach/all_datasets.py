import music21

from transformer_bach.DatasetManager.chorale_dataset import ChoraleDataset, ChoraleBeatsDataset
from transformer_bach.DatasetManager.helpers import ShortChoraleIteratorGen


def get_all_datasets():
    return {
        'bach_chorales':
            {
                'dataset_class_name': ChoraleDataset,
                'corpus_it_gen':      music21.corpus.chorales.Iterator
            },
        'bach_chorales_test':
            {
                'dataset_class_name': ChoraleDataset,
                'corpus_it_gen':      ShortChoraleIteratorGen()
            },
        'bach_chorales_beats':
            {
                'dataset_class_name': ChoraleBeatsDataset,
                'corpus_it_gen':      music21.corpus.chorales.Iterator
            },
        'bach_chorales_beats_test':
            {
                'dataset_class_name': ChoraleBeatsDataset,
                'corpus_it_gen':      ShortChoraleIteratorGen()
            },
    }
