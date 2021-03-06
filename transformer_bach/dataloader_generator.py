class DataloaderGenerator:
    """
    Base abstract class for data loader generators dataloaders
    """
    # def __init__(self, train_dataset, val_dataset):
    #     self.train_dataset = train_dataset
    #     self.val_dataset = val_dataset

    def __init__(self, dataset):
        self.dataset = dataset

    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
                    shuffle_val=True):
        raise NotImplementedError

