from torch.utils import data
import random

class IterDataset():
    def __init__(self, x):
        self.a = range(x, x+x)

    def __iter__(self):
        while True:
            for num in self.a:
                yield num

class DatasetForDataLoader(data.Dataset):
    def __init__(self, datasets, batch_size, count_datasets):
        self.datasets = datasets
        self.count_datasets = count_datasets
        self.batch_size = batch_size
    
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        data_idx = random.choice(range(self.count_datasets))
        return next(self.datasets[data_idx])

BATCH_SIZE = 8
lens1 = [4, 15, 36]
datasets1 = [iter(IterDataset(x)) for x in lens1]

lens2 = [80, 200]
datasets2 = [iter(IterDataset(x)) for x in lens2]

datasets_for_dataloader1 = DatasetForDataLoader(datasets1, BATCH_SIZE, len(lens1))
datasets_for_dataloader2 = DatasetForDataLoader(datasets2, BATCH_SIZE, len(lens2))

data_loader1 = data.DataLoader(datasets_for_dataloader1, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
# while True:
#     data = next(iter(data_loader))
#     print(data)

data_loader2 = data.DataLoader(datasets_for_dataloader2, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
# while True:
#     data = next(iter(data_loader2))
#     print(data)

for i in range(1000):
    idx = random.choice([1,2])
    if idx == 1:
        data = next(iter(data_loader1))
        print(data)
    else:
        data = next(iter(data_loader2))
        print(data)