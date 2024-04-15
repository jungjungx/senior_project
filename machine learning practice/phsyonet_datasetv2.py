import os
import numpy as np
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load data from all .dat files
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.dat'):
                file_path = os.path.join(data_dir, file_name)
                data = np.load(file_path)  # Assuming data is stored in numpy format
                samples = [{'features': sample['features'], 'label': sample['label']} 
                           for sample in data['samples']]
                self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
