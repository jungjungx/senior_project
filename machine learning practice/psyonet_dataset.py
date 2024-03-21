import os
import torch
from torch.utils.data import Dataset
import wfdb
import matplotlib.pyplot as plt
import numpy as np

class PhysioNetDataset(Dataset):
    def __init__(self, dat_dir, transform=None,max_length=None):
        self.dat_dir = dat_dir #the directory of the database
        self.transform = transform
        self.record_list = [file.split('.')[0] for file in os.listdir(dat_dir) if file.endswith('.dat')] #the .dat files
        self.max_length = max_length  # Maximum length for signal truncation

        # Define a mapping from label strings to numerical values
        self.label_mapping = {
            'emg_healthy': 0,
            'emg_myopathy': 1,
            'emg_neuropathy': 2
        }
            

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        record_name = self.record_list[idx]

        # Read signal data and metadata from .dat and .hea files
        signals, fields = wfdb.rdsamp(os.path.join(self.dat_dir, record_name))
        record_metadata = wfdb.rdheader(os.path.join(self.dat_dir, record_name))

        # Assuming EMG signal is the first channel
        emg_signal = signals[:, 0]

        # Truncate or pad the EMG signal to the specified maximum length
        if self.max_length is not None:
            if len(emg_signal) > self.max_length:
                emg_signal = emg_signal[:self.max_length]  # Truncate if longer
            elif len(emg_signal) < self.max_length:
                pad_length = self.max_length - len(emg_signal)
                emg_signal = np.pad(emg_signal, (0, pad_length), 'constant')  # Pad with zeros if shorter

        # Convert NumPy array to torch tensor
        emg_signal = torch.tensor(emg_signal)

        # label should store the type of classification
        #label = '\n'.join(record_metadata.comments)
        #label should instead store the numerical value
        if(record_name == "emg_healthy"):
            label = 0
        elif(record_name == "emg_myopathy"):
            label = 1
        elif(record_name == "emg_neuropathy"):
            label = 2
        else:
            label = -1

        sample = {'emg': emg_signal, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        print("Sample:", record_name)
        print("EMG Signal Length:", len(emg_signal))
        print("Label:", label)

        #plt.plot(emg_signal)
        #plt.title("EMG Signal")
        #plt.xlabel("Time")
        #plt.ylabel("Amplitude")
        #plt.show()

        #print(sample)
        return sample


if __name__ == "__main__":
    # Example usage:
    dat_dir = 'C:\\Users\\Jakeeer\\Desktop\\Senior Project\\database'  # Path to directory containing .dat files
    physionet_dataset = PhysioNetDataset(dat_dir)
    sample = physionet_dataset[0]  # Uses __getitem__ defined in the separate file
