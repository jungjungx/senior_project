import os
import torch
from torch.utils.data import Dataset
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Import pandas
#from phsyonet_datasetv2 import MultiPhysioNetDataset

input_size = 5000

class PhysioNetDataset(Dataset):
    def __init__(self, dat_dir, transform=None,max_length=None):
        self.dat_dir = dat_dir #the directory of the database
        self.transform = transform
        self.max_length = max_length  # Maximum length for signal truncation

        self.record_list = []
        #self.record_list = [file.split('.')[0] for file in os.listdir(dat_dir) if file.endswith('.dat')] #the .dat files 
        #edited to accept .csv as well
        for file in os.listdir(dat_dir): 
            if file.endswith('.dat') or file.endswith('.csv'):
                self.record_list.append(os.path.splitext(file)[0])
            #print("added file: "+file)

        # Define a mapping from label strings to numerical values
        self.label_mapping = {
            'emg_healthy': 0,
            'emg_myopathy': 1,
            'emg_neuropathy': 2
        }
            
    def __len__(self):
        return len(self.record_list)
        #return self.max_length * len(self.record_list)

    def tot_samp(self):
        return self.max_length * len(self.record_list)

    def __getitem__(self, idx):
        record_name = self.record_list[idx]

        # Read signal data and metadata from .dat and .hea files
        #signals, fields = wfdb.rdsamp(os.path.join(self.dat_dir, record_name))
        #record_metadata = wfdb.rdheader(os.path.join(self.dat_dir, record_name))

        # Check file format and read data accordingly
        if os.path.exists(os.path.join(self.dat_dir, record_name + '.dat')):
            signals, fields = wfdb.rdsamp(os.path.join(self.dat_dir, record_name))
        elif os.path.exists(os.path.join(self.dat_dir, record_name + '.csv')):
            df = pd.read_csv(os.path.join(self.dat_dir, record_name + '.csv'))
            signals = df.values
        else:
            # Handle the case where the file doesn't exist
            raise FileNotFoundError("File {} not found.".format(record_name))

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
        #emg_signal = torch.tensor(emg_signal).unsqueeze(0)

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

        print("Sample:", record_name,"of length:",len(emg_signal))
        #print("EMG Signal Length:", len(emg_signal))
        #print("Label:", label)

        #plt.plot(emg_signal)
        #plt.title("EMG Signal")
        #plt.xlabel("Time")
        #plt.ylabel("Amplitude")
        #plt.show()

        #print(sample)
        return sample

class SinglePhysioNetDataset(Dataset):
    def __init__(self,data,label,length=None):
        self.data = data
        self.label = label
        if length == None:
            length = 50000 #standard length if not given
        else:
            self.length = length

    def __getitem__(self, index):
        # Retrieve data and label at the given index
        emg_signal = self.data[index]

        #emg_signal = emg_signal.reshape(64, 5000)

        sample = {'emg': emg_signal, 'label': self.label}
        
        return sample

    def __len__(self):
        return self.length
        #return len(self.data)
    
    def __str__(self):
        #return f"EMG Data: {self.data}, Label: {self.label}"  
        return f"EMG Data: tensor(shape={self.data.shape}, Label: {self.label}, Size: {self.length}, Data: {self.data}"




if __name__ == "__main__":
    # Example usage:
    #C:\Users\Jakeeer\git\senior_project\machine learning practice\test_database
    dat_dir = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\test_database'  # Path to directory containing .dat files
    physionet_dataset = PhysioNetDataset(dat_dir)
    #print("Here are the files inside physionet_dataset: ")
    #for idx in range(len(physionet_dataset)):
    #    record_name = physionet_dataset.record_list[idx]
    #    print(record_name)
    sample = physionet_dataset[3]  # Uses __getitem__ defined in the separate file

    print("----------------testing singlePhysioNetDataset")
    #sphysionet_datasets = [] #array of physionet datasets
    #for idx in range(len(physionet_dataset)):
    #        sample = physionet_dataset[idx]
    #        sphysionet_datasets.append(SinglePhysioNetDataset(sample['emg'],sample['label'],input_size))
    #        print(sphysionet_datasets[idx])

    #healthy = physionet_dataset[0]
    #myopathy = physionet_dataset[1]
    #neuropathy = physionet_dataset[2]
    #physionet_datasetv2 = MultiPhysioNetDataset(healthy['emg'],myopathy['emg'],neuropathy['emg'])