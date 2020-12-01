import random
import os
import pickle
from tensorflow.python import keras
import numpy as np
import midi_processor.processor as processor



class Data:
    def __init__(self, dir_path):
        self.files = list(self.find_pickle_files(dir_path))
        self.file_dict = {
            'train': self.files[:int(len(self.files) * 0.8)],
            'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
            'test': self.files[int(len(self.files) * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        self.pad_token = processor.RANGE_NOTE_ON + processor.RANGE_NOTE_OFF + processor.RANGE_TIME_SHIFT + processor.RANGE_VEL
        pass

    def batch(self, batch_size, length, mode='train'):

        batch_files = random.sample(self.file_dict[mode], k=batch_size)

        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return np.array(batch_data)  # batch_size, seq_len
    
    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y
    
    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                data = np.append(data, self.pad_token+2)
                while len(data) < max_length:
                    data = np.append(data, self.pad_token)
        return data

    def find_pickle_files(self, root):
    
        for path, _, files in os.walk(root):
            for name in files:
                if name.endswith('pickle'):
                    yield os.path.join(path, name)

