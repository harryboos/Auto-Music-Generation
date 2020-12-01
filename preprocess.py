import pickle
import os
import re
import sys
import hashlib
from progress.bar import Bar
from processor import encode_midi
import config
import random

def find_midi_files(root):
    for path, _, files in os.walk(root):
        for name in files:
            if name.endswith('mid'):
                yield os.path.join(path, name)


def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(find_midi_files(midi_root))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = encode_midi(path)
        except EOFError:
            print('EOF Error')

        with open('{}\\{}.pickle'.format(save_dir,path.split('\\')[-1]), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])

