import numpy as np
from pypianoroll import Multitrack, Track, plot_pianoroll, read
from matplotlib import pyplot as plt
import pretty_midi

def plot_piano_roll(pr):
    
    fig, ax = plt.subplots()

    img = plot_pianoroll(ax, pr)
    
    plt.show()


def main():
    filepath = '.\\dataset\\midi\\generated.mid'
    pm = pretty_midi.PrettyMIDI(filepath)
    pr = pm.get_piano_roll()
    
    #plot_piano_roll(np.transpose(np.array(pr)))
    
if __name__ == "__main__":
    main()