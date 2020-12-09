from layers import *
import sys
from tensorflow.python import keras
import json
import tensorflow_probability as tfp
import random
from progress.bar import Bar
import pandas as pd
import matplotlib.pyplot as plt
from processor import Event, _merge_note, _event_seq2snote_seq


class MusicTransformerDecoder(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=1024, dropout=0.2, loader_path=None):
        super(MusicTransformerDecoder, self).__init__()

        if loader_path:
            self.load_config_file(loader_path)
        else:
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size

        self.Decoder = Decoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)

        self.fc = keras.layers.Dense(self.vocab_size, activation=None, name='output')

        if loader_path:
            self.load_ckpt_file(loader_path)

    def call(self, inputs, training=None, eval=None, lookup_mask=None):
        decoder, weights = self.Decoder(inputs, training=training, mask=lookup_mask)
        fc = self.fc(decoder)
        
        if training:
            return fc, weights
        else:
            return tf.nn.softmax(fc), weights

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, training=True):
        look_ahead_mask = self.create_look_ahead_mask(self.max_seq)
        with tf.GradientTape() as tape:
            predictions, weights = self.call(
                inputs=x, lookup_mask=look_ahead_mask, training=training
            )
            self.loss_value = self.loss(y, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        loss = tf.reduce_mean(self.loss_value)
        
        return [loss.numpy()], weights
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def save(self, filepath, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config.json'
        ckpt_path = filepath+'/ckpt'

        self.save_weights(ckpt_path, save_format='tf')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath + '/' + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath + '/' + ckpt_name
        try:
            self.load_weights(ckpt_path).expect_partial()
        except FileNotFoundError:
            print("[Warning] model will be initialized...")
    
    def get_config(self):
        config = {}
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        return config

    
    def generate(self, prior: list, length=1024, vis_enable=False):
        decode_array = prior
        decode_array = tf.constant([decode_array])
        sp = decode_array.shape[1]
        
        for i in Bar('generating').iter(range(min(self.max_seq, length))):
            if decode_array.shape[1] >= self.max_seq:
                break
            
            look_ahead_mask = self.create_look_ahead_mask(decode_array.shape[1])

            result, weights = self.call(decode_array, lookup_mask=look_ahead_mask, training=False)
            
            if vis_enable and decode_array.shape[1] == sp:
                #shape of atten weights: [num_layers, 1, num_heads, length, length]
                show_atten_heatmap(decode_array, weights[0][0])

            u = random.uniform(0, 1)
            if u > 1:
                result = tf.argmax(result[:, -1], -1)
                result = tf.cast(result, tf.int32)
                decode_array = tf.concat([decode_array, tf.expand_dims(result, -1)], -1)
            else:
                pdf = tfp.distributions.Categorical(probs=result[:, -1])
                result = pdf.sample(1)
                result = tf.transpose(result, (1, 0))
                result = tf.cast(result, tf.int32)
                decode_array = tf.concat([decode_array, result], -1)
            del look_ahead_mask
        decode_array = decode_array[0]

        return decode_array.numpy()

    def __load_config(self, config):
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']

'''
    utils function borrowed from hw4 to visualize the attention matrix    
'''   
def show_atten_heatmap(arr, weights):
    event_sequence = [Event.from_int(idx) for idx in arr[0].numpy()]
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = [note.value for note in snote_seq]
    #weights shape : num_heads, len, len
    fig, ax = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            weight = weights[i*2+j]
            _setup_atten_heatmap(ax[i][j], weight, note_seq)
    fig.tight_layout()
    plt.show()

def _setup_atten_heatmap(ax, weights, arr):
        """
        Create a heatmap from a numpy array and two lists of labels.
    
        This function derived from:
        https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

        ax - a "matplotlib.axes.Axes" instance to which the heatmap is plotted
        """
        
        data = np.array(weights)
        row_labels = col_labels = [str(label) for label in arr]
        
        cbarlabel="Attention Score"
        cbar_kw={}

        # Plot the heatmap
        im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)