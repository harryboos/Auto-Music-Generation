from decoder import Decoder
import sys
from tensorflow.python import keras
import json
import tensorflow as tf
import tensorflow_probability as tfp
import random
from progress.bar import Bar


class MusicTransformerDecoder(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, loader_path=None):
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
        decoder, _ = self.Decoder(inputs, training=training, mask=lookup_mask)
        fc = self.fc(decoder)
        if training:
            return fc
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, training=True):
        look_ahead_mask = self.create_look_ahead_mask(self.max_seq)
        with tf.GradientTape() as tape:
            predictions = self.call(
                inputs=x, lookup_mask=look_ahead_mask, training=training
            )
            self.loss_value = self.loss(y, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        loss = tf.reduce_mean(self.loss_value)
        
        return [loss.numpy()]
    
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

    def generate(self, prior: list, length=2048):
        decode_array = prior
        decode_array = tf.constant([decode_array])
        
        for i in Bar('generating').iter(range(min(self.max_seq, length))):
            if decode_array.shape[1] >= self.max_seq:
                break
            
            look_ahead_mask = self.create_look_ahead_mask(decode_array.shape[1])

            result = self.call(decode_array, lookup_mask=look_ahead_mask, training=False)
            
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
