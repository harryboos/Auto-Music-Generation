import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import math
from attention import RelativeGlobalAttention
from pe import DynamicPositionEmbedding

class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = keras.layers.Dense(self.d_model // 2, activation=tf.nn.relu)
        self.FFN_suf = keras.layers.Dense(self.d_model)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False, **kwargs):
        attn_out, w = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w

class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)
        self.enc_layers = [DecoderLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
                           for i in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, mask=None, training=False):
        weights = []

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask, training=training)
            weights.append(w)
        return x, weights