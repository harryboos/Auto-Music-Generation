from model import MusicTransformerDecoder
import tensorflow as tf
from data import Data
import datetime
import sys

#hyperparameters
batch_size = 2
max_seq = 2048
epochs = 1000
load_path = None
save_path = 'result/result_model'
num_layer = 6
pad_token = 388
vocab_size = 391
embedding_dim = 256


# load data
dataset = Data('data')

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.math.logical_not(tf.math.equal(y_true, pad_token))
    mask = tf.cast(mask, tf.float32)

    y_true_vector = tf.one_hot(y_true, vocab_size)

    _loss = tf.nn.softmax_cross_entropy_with_logits(y_true_vector, y_pred)

    _loss *= mask

    return _loss

def main():
    learning_rate = CustomSchedule(embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

    mt = MusicTransformerDecoder(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            loader_path=load_path)
            
    mt.compile(optimizer=optimizer, loss=loss_function)

    for e in range(epochs):    
        for b in range(len(dataset.files) // batch_size):
            try:
                batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
            except:
                continue
            result_metrics = mt.train_on_batch(batch_x, batch_y, training=True)
            if b % 100 == 0:
                mt.save(save_path)
            
                print('\n====================================================')
                print('Epoch/Batch: {}/{}'.format(e, b))
                print('Train >>>> Loss: {:6.6}'.format(result_metrics[0]))
    
if __name__ == "__main__":
    main()
