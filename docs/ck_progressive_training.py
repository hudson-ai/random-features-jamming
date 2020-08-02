import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from tqdm import auto as tqdm
from functools import lru_cache

import pickle
import os


# Prepare data


def normalize(a, axis=None):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)

@lru_cache()
def make_mnist_data(P=5000, d=20, P_test=None):
    # Load data from https://www.openml.org/d/554
    from sklearn.datasets import fetch_openml
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True)

    if P_test is None:
        P_test = P

    X = X_raw[:P+P_test]
    y = (2*(y_raw.astype(int) % 2) - 1)[:P+P_test].reshape(-1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=P_test, random_state=42)

    from sklearn.decomposition import PCA
    pca = PCA(n_components = d)
    pca = pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # project to hyper-sphere of radius sqrt(d)
    X_train = np.sqrt(d) * normalize(X_train, axis=1)
    X_test = np.sqrt(d) * normalize(X_test, axis=1)

    return X_train, X_test, y_train, y_test


# Prepare network


def make_CK_model(N, loss='squared_hinge', optimizer='adam'):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(d, name='inputs'),
        tf.keras.layers.Dense(N, use_bias=False, activation='tanh', name='intermediate'),
        tf.keras.layers.Dense(1, use_bias=False, name='outputs')
    ])
    def accuracy(y_true, y_pred):
        return tf.reduce_mean(tf.cast(y_true*y_pred > 0, float))
    model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy])
    return model


class SaveCallback(tf.keras.callbacks.Callback):
    """
        Allows for logarithmically spaced model saving and checkpointing
    """

    def __init__(self, filepath, save_every=1, logarithmic=False):
        self.filepath = filepath
        self.save_every = save_every
        self.logarithmic = logarithmic
        self.last_checkpoint_epoch = 0
        self.epoch = 0

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self._save_model('MODEL', weights_only=False)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

        if self.logarithmic:
            save_this_epoch = (epoch >= self.last_checkpoint_epoch * 10**self.save_every)
        else:
            save_this_epoch = (epoch >= self.last_checkpoint_epoch + self.save_every)

        if save_this_epoch:
            self.last_checkpoint_epoch = epoch
            self._save_model(f'checkpoint_{epoch}.ckpt', weights_only=True)

    def on_train_end(self, logs=None):
        self._save_model(f'checkpoint_{self.epoch}.ckpt', weights_only=True)

    def _save_model(self, filename, weights_only=True):
        filepath = os.path.join(self.filepath, filename)
        if weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)


# Train!

P = 5000
d = 20
X_train, X_test, y_train, y_test = make_mnist_data(P=P, d=d)

for N in tqdm.tqdm(np.logspace(0, np.log10(1.1*P), 100)):
    N = int(N)

    model_directory = f'models/P_{P}-d_{d}-N_{N}_mnist'

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = make_CK_model(N)

    batch_size = min(1024, P//2)
    n_steps = int(5e3)
    n_saves = 100

    batches_per_epoch = P//batch_size
    epochs = n_steps//batches_per_epoch

    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    save_callback = SaveCallback(model_directory, logarithmic=True, save_every=np.log10(epochs)/(n_saves))

    print(f"Training. P = {P}, d = {d}, N = {N} (mnist)")
    result = model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, verbose=0, callbacks=[tqdm_callback, save_callback])

    pickle.dump(result, open(os.path.join(model_directory, 'result.pkl'), 'wb'))



# Post-Processing
import pickle
import pandas as pd
import os

model_dir = 'models'

Ns = []
losses = []
for subdir in os.listdir(model_dir):
     N = int(subdir.split('-')[2].split('_')[1])
     path = os.path.join(model_dir, subdir, 'result.pkl')
     try:
         result = pickle.load(open(path, 'rb'))
         loss = result['loss'][-1]
         Ns.append(N)
         losses.append(loss)
         # model = tf.saved_model.load(os.path.join(model_dir, subdir, 'MODEL'))

     except:
         continue
Ns = np.array(Ns)
losses = np.array(losses)
