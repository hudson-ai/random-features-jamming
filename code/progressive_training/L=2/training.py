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
def make_mnist_data(P=5000, d=20, P_test=None, random_state=42):
    # Load data from https://www.openml.org/d/554
    from sklearn.datasets import fetch_openml
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True)

    if P_test is None:
        P_test = P

    X = X_raw[:P+P_test]
    y = (2*(y_raw.astype(int) % 2) - 1)[:P+P_test].reshape(-1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=P_test, random_state=random_state)

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

def find_h(N, L, d, n=1, bias=False):
    # Modified from https://github.com/mariogeiger/nn_jamming/blob/master/constN.py
    '''
        For a network with: 
        
        d input dimensionality, 
        L layers, 
        N total parameters, 
        n final outputs,
        
        this finds the corresponding width h 
    '''
    assert L >= 1

    if L == 1:
        if bias:
            # solve : N = h*(d+1) + n*(h+1)
            h = (N - n) / (d + n + 1)
        else:
            # solve : N = h*d + n*h
            h = N/(d+n)
    else:
        if bias:
            # solve : N = h*(d+1) + (L-1)*h*(h+1) + n*(h+1)
            h = -(d+L+n - ((d+L+n)**2 + 4*(L-1)*(N-n))**.5)/(2*(L-1))
        else:
            # solve: N = h*d + (L-1)*h*h + n*h
            h = -((n+d) - ((n+d)**2 + 4*(L-1)*N)**.5)/(2*(L-1))
    return round(h)

def find_N(h, L, d, n=1, bias=False):
    '''
        For a network with: 
        
        d input dimensionality, 
        L layers,
        n final outputs,
        h width
        
        this finds the corresponding total number of parameters N
    '''
    
    if bias:
        return h*(d+1) + (L-1)*h*(h+1) + n*(h+1)
    else:
        return h*d + (L-1)*h*h + n*h

def make_model(N, L, d, loss='squared_hinge', optimizer='adam'):
    h = find_h(N, L, d, bias=False)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(d, name='inputs'),
        *[tf.keras.layers.Dense(h, use_bias=False, activation='tanh', name=f'intermediate_{i}') for i in range(L)],
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

    def __init__(self, filepath, save_every=1, logarithmic=False, batches_per_epoch=1):
        self.filepath = filepath
        self.save_every = save_every
        self.logarithmic = logarithmic
        self.last_checkpoint_epoch = 0
        self.epoch = 0
        self.batches_per_epoch = batches_per_epoch

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
            self._save_model(f'checkpoint_{epoch*self.batches_per_epoch}.ckpt', weights_only=True)

    def on_train_end(self, logs=None):
        self._save_model(f'checkpoint_{self.epoch*self.batches_per_epoch}.ckpt', weights_only=True)

    def _save_model(self, filename, weights_only=True):
        filepath = os.path.join(self.filepath, filename)
        if weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)


# Train!

P = 5000
d = 20
L = 2
print('Loading data.')
try: 
    X_train, X_test, y_train, y_test = pickle.load(open(f'mnist_P={P}_d={d}.pkl', 'rb'))
except:
    X_train, X_test, y_train, y_test = make_mnist_data(P=P, d=d)
    pickle.dump((X_train, X_test, y_train, y_test), open(f'mnist_P={P}_d={d}.pkl', 'wb'))
print('Done.')

num_Ns = 100
max_N = find_N(P, L, d) #I want to see the case where h = N_del, which may happen as high as h=P (VERY overparameterized)
Ns = np.unique(np.logspace(0, np.log10(max_N), num_Ns).astype(int))
for N in tqdm.tqdm(Ns):
    model_directory = f'models/P={P}_d={d}_N={N}_L={L}_mnist'

    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model = make_model(N, L, d)

    batch_size = min(1024, P//2)
    n_steps = int(1e6)
    n_saves = 100

    batches_per_epoch = P//batch_size
    epochs = n_steps//batches_per_epoch

    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    save_callback = SaveCallback(
                        model_directory, 
                        logarithmic=True, 
                        save_every=np.log10(epochs)/(n_saves), 
                        batches_per_epoch=batches_per_epoch
                    )

    print(f"Training. P = {P}, d = {d}, N = {N}, L = {L} (mnist)")
    result = model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, verbose=0, callbacks=[tqdm_callback, save_callback])
    
    pickle.dump(result.history, open(os.path.join(model_directory, 'result.pkl'), 'wb'))
