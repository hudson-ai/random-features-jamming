import pickle
import pandas as pd 
import os

import tensorflow as tf

from matplotlib import pyplot as plt
import tqdm.auto as tqdm

model_dir = 'models'
data_pkl = 'mnist.pkl'

model_names = os.listdir(model_dir)
X_train, X_test, y_train, y_test = pickle.load(open(data_pkl, 'rb'))

results = []

for model_name in tqdm.tqdm(model_names):
    model_path = os.path.join(model_dir, model_name)

    P = int(model_name.split('-')[0].split('_')[1])
    d = int(model_name.split('-')[1].split('_')[1])
    N = int(model_name.split('-')[2].split('_')[1])

    full_model = tf.keras.models.load_model(model_path+'/MODEL')
    weight_path = tf.train.latest_checkpoint(model_path)
    full_model.load_weights(weight_path)
    
    N_tilde = sum([np.prod(w.shape) for w in full_model.trainable_weights])
    step = int(weight_path.split('/')[-1].split('.')[0].split('_')[1])

    y_train_hat = full_model(X_train).numpy().reshape(-1)
    y_test_hat = full_model(X_test).numpy().reshape(-1)

    result = {
        "P": P,
        "N": N,
        "N_tilde": N_tilde,
        "d": d,
        "lambda": 0.,
        "C": np.inf,
        'step': step,
        "y_train_hat": y_train_hat,
        "y_test_hat": y_test_hat

    }
    results.append(result)
    del full_model
    
pickle.dump(results, open('results_end_to_end.pkl', 'wb'))
