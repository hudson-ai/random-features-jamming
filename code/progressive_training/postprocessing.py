# Post-Processing
import pickle
import pandas as pd 
import os

import tensorflow as tf

from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC as SVM

import tqdm.auto as tqdm

model_dir = 'models'
data_pkl = 'mnist.pkl'

model_names = os.listdir(model_dir)
X_train, X_test, y_train, y_test = pickle.load(open(data_pkl, 'rb'))
lamb = 1e-13

results = []

for model_name in tqdm.tqdm(model_names):
    model_path = os.path.join(model_dir, model_name)

    P = int(model_name.split('-')[0].split('_')[1])
    d = int(model_name.split('-')[1].split('_')[1])
    N = int(model_name.split('-')[2].split('_')[1])

    full_model = tf.keras.models.load_model(model_path+'/MODEL')
    weight_names = [p[:p.find('.index')] for p in os.listdir(model_path) if p.endswith('.ckpt.index')]

    for weight_name in tqdm.tqdm(weight_names, leave=False):
        step = int(weight_name.split('.')[0].split('_')[1])

        weight_path = os.path.join(model_path, weight_name)
        full_model.load_weights(weight_path)

        intermed_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(d, name='inputs'),
            full_model.get_layer('intermediate')
        ])

        train_features = intermed_model(X_train)
        test_features = intermed_model(X_test)

        C = 1/(N*lamb)

        svm = SVM(penalty='l2', loss='squared_hinge', dual=False, fit_intercept=False, C=C, )
        svm = svm.fit(train_features, y_train)

        y_train_hat = svm.decision_function(train_features)
        y_test_hat = svm.decision_function(test_features)

        result = {
            "P": P,
            "N": N,
            "d": d,
            "lambda": lamb,
            "C": C,
            'step': step,
            "y_train_hat": y_train_hat,
            "y_test_hat": y_test_hat

        }
        results.append(result)
        del intermed_model
    del full_model
pickle.dump(results, open('results.pkl', 'wb'))
