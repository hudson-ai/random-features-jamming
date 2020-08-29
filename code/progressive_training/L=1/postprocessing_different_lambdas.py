#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Post-Processing
import pickle
import pandas as pd 
import numpy as np
import os

import tensorflow as tf

from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC as SVM

import tqdm.auto as tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[84]:


model_dir = 'models'
model_names = os.listdir(model_dir)
X_train, X_test, y_train, y_test = pickle.load(open('mnist.pkl', 'rb'))
lambs = np.logspace(-15, 0, num=16)

results = []


# In[85]:


for model_name in tqdm.tqdm(model_names):
    model_path = os.path.join(model_dir, model_name)

    P = int(model_name.split('-')[0].split('_')[1])
    d = int(model_name.split('-')[1].split('_')[1])
    N = int(model_name.split('-')[2].split('_')[1])

    full_model = tf.keras.models.load_model(model_path+'/MODEL')
    weight_names = [p[:p.find('.index')] for p in os.listdir(model_path) if p.endswith('.ckpt.index')]
    weight_names = sorted(weight_names, key = lambda s:int(s.split('_')[1].split('.')[0]))

#     for weight_name in tqdm.tqdm(weight_names[-1:], leave=False):
    weight_name = weight_names[-1]
    for lamb in lambs:
        batches_per_epoch = 5
        step = int(weight_name.split('.')[0].split('_')[1])*batches_per_epoch

        weight_path = os.path.join(model_path, weight_name)
        full_model.load_weights(weight_path).expect_partial()

        intermed_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(d, name='inputs'),
            full_model.get_layer('intermediate')
        ])

        train_features = intermed_model(X_train)
        test_features = intermed_model(X_test)

        # Liblinear loss = .5*sum_{i=1}^N[w_i^2] + C*sum_{j=1}^P[y_j - sum_{i=1}^N w_i*x_ji]
        # Mei et al loss = (N*lam/d)*sum_{i=1}^N[w_i^2] + (1/P)*sum_{j=1}^P[y_j - sum_{i=1}^N w_i*x_ji]
        C = d/(2*N*P*lamb)
#             print(C)

        svm = SVM(penalty='l2', loss='squared_hinge', dual=False, fit_intercept=False, C=C)
        svm = svm.fit(train_features, y_train)

        y_train_hat = svm.decision_function(train_features)
        y_test_hat = svm.decision_function(test_features)

        result = {
            "P": P,
            "N": N,
            "d": d,
            "step": step,
            "lambda": lamb,
            "C": C,
            "y_train_hat": y_train_hat,
            "y_test_hat": y_test_hat

        }
        results.append(result)
        
pickle.dump(results, open('results_different_lambdas.pkl', 'wb'))

