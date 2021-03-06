# Post-Processing
import pickle
import pandas as pd 
import os
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC as SVM

from jax import hessian, jacobian, numpy as jnp

import tqdm.auto as tqdm

def normalize(a, axis=None):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)

model_dir = 'models'
model_names = os.listdir(model_dir)
lamb = 1e-13

results = []


for model_name in tqdm.tqdm(model_names):
    model_path = os.path.join(model_dir, model_name)
    
    P = int(model_name.split('_')[0].split('=')[1])
    d = int(model_name.split('_')[1].split('=')[1])
    N = int(model_name.split('_')[2].split('=')[1])
    L = int(model_name.split('_')[3].split('=')[1])

    X_train, X_test, y_train, y_test = pickle.load(open(f'mnist_P={P}_d={d}.pkl', 'rb'))

    full_model = tf.keras.models.load_model(model_path+'/MODEL')
    weight_names = [p[:p.find('.index')] for p in os.listdir(model_path) if p.endswith('.ckpt.index')]

    for weight_name in tqdm.tqdm(weight_names, leave=False):
        step = int(weight_name.split('.')[0].split('_')[1])

        weight_path = os.path.join(model_path, weight_name)
        full_model.load_weights(weight_path).expect_partial()

        intermed_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(d, name='inputs'),
            *[full_model.get_layer(f'intermediate_{i}') for i in range(L)]
        ])

        train_features = intermed_model(X_train).numpy()
        test_features = intermed_model(X_test).numpy()
        
        # Normalize features to ball of radius sqrt(h)
        h = train_features.shape[1]
        train_features = np.sqrt(h) * normalize(train_features, axis=1)
        test_features = np.sqrt(h) * normalize(test_features, axis=1)

        # Liblinear loss = .5*sum_{i=1}^N[w_i^2] + C*sum_{j=1}^P[y_j - sum_{i=1}^N w_i*x_ji]
        # Mei et al loss = (N*lam/d)*sum_{i=1}^N[w_i^2] + (1/P)*sum_{j=1}^P[y_j - sum_{i=1}^N w_i*x_ji]
        C = d/(2*h*P*lamb)

        svm = SVM(penalty='l2', loss='squared_hinge', dual=False, fit_intercept=False, C=C)
        svm = svm.fit(train_features, y_train)

        y_train_hat = svm.decision_function(train_features)
        y_test_hat = svm.decision_function(test_features)
        
        # Hessian
        force = lambda w: jnp.maximum(0, 1 - y_train*(train_features@w))
        U = lambda w: jnp.sum(1/2*force(w)**2)
        w = svm.coef_.ravel()
        H = hessian(U)(w)
        jac_del = jacobian(force)(w)
        H0 = jnp.einsum('ni,nj->ij', jac_del, jac_del)
        HP = H - H0
        
        eigs = np.linalg.eigvalsh(H)
        eigs0 = np.linalg.eigvalsh(H0)
        eigsP = np.linalg.eigvalsh(HP)

        result = {
            "P": P,
            "N": N,
            "d": d,
            "L": L,
            "h": h,
            "step": step,
            "lambda": lamb,
            "C": C,
            "y_train_hat": y_train_hat,
            "y_test_hat": y_test_hat,
            "eigs": eigs,
            "eigs0": eigs0,
            "eigsP": eigsP

        }
        results.append(result)
    del full_model #Garbage collection       
pickle.dump(results, open('results.pkl', 'wb'))
