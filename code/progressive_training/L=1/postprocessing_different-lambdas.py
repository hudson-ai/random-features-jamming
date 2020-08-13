#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Post-Processing
import pickle
import pandas as pd 
import os

import tensorflow as tf

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.svm import LinearSVC as SVM

import tqdm as tqdm

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[97]:


from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    'Mei2019', 
    np.array([
        (243, 232, 29),
        (245, 173, 47),
        (140, 193, 53),
        (50,  191, 133),
        (23,  167, 198),
        (36,  123, 235),
        (53,  69,  252),
        (52,  27,  203)
    ])/255., 
    N=256
)

# cmap = cc.m_bmy

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
fig = plt.figure(figsize=(6,.5))
img = plt.imshow(gradient, aspect='auto', cmap=cmap)
title = plt.title('Colormap stolen from Mei2019')

norm=mcolors.LogNorm()


# In[9]:


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[84]:


model_dir = 'models'
model_names = os.listdir(model_dir)
X_train, X_test, y_train, y_test = pickle.load(open('mnist.pkl', 'rb'))
lambs = np.logspace(-15, -3, num=5)

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
        
# pickle.dump(results, open('results.pkl', 'wb'))


# In[90]:


result_df = pd.DataFrame(results)


# In[91]:


result_df.head()


# In[92]:


force = lambda y,f: 1 - y*f
loss = lambda y,f: np.mean(np.maximum(0, force(y,f))**2, -1)
N_del = lambda y,f: np.sum(force(y,f) >= 0, -1)

result_df['test_loss'] = result_df.y_test_hat.apply(lambda f: loss(y_test, f))
result_df['train_loss'] = result_df.y_train_hat.apply(lambda f: loss(y_train, f))
result_df['N_del'] = result_df.y_train_hat.apply(lambda f: N_del(y_train, f))

result_df['N/P'] = result_df['N']/result_df['P']
result_df['P/N'] = result_df['P']/result_df['N']
result_df['N_del/P'] = result_df['N_del']/result_df['P']
result_df['N_del/N'] = result_df['N_del']/result_df['N']


# In[101]:


data = result_df.sort_values('lambda', ascending=False)
plt.scatter(data['N_del/N'], data['test_loss'], c=data['lambda'], alpha=.7, cmap=cmap, norm=norm)
plt.yscale('log')
plt.xscale('log')
plt.colorbar(label=r'$\lambda$ (regularization)')


# In[ ]:


for lamb in result_df['lambda'].unique():
    data = 
    plt.scatter(data['N_del/N'], data['test_loss'], c=data['lambda'], alpha=.7, cmap=cmap, norm=norm)
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar(label=r'$\lambda$ (regularization)')


# In[96]:


min(result_df['lambda'])


# In[99]:





# In[ ]:




