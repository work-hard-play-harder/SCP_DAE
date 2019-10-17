#!/usr/bin/env python
# coding: utf-8

# # CNN DAE Predictor
#
# This is a onehot coding vector based CNN denoising autoencoder for phenotype prediction. The performance is assayed on yeast genotype.

# In[108]:

import sys
import random
from time import time
import gzip

from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import keras
from keras.layers import Input, Embedding, Conv1D, Conv2D, MaxPooling1D, AveragePooling1D, MaxPooling2D, UpSampling1D, UpSampling2D
from keras.layers import Dropout, BatchNormalization, Activation, Flatten, Reshape, Dense
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras.regularizers import l1, l2, l1_l2
from keras.utils import to_categorical


# In[2]:
pheno_i = int(sys.argv[1])
print('pheno: {}'.format(pheno_i))
# num_kernels = 8
num_kernels = int(sys.argv[2])
print('number of kernels: {}'.format(num_kernels))
# kr = 1e-5
kr = float(sys.argv[3])
print('kr: {}'.format(kr))
# specify a seed for repeating the exactly results, start with 28213
seed = int(sys.argv[4])
np.random.seed(seed=seed)
print('seed: {}'.format(seed))
model_name = sys.argv[5]
print('model name : {}'.format(model_name))

models_dir = 'models/'
figs_dir = 'figs/'

print('loading model')
autoencoder = load_model(models_dir + model_name)
autoencoder.summary()
# In[62]:

# load data
genotype_file = 'genotype.csv'
genotype = pd.read_csv(genotype_file, sep='\t', index_col=0)
print('genotype_file shape:', genotype.shape)

# phenotype
phenotype_file = 'phenotype.csv'
multi_pheno = pd.read_csv(phenotype_file, sep=',', index_col=0)
print('phenotype_multi shape:', multi_pheno.shape)

# In[66]:
# take a small part to test code
# genotype
X = genotype
# X = genotype.iloc[0:1000:, 0:5000]
# single_pheno
Y = multi_pheno.iloc[:, pheno_i]
# Y = multi_pheno.iloc[0:1000, pheno_i]


# In[68]:
# # Add noise
# random missing masker
missing_perc = 0.1
nonmissing_ones = np.random.binomial(
    1, 1 - missing_perc, size=X.shape[0] * X.shape[1])
nonmissing_ones = nonmissing_ones.reshape(X.shape[0], X.shape[1])
nonmissing_ones, nonmissing_ones.shape

corrupted_X = X * nonmissing_ones
# corrupted_X.head()

# # Prepare data
# ## One-hot encoding
# In[69]:
X_onehot = to_categorical(X)
corrupted_X_onehot = to_categorical(corrupted_X)
# corrupted_X_onehot.shape

# normlization
scaled_Y = (Y - Y.min()) / (Y.max() - Y.min())

def detect_outliers(df):
    outlier_indices = []

    Q1 = np.percentile(df, 25)
    Q3 = np.percentile(df, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR

    outlier_indices = df[(df < Q1 - outlier_step) |
                         (df > Q3 + outlier_step)].index

    return outlier_indices


temp_Y = scaled_Y[~scaled_Y.isna()]
outliers_index = detect_outliers(temp_Y)
# set outliers as NAN
scaled_Y[outliers_index] = np.nan


# ## Split train and test
train_X, test_X, corrupted_train_X, corrupted_test_X, train_Y, test_Y = train_test_split(
    X_onehot, corrupted_X_onehot, scaled_Y, test_size=0.1)

# split df to train and valid
train_X, valid_X, corrupted_train_X, corrupted_valid_X, train_Y, valid_Y = train_test_split(
    train_X, corrupted_train_X, train_Y, test_size=0.1)


# move the gene loci with NA traits
train_X = train_X[~np.isnan(train_Y)]
train_Y = train_Y[~np.isnan(train_Y)]

valid_X = valid_X[~np.isnan(valid_Y)]
valid_Y = valid_Y[~np.isnan(valid_Y)]

test_X = test_X[~np.isnan(test_Y)]
test_Y = test_Y[~np.isnan(test_Y)]
# # Method

# ## Build autoencoder model

# hyperparameters
batch_size = 32
lr = 0.001
epochs = 1000

# conv 1D
feature_size = X.shape[1]
inChannel = 3
drop_prec = 0.25

# ## generate data
# Generates data for denoising autoencoder.

# # Build predicition model
predictor = Sequential()

# three ways to extract encoder model
# 1. create a new Sequential model (Recommendation)
for i in range(6):
    # make encoder layers in autoencoder non-trainable
    autoencoder.layers[i].trainable = False
    predictor.add(autoencoder.layers[i])

# flatten and connect with dense layer
# predictor.add(Flatten())

predictor.add(
    Dense(int(feature_size / 10), activation='relu', name='predictor_dense'))
predictor.add(Dropout(drop_prec, name='predictor_drop'))
predictor.add(Dense(1, name='predictor_output'))

# compile
predictor.compile(loss='mse', optimizer='adam', metrics=['mse'])
# summary
predictor.summary()


# In[64]:
# early stopping call back with val_loss monitor
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    # restore_best_weights=True
)

# # model checkpoint call back with val_acc monitor
# ModelCheckpoint = keras.callbacks.ModelCheckpoint(
#     'models/untrainable_predictor_checkpoint_model.{epoch:02d}-{val_mean_squared_error:.4f}.h5',
#     monitor='val_mean_squared_error',
#     verbose=0,
#     save_best_only=True,
#     save_weights_only=False,
#     mode='auto',
#     period=1)


# # In[65]:
epochs = 50

# predictor_train = predictor.fit_generator(
#     generator=train_generator,
#     validation_data=valid_generator,
#     epochs=epochs,
#     verbose=2,
#     callbacks=[EarlyStopping])

predictor_train = predictor.fit(
    train_X,
    train_Y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(valid_X, valid_Y),
    callbacks=[EarlyStopping])

# # trainable encoder layers

# In[68]:
for i in range(9):
    # make encoder layers in autoencoder non-trainable
    predictor.layers[i].trainable = True

predictor.compile(loss='mse', optimizer='adam', metrics=['mse'])

predictor.summary()


# In[69]:


# early stopping call back with val_loss monitor
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=0,
    mode='auto',
    baseline=None,
    # restore_best_weights=True
)

# model checkpoint call back with val_acc monitor
ModelCheckpoint = keras.callbacks.ModelCheckpoint(
    models_dir +
    'pheno{}_'.format(pheno_i) +
    'num_kernels{}_'.format(num_kernels) +
    'kr{}_'.format(kr) +
    'seed{}_'.format(seed) +
    'dae3_predictor_checkpoint.{epoch:03d}-{val_mean_squared_error:.4f}.h5',
    monitor='val_mean_squared_error',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)


epochs = 1000
predictor_train = predictor.fit(
    train_X,
    train_Y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(valid_X, valid_Y),
    callbacks=[EarlyStopping, ModelCheckpoint])

# predictor_train = predictor.fit_generator(
#     generator=train_generator,
#     validation_data=valid_generator,
#     epochs=epochs,
#     verbose=2,
#     callbacks=[EarlyStopping, ModelCheckpoint])


# In[71]:


# predictor.save('models/last_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
# autoencoder = load_model('models/best_autoencoder_model.85-0.9999.h5')


# # plot loss curve on validation data
# loss = predictor_train.history['loss']
# val_loss = predictor_train.history['val_loss']

# plt.figure()
# plt.plot(range(len(loss)), loss, 'bo', label='Training loss')
# plt.plot(range(len(val_loss)), val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig('predictor_loss_curve.png')

evaluate_data = predictor.evaluate(
    test_X, test_Y, verbose=2, batch_size=batch_size)
print(evaluate_data)

# # Prediction on test data

# predict
predict_data = predictor.predict(test_X)

# # Visualization

# In[74]:

plt.figure(figsize=(50, 20))
x = range(1, predict_data.shape[0] + 1)
y = test_Y.reshape(-1)
y_prime = predict_data.reshape(-1)
plt.plot(x, y, '-o', color='red')
plt.plot(x, y_prime, '-o', color='black')

plt.savefig(
    figs_dir +
    'pheno{}_'.format(pheno_i) +
    'num_kernels{}_'.format(num_kernels) +
    'kr{}_'.format(kr) +
    'seed{}_'.format(seed) +
    'result_visualization.png')
