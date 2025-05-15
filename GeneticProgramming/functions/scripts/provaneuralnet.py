import numpy as np
import pandas as pd
from itertools import product
import math
import statsmodels.api as sm
import dill as pickle
import sympy as sy
import matplotlib.pyplot as plt
from configs.config_Nusselt import config
import functions.helpers as h
import itertools as it
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split


config = config()

db = pd.read_excel(config['userdata']['db_filename'],
                   header=0)
db = h.add_noise_to_db(db, 0)
Xtrain, Xtest = train_test_split(db,test_size=0.15)

ytrain = Xtrain.loc[:,config['userdata']['Target_name']].values
ytrain = ytrain/np.max(ytrain)

ytest = Xtest.loc[:,config['userdata']['Target_name']].values
ytest = ytest/np.max(ytest)
    
Xtrain = Xtrain.loc[:,['Re','Pr']].values
Xtrain = Xtrain/np.max(Xtrain, axis=0)
Xtest = Xtest.loc[:,['Re','Pr']].values
Xtest = Xtest/np.max(Xtest, axis=0)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=None)

history = model.fit(Xtrain, ytrain, epochs = 20, validation_split = 0.1)
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')

plt.figure()
plt.scatter(model.predict(Xtest), ytest)