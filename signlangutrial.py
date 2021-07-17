# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:26:47 2021

@author: Rashmi S
"""
import numpy as np 
import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import zipfile
import random
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from shutil import copyfile
from sklearn.metrics import accuracy_score, classification_report
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.figure(figsize = (16,16))
img = plt.imread('C:/Users/Rashmi S/OneDrive/Desktop/Rash Spyder/amer_sign2.png')
_ = plt.imshow(img)
df_train = pd.read_csv('C:/Users/Rashmi S/OneDrive/Desktop/Rash Spyder/sign_mnist_train.csv')
df_train.head()
df_train.shape
df_train['label'].nunique()
df_test = pd.read_csv('C:/Users/Rashmi S/OneDrive/Desktop/Rash Spyder/sign_mnist_test.csv')
df_test.head()
df_test.shape
X_train = df_train.drop(['label'], axis = 1).values
X_test = df_test.drop(['label'], axis = 1).values

y_train = df_train['label']
y_test = df_test['label']
X_train
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
X_train
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)
y_test[:5]
X_train = X_train / 255
X_test = X_test / 255

plt.figure(figsize=(9,7))
for i in range(8):
    plt.subplot(2,3,i+1)
    plt.imshow(X_train[i],cmap='gray')
    plt.xlabel(np.argmax(y_train[i]))
    
plt.show()
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.1,zoom_range=0.1)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, y_train, batch_size= 128)

early_stopping = EarlyStopping( monitor = 'val-accuracy', min_delta=0.001, # minimium amount of change to count as an improvement
                                patience=10, # how many epochs to wait before stopping
                                restore_best_weights=True
                              )

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Sequential([ Conv2D(128 , (3,3)  , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)),
                         #BatchNormalization(),
                         MaxPool2D(2,2),
                         Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
                         Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
                         MaxPool2D(2,2),
                         Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'),
                         #BatchNormalization(),
                         MaxPool2D(2,2),
                         
                         Flatten(),
                         Dense(units = 512 , activation = 'relu'),
                         Dropout(0.2),
                         Dense(units = 24 , activation = 'softmax')
                       ])
    
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
                        
model.summary()

history = model.fit(train_generator, validation_data = (X_test, y_test), epochs=50, callbacks = [early_stopping])

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()

model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test),axis = 1) 
y_pred
y_true = np.argmax(y_test, axis =1)
y_true

print ( 'Model Accuracy = ', np.round(accuracy_score(y_true, y_pred), 2)*100)
print(classification_report(y_true, y_pred))
