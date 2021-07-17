# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:08:44 2021

@author: Rashmi S
"""

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten,Dense,Dropout,MaxPool2D,Conv2D



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv("C:/Users/Rashmi S/OneDrive/Desktop/Rash Spyder/sign_mnist_train.csv")
df_train.head()
df_train.describe()
df_test = pd.read_csv("C:/Users/Rashmi S/OneDrive/Desktop/Rash Spyder/sign_mnist_test.csv")
df_test.head()
df_train.info()
df_test.info()
train_label = df_train["label"]
test_label = df_test["label"]
plt.style.use("ggplot")
plt.figure(figsize =(9,5))
sns.countplot(x= df_train['label'],data = df_train)
plt.show()
df_train.drop("label",axis=1,inplace=True)
df_train.head()
df_test.drop("label",axis=1,inplace=True)
df_test.head(2)
x_train = df_train.values
x_train
x_train = x_train.reshape(-1,28,28,1)
x_test = df_test.values.reshape(-1,28,28,1)
lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)
plt.figure(figsize=(9,7))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.xlabel(np.argmax(y_train[i]))
    
plt.show()
train_datagen = ImageDataGenerator(rescale=(1./255),rotation_range = 30,
                                  width_shift_range = 0.2,height_shift_range =0.2,
                                  shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=(1./255))
#Model Building
from tensorflow.keras import Sequential
model = Sequential()
model.add(Conv2D(32,(3,3),padding = 'same',input_shape=(28,28,1),activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),padding = 'same',activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),padding = 'same',activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(24,activation="softmax"))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

checkpoint = ModelCheckpoint('sign_lan.h5',monitor ='val_acc',verbose=1,save_best_only=True,mode='max')
earlystop = EarlyStopping(monitor = 'val_acc',verbose=1,mode='max')
history = model.fit_generator(generator = train_datagen.flow(x_train,y_train,batch_size=32),
                              validation_data = val_datagen.flow(x_test,y_test),epochs=15,verbose=1)
import warnings
warnings.filterwarnings("ignore")
loss,acc = model.evaluate_generator(val_datagen.flow(x_test,y_test))
print(f"Accuracy: {acc*100}")
print(f"Loss: {loss}")
x_test = x_test/255.
y_pred = model.predict_classes(x_test)
y_te = np.argmax(y_test,axis=1)
y_te
from sklearn.metrics import accuracy_score
accuracy_score(y_te,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_te,y_pred))
plt.figure(figsize=(12,8))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i],cmap='gray')
    plt.xlabel(f"Actual: {y_te[i]}\n Predicted: {y_pred[i]}")
    
plt.tight_layout()
plt.show()
