import tensorflow as tf
import keras
import cv2
import glob
import numpy as np
import random
from keras.layers import Dense, Flatten,Conv2D, Softmax, BatchNormalization, ReLU, Dropout, ZeroPadding2D, MaxPool2D
from keras import Sequential
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy,binary_crossentropy
from keras.activations import sigmoid, softmax

from keras.preprocessing.image import ImageDataGenerator



path = '/home/ashrafi/Documents/Shetu mam/final dataset/*/*.*'
classs = ['nv', 'akiec', 'bcc', 'vasc', 'df', 'mel', 'bkl']

td = []

for file in (glob.glob(path)):
    img = cv2.resize(cv2.imread(file), (32, 32))
    deg_name = file.split('/')[-2]
    label=classs.index(deg_name)
    td.append([img, label])
random.shuffle(td)

train_img = []
train_label = []

for img, la in td:
    train_img.append(img)
    train_label.append(la)

train_img = np.asarray(train_img).reshape(-1, 32, 32, 3)



model= Sequential()
model.add(Conv2D(32, (3,3),activation='relu', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


model.fit(train_img,train_label,epochs=40, batch_size=32,
          validation_split=0.1)

model.save('/home/ashrafi/Documents/Shetu mam/acc7105shetu.h5')