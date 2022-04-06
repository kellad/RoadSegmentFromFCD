# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:22:20 2022

@author: Engin
"""
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras

vectorDataFile = "C:/SegmentGeneration/202112_c10_b_vector_data.txt"
valVectorDataFile = "C:/SegmentGeneration/202112_c10_y_vector_data.txt"
vectorLength = 748

trainDataMaxSize = 100000
indexesTrain = np.empty(shape=(trainDataMaxSize))
yTrain = np.empty(shape=(trainDataMaxSize))
xTrain = np.empty(shape=(trainDataMaxSize,vectorLength))

valDataMaxSize = 40000
indexesValidation = np.empty(shape=(valDataMaxSize))
yValidation = np.empty(shape=(valDataMaxSize))
xValidation = np.empty(shape=(valDataMaxSize,vectorLength))

validationRate = 10
#trainSize = 72000
iTrain = 0
iValidation = 0

from tensorflow.keras.callbacks import EarlyStopping

def scheduler(epoch, lr):
  if epoch < 200:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

print("Loading training data...\n")                
with open(vectorDataFile, "r") as r:
    while True:
        line = r.readline()
        if not line:
            break
        
        cells = line.split(",")

        if len(cells) < 750:
            raise ValueError

        #rnd = random.random() * validationRate
        
        if True:#rnd > 1:
            indexesTrain[iTrain] = int(cells[0])
            yTrain[iTrain] = int(cells[1])
            xTrain[iTrain] = np.array(cells[2:]).astype(float).reshape(1,748)
            iTrain += 1
        else:
            indexesValidation[iValidation] = int(cells[0])
            yValidation[iValidation] = int(cells[1])
            xValidation[iValidation] = np.array(cells[2:]).astype(float).reshape(1,748)
            iValidation += 1

print("Training data size:" + str(iTrain) + "\n")                

print("Loading validation data...\n")                
with open(valVectorDataFile, "r") as r:
    while True:
        line = r.readline()
        if not line:
            break
        
        cells = line.split(",")

        if len(cells) < 750:
            raise ValueError

        indexesValidation[iValidation] = int(cells[0])
        yValidation[iValidation] = int(cells[1])
        xValidation[iValidation] = np.array(cells[2:]).astype(float).reshape(1,748)
        iValidation += 1

print("Validation data size:" + str(iValidation) + "\n")                

#Trim data lists
xValidation = xValidation[:iValidation]
yValidation = yValidation[:iValidation]    
xTrain = xTrain[:iTrain]
yTrain = yTrain[:iTrain]

validationSize = len(xValidation)
trainSize = len(xTrain)

batchSize = round(trainSize/6)
trainDataset = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)).batch(batchSize)
validationDataset = tf.data.Dataset.from_tensor_slices((xValidation, yValidation)).batch(batchSize)

while True:

    model = keras.Sequential([                         
        # Hidden fully connected layer
        keras.layers.Dense(units=256, activation='relu', input_shape=(748,)),
        keras.layers.Dropout(rate=0.25),
        # Hidden fully connected layer
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(rate=0.25),
        # Hidden fully connected layer
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(rate=0.25),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(rate=0.25),
        # Output fully connected layer 
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy'])
    
    es_callback = EarlyStopping(monitor="val_accuracy", patience=200, restore_best_weights=True)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(
        trainDataset.repeat(),
        batch_size = batchSize,
        epochs=2000, 
        steps_per_epoch=trainSize / batchSize,
        validation_data=validationDataset.repeat(),
        validation_steps=validationSize/batchSize,
        callbacks=[es_callback,lr_callback]
    )
    
    max_acc = 0
    max_acc_i = 0
    for i in range(len(history.history['val_accuracy'])):
        if max_acc < history.history['val_accuracy'][i]:
            max_acc = history.history['val_accuracy'][i]
            max_acc_i = i
            
    loss = round(history.history['val_loss'][max_acc_i],4)
    acc = round(history.history['val_accuracy'][max_acc_i],4)
    if acc > 0.8275:
        model.save("C:/SegmentGeneration/Model" + str(loss) + '-' + str(acc))
    
    tf.keras.backend.clear_session()
