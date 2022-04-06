# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:05:54 2022

@author: Engin
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras

vectorDataFile = "C:/SegmentGeneration/202112_c10_y_vector_data.txt"
inferenceResultFile = "C:/SegmentGeneration/202112_c10_y_inference.txt"
vectorLength = 748

# dataMaxSize = 40000
# indexesData = np.empty(shape=(dataMaxSize))
# yData = np.empty(shape=(dataMaxSize))
# xData = np.empty(shape=(dataMaxSize,vectorLength))

iData = 0

print("Loading model...\n")                
model = tf.keras.models.load_model("C:/SegmentGeneration/Model0.3495-0.8508")
print("Model loaded\n")                

print("Loading data and doing inference...\n")                
with open(vectorDataFile, "r") as r:
    with open(inferenceResultFile, "w") as w:    
        while True:
            line = r.readline()
            if not line:
                break
            
            cells = line.split(",")
    
            if len(cells) < 750:
                raise ValueError
    
            # indexesData[iData] = int(cells[0])
            # yData[iData] = int(cells[1])
            # xData[iData] = np.array(cells[2:]).astype(float).reshape(1,748)
            iData += 1
            
            prediction = model.predict(np.array(cells[2:]).astype(float).reshape(1,748))
            w.write(cells[0])
            w.write(",")
            w.write(str(round(prediction[0][0])))
            w.write("\n")
            
print("Data size:" + str(iData) + "\n")                
        



