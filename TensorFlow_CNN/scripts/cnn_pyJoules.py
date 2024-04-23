import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import sys
import os
import cv2
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pyJoules.energy_meter import measure_energy
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplCoreDomain, RaplUncoreDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device import DeviceFactory
from pyJoules.energy_meter import EnergyMeter
import time

def CNN_create_model(input_shape):
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(10, 5, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4,4)))
    #model.add(Conv2D(10, 5, input_shape=input_shape))
    model.add(Conv2D(5, 4, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3)))
    #model.add(Conv2D(10, 5, input_shape=input_shape))
    model.add(Conv2D(2, 3, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(10, 5, input_shape=input_shape))

    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(100, activation=tf.nn.relu))
    model.add(Dense(50, activation=tf.nn.relu))
    model.add(Dense(25, activation=tf.nn.relu))
    # model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])
    return model

def CNN_import_model(model, path):
    model.load_weights(path)
    return model

def cnnPredict(image):
    image = np.array(image, dtype=np.float32)
    pred = cnnModel.predict(np.expand_dims(image, axis=0))[0][0]

    return round(pred)

def predict_result(img):
    # Grab Image
    imgToUse = cv2.imread(img, cv2.IMREAD_COLOR)

    # Resize Image
    resizedImage = cv2.resize(imgToUse, image_size, cv2.INTER_AREA)
    # Predicting with CNN
    class_idx = cnnPredict(resizedImage)

    # Assign Label
    class_label = class_labels[class_idx]
    return class_label


# Define the directory path
dir_path = "test120"

# Define the image size
image_size = (254, 254)
class_labels = ['No Fire', 'Fire']
class_label = 0
# Creating and importing CNN
cnnModel = CNN_create_model(image_size + (3, ))
cnnModel = CNN_import_model(cnnModel, f'./models/cnn_m1.h5')

# Create csv handlers for each output file
csv_handler_time_and_prediction = CSVHandler('cnn_or_120_pred3.csv')
csv_handler_energy = CSVHandler('cnn_or_120_energy.csv')

def foo():
    # Open the csv file
    with open('cnn_or_120_pred3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["filename", "prediction", "time"])
        # Iterate over all files in the directory
        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Full path
                full_path = os.path.join(dir_path, filename)
                # Start timing
                start_time = time.time()
                # Make a prediction
                class_label_prediction = predict_result(full_path)
                # End timing
                end_time = time.time()
                # Calculate the time difference
                time_diff = end_time - start_time
                print(f"{filename} has {class_label_prediction}. Time consumed: {time_diff} seconds.")
                # Write the data to the csv file
                writer.writerow([filename, class_label_prediction, time_diff])

# Record energy consumption
with EnergyContext(handler=csv_handler_energy, domains=[RaplPackageDomain(0), RaplUncoreDomain(0), RaplCoreDomain(0)], start_tag='foo') as ctx:
    foo()

# Save the total time and energy consumption data to the CSV file
with open('cnn_or_120_energy.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["begin timestamp", "tag", "duration(seconds)", "whole CPU energy", "integrated GPU energy", "RAPL Core energy"])
csv_handler_energy.save_data()

