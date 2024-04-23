from flask import Flask, render_template, request

import cv2
import os
import csv
import time
import psutil
import tensorflow as tf
import numpy as np
import torch
from pyJoules.energy_meter import measure_energy
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplCoreDomain, RaplUncoreDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device import DeviceFactory
from pyJoules.energy_meter import EnergyMeter
import time

def segmentationPredict(image):
    ae_threshold = 66
    img_normalized = image.astype('float32') / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=0)
    reconstructed_img = aeModel.predict(img_normalized)
    
    reconstructed_img = reconstructed_img[0]
    
    # Reconstruct and save image
    reconstructed_img_color = np.clip(reconstructed_img * 255.0, 0, 255).astype('uint8')
    #recon_output_file_path = image_path[:-4] + '_reconstructed.png'
    #cv2.imwrite(recon_output_file_path, reconstructed_img_color)
    
    # Calculate Error
    mse_pix = np.mean(np.square(image - reconstructed_img_color), axis=-1)
    mse = np.mean(np.square(image - reconstructed_img_color))
    print(f"The mean squared error {mse}")
    
    # Checks if there is an anomoly detected, if so, saves an image with the largest anomoly with it boxed
    if mse > ae_threshold:
        print('Anomaly detected in the image!')
        max_mse_pixel = np.unravel_index(np.argmax(mse_pix), mse_pix.shape)
        print('Pixel with highest MSE:', max_mse_pixel)
        return 1
    else:
        print('No anomaly detected in the image.')
        return 0


def create_ae_model(input_shape):
    # Encoder
    inputs = tf.keras.layers.Input(shape=input_shape, name="Input")
    enc1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv1')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool1')(enc1)
    enc2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv2')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool2')(enc2)
    enc3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', name='Conv3')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='MaxPool3')(enc3)
    
    # Decoder
    dec1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', name='Conv4')(pool3)
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up1')(dec1)
    dec2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='Conv5')(up1)
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up2')(dec2)
    dec3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv6')(up2)
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up3')(dec3)
    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='valid', name='Output')(up3)
    
    # In progress to convert above decoder with these layers - more efficient and cleaner
    #tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT1'),
    #tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT2'),
    #tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides = (2, 2), activation='relu', padding='same', name='ConvT3'),
    #tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, activation='sigmoid', padding='same', name='Output')
    
    # Autoencoder model using all layers 
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Autoencoder')
    
    return autoencoder

def import_ae_model(model, path):
    model.load_weights(path)
    return model

def predict_result(img):
    # Grab Image
    imgToUse = cv2.imread(img, cv2.IMREAD_COLOR)

    # Resize Image
    resizedImage = cv2.resize(imgToUse, image_size, cv2.INTER_AREA)
    class_idx = segmentationPredict(resizedImage)
    
    # Assign Label
    class_label = class_labels[class_idx]
    return class_label

# Define the directory path
dir_path = "testImages"

image_size = (254, 254)
class_labels = ['No Fire', 'Fire']
class_label = 0
# Creating and importing Autoencoder
aeModel = create_ae_model(image_size + (3, ))
aeModel = import_ae_model(aeModel, f'ae_3En_3De.h5')
csv_handler_time_and_prediction = CSVHandler('ae_result.csv')
csv_handler_energy = CSVHandler('ae_energy_3En_3De.csv')

def foo():
    # Open the csv file
    with open('ae_result.csv', 'w', newline='') as file:
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
with open('ae_energy_3En_3De.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["begin timestamp", "tag", "duration(seconds)", "whole CPU energy", "integrated GPU energy", "RAPL Core energy"])
csv_handler_energy.save_data()