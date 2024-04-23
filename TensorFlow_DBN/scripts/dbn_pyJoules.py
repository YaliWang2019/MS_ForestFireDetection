import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from flask import Flask, render_template, request

def DBN_import_dataset(image_size):
    train_dir = './training'
    val_dir = './Test/Test'

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)
    train_dataset = data_gen.flow_from_directory(
        train_dir, class_mode='categorical', classes=['No_Fire', 'Fire'], 
        seed=123, shuffle=True, target_size=image_size, subset='training')
        
    val_dataset = data_gen.flow_from_directory(
        val_dir, class_mode = 'categorical', classes=['No_Fire', 'Fire'], 
        seed=123, shuffle=True, target_size=image_size, subset = 'validation')

    train_image, train_label = next(train_dataset)
    train_image = np.array(train_image)
    train_label = np.array(train_label)
    train_label = np.argmax(train_label, axis=1)

    val_image, val_label = next(val_dataset)
    val_image = np.array(val_image)
    val_label = np.array(val_label)
    val_label = np.argmax(val_label, axis=1)

    train_image = train_image.reshape(-1, 193548) / 255.0
    val_image = val_image.reshape(-1, 193548) / 255.0

    return train_image, train_label, val_image, val_label

def DBN_create_model(batch_size):
    # Define RBMs and logistic regression
    rbm1 = BernoulliRBM(n_components=256, learning_rate=0.001, n_iter=20, batch_size=batch_size, verbose=True)
    rbm2 = BernoulliRBM(n_components=256, learning_rate=0.001, n_iter=20, batch_size=batch_size, verbose=True)
    # can add more unsupervised RBMs layers here

    # Supervised logistic regression model to be layered on top
    logistic = LogisticRegression(solver='lbfgs', max_iter=10000, multi_class='multinomial')

    # Create a pipeline (stacking the layers)
    pipeline = Pipeline(steps=[('rbm1', rbm1), ('rbm2', rbm2), ('logistic', logistic)])

    return pipeline

def DBN_train_model(pipeline, x_train, y_train):
    # Train the model
    print('Start Deep Belief Network training...')
    pipeline.fit(x_train, y_train)
    return pipeline

def DBN_save_model(pipeline, model_filename):
    # Save the model to a file
    dump(pipeline, model_filename)
    print(f'Model saved to {model_filename}')

def DBN_import_model(model_filename):
    loaded_pipeline = load(model_filename)
    return loaded_pipeline
    print(f'Model loaded from {model_filename}')

def DBN_evaluate_model(pipeline, x_test, y_test):
    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy = {accuracy}")

def dbnPredict(image):
    image = np.array(image, dtype=np.float32)
    image = image.reshape(-1, 193548) / 255.0
    pred = dbModel.predict(image)
    return pred[0]

# Creating and importing Deep Belief Network
dbModel = DBN_import_model('dbn_2RBM_256.joblib')

# Define a list of class labels
class_labels = ['No Fire', 'Fire']
class_label = 0


def predict_result(img):
    # Grab Image
    imgToUse = cv2.imread(img, cv2.IMREAD_COLOR)

    # Resize Image
    resizedImage = cv2.resize(imgToUse, image_size, cv2.INTER_AREA)
    class_idx = dbnPredict(resizedImage)
    
    # Assign Label
    class_label = class_labels[class_idx]
    return class_label

# Define the directory path
dir_path = "testImages"

image_size = (254, 254)
class_labels = ['No Fire', 'Fire']
class_label = 0
csv_handler_time_and_prediction = CSVHandler('dbn_2RBM.csv')
csv_handler_energy = CSVHandler('dbn_2RBM_energy.csv')

def foo():
    # Open the csv file
    with open('dbn_2RBM.csv', 'w', newline='') as file:
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
with open('dbn_2RBM_energy.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["begin timestamp", "tag", "duration(seconds)", "whole CPU energy", "integrated GPU energy", "RAPL Core energy"])
csv_handler_energy.save_data()