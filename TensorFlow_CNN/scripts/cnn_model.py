import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def CNN_import_dataset(image_size):
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        '../Downloads/training', label_mode='binary', class_names=['No_Fire', 'Fire'],
        seed=123, shuffle=True, image_size=image_size)

    dataset_size = full_dataset.cardinality().numpy()
    train_size = int(0.8 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = int(0.10 * dataset_size)

    train_ds = full_dataset.take(train_size)
    test_ds = full_dataset.skip(train_size)
    validation_ds = test_ds.skip(val_size)
    test_ds = test_ds.take(test_size)

    return train_ds, validation_ds, test_ds

def CNN_create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(10, 5, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Conv2D(5, 4, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(2, 3, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(100, activation=tf.nn.relu))
    model.add(Dense(50, activation=tf.nn.relu))
    model.add(Dense(25, activation=tf.nn.relu))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def CNN_train(model, train_ds, validation_ds):
    model.fit(x=train_ds, validation_data=validation_ds, epochs=10)
    return model

def CNN_evaluate(model, dataset):
    evaluation_metrics = model.evaluate(x=dataset)
    print(f'Loss: {evaluation_metrics[0]}')
    print(f'Accuracy: {evaluation_metrics[1]}')

# setup
image_size = (254, 254)
train_ds, validation_ds, test_ds = CNN_import_dataset(image_size)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# create
model = CNN_create_model(image_size + (3, ))
model.summary()

# train
model = CNN_train(model, train_ds, validation_ds)

# evaluate
evaluation = CNN_evaluate(model, test_ds)

# save
model.save('cnn_or.h5')

