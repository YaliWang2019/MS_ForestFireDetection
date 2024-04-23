import tensorflow as tf
from tensorflow import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def import_classification_datasets(image_size, batch_size):

    # Create an ImageDataGenerator to load the images
    image_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
    test_image_datagen = ImageDataGenerator(rescale=1/255)

    no_fire_train_dir   = './ae_training'
    no_fire_test_dir    = './ae_training'
    fire_train_dir      = './ae_training'
    fire_test_dir       = './Test/Test'

    ### No Fire Datasets ###
    no_fire_train_ds = image_datagen.flow_from_directory(
        no_fire_train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='training')

    no_fire_validation_ds = image_datagen.flow_from_directory(
        no_fire_test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='validation')

    no_fire_test_ds = test_image_datagen.flow_from_directory(
        no_fire_test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb')

    ### Fire Datasets ###
    fire_train_ds = image_datagen.flow_from_directory(
        fire_train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='training')

    fire_validation_ds = image_datagen.flow_from_directory(
        fire_train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb',
        subset='validation')

    fire_test_ds = test_image_datagen.flow_from_directory(
        fire_test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='input',
        color_mode='rgb')

    return no_fire_train_ds, no_fire_validation_ds, no_fire_test_ds, fire_train_ds, fire_validation_ds, fire_test_ds

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
    dec3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, activation='relu', padding='same', name='Conv6')(up2)
    up3 = tf.keras.layers.UpSampling2D(size=(2, 2), name='Up3')(dec3)

    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation='sigmoid', padding='valid', name='Output')(up3)

    # Autoencoder model using all layers
    autoencoder = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Autoencoder')

    return autoencoder

# Structural Similarity Index Measure loss function
def ssim_loss(y_true, y_pred):
    return 1 - image.ssim(y_true, y_pred, max_val=1.0)


def train(model, train_ds, validation_ds, epochs):
    model.fit(train_ds, validation_data=validation_ds, epochs=epochs)
    return model

def import_ae_model(model, path):
    model.load_weights(path)
    return model

def evaluate(model, dataset):
    model.evaluate(x=dataset)

image_size = (254, 254)
batch_size = 32
no_fire_train_ds, no_fire_validation_ds, no_fire_test_ds, fire_train_ds, fire_validation_ds, fire_test_ds = import_classification_datasets(image_size, batch_size)
image_shape = image_size + (3, )
model = create_ae_model(image_shape)
model.build((None, ) + image_shape)
optimizer = 'adam'
loss_function_name = 'ssim'
loss_function = ssim_loss
epochs = 5
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
model = train(model, no_fire_train_ds, no_fire_validation_ds, epochs)
model.save(f'ae_3En_3De.h5')

test_loss, test_acc = model.evaluate(no_fire_test_ds)
print('Test Loss: {:.2f}'.format(test_loss))
print('Test Accuracy: {:.2f}'.format(test_acc))

