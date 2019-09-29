# keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

# timestamp
from time import time

# numpy and sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# plots
import matplotlib.pyplot as plt


# method for building the neural network architecture
def get_classifier():
    model = Sequential()
    model.add(Conv2D(128, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(43))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, clipvalue=0.1),
                  metrics=['accuracy'])
    return model


# method for defining the training callbacks
def get_callbacks():
    return [TensorBoard(log_dir='logs/{}'.format(time())),
            ModelCheckpoint(filepath="nets/final.h5", monitor='val_acc',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]


# method for plotting the confusion matrix
def plot_confusion_matrix():
    model = load_model('nets/final.h5')
    y_pred = model.predict_generator(validation_generator)
    y_pred = np.argmax(y_pred, axis=-1)

    classes = validation_generator.classes

    cm = confusion_matrix(classes, y_pred)
    cr = classification_report(classes, y_pred)
    plt.imshow(cm, cmap='hot', interpolation='nearest')
    plt.show()
    print(cr)


# constants for training
img_width, img_height = 64, 64
train_data_dir = 'Train'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 800
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# augmentation used for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)

# train and validation generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

classifier = get_classifier()

classifier.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=get_callbacks())

plot_confusion_matrix()


