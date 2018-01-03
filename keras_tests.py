from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

batch_size = 128
num_classes = 10
epochs = 15
img_rows, img_cols = 28, 28
model_file = 'models/keras_mnist_cnn.h5'

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# noinspection PyBroadException
try:
    print("Loading existing model...")
    model = load_model(model_file)
    print("Model loaded.")
except:
    print("No model found or model could not be loaded. Starting over.")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1'))
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='dense1'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(num_classes, activation='softmax', name='dense2'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

filename = 'models/keras_mnist_cnn - epoch {epoch:02d} - accuracy {val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filename, verbose=1, monitor='val_acc', save_best_only=False, mode='auto')
mainsave = ModelCheckpoint(model_file, verbose=1, save_best_only=False, mode='auto')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint, mainsave])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
