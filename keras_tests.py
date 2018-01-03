from __future__ import print_function
import keras
from keras import backend as k
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
checkpoint_file = 'models/keras_mnist_cnn - epoch {epoch:02d} - accuracy {val_acc:.4f}.h5'

# Load, format, and normalize MNIST sample data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert to a binary class matrix for use with categorical_crossentropy.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

try:
    # Load the model if it already exists.
    print("Loading existing model...")
    model = load_model(model_file)
    print("Model loaded.")
except:
    # Build it from scratch if it doesn't.
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
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])

# Set up our checkpoint saves.
# We'll save the main model every epoch but save checkpoints only when they beat the accuracy record.
checkpoint = ModelCheckpoint(checkpoint_file, verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
mainsave = ModelCheckpoint(model_file, verbose=1, save_best_only=False, mode='auto')

# Do the training.
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint, mainsave])

# See how we did.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
