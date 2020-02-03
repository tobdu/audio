from keras.layers import Conv2D, BatchNormalization, Convolution2D, ELU, MaxPooling2D, GaussianDropout, GaussianNoise, \
    Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential
import tensorflow as tf

def build(input_shape, output_shape):
    channel_axis = 1
    freq_axis = 2

    # create model
    model = Sequential()

    # Input block
    model.add(BatchNormalization(axis=freq_axis, name='bn_0_freq', input_shape=input_shape))

    # model.add(GaussianNoise(0.1))

    # Conv block 1
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv1'))
    model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn1'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 4), name='pool1',dim_ordering="th"))
    model.add(Dropout(0.05, name='dropout1'))

    # Conv block 2
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv2'))
    model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 4), name='pool2', dim_ordering="th"))
    model.add(Dropout(0.05, name='dropout2'))

    # # Conv block 3
    # model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv3'))
    # model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn3'))
    # model.add(ELU())
    # model.add(MaxPooling2D(pool_size=(2, 4), name='pool3',dim_ordering="th"))
    # model.add(Dropout(0.1, name='dropout3'))
    #
    # # Conv block 4
    # model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv4'))
    # model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn4'))
    # model.add(ELU())
    # model.add(MaxPooling2D(pool_size=(3, 5), name='pool4', dim_ordering="th"))
    # model.add(Dropout(0.1, name='dropout4'))

    # Conv block 5
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv5'))
    model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn5'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4, 4), name='pool5', dim_ordering="th"))
    model.add(Dropout(0.05, name='dropout5'))

    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid', name='output'))

    model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())

    return model