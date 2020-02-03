from kapre.filterbank import Filterbank
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


    # model.add(Filterbank(n_fbs=1, trainable_fb=True, sr=12000, init='mel', fmin=0., fmax=None,
    #                      bins_per_octave=12, image_data_format='default', input_shape=input_shape))

    # Input block
    model.add(BatchNormalization(axis=freq_axis, name='bn_0_freq', input_shape=input_shape))


    # Conv block 1
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv1'))
    model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn1'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 4), name='pool1',dim_ordering="th"))
    model.add(Dropout(0.2, name='dropout1'))

    # Conv block 2
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv2'))
    model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn2'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 4), name='pool2', dim_ordering="th"))
    model.add(Dropout(0.2, name='dropout2'))

    # Conv block 3
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv3'))
    model.add(BatchNormalization(axis=channel_axis, mode=0, name='bn3'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(4, 4), name='pool3', dim_ordering="th"))
    model.add(Dropout(0.1, name='dropout3'))

    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid', name='output'))

    model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())

    return model