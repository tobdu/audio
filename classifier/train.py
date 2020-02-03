import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

# from utils.confusion_matrix import plot_confusion_matrix
from model import cv2_big
from model import cv2_simple
from utils import histories
from utils import confusion_matrix
from preprocess import label_encoding

data_path = "data/melgrams.pkl"
encodings_path = "data/index/encoding.csv"


def load_data(path, encoder):
    print("loading data")
    df = pd.read_pickle(path)

    # genres = {"blues", "classical"}
    # df = df[df['id'].apply(lambda x: x in genres)]

    print("formatting X")
    X = [i[0] for i in df["melgram"]]
    X = np.array([x for x in X])
    map(lambda x : tf.convert_to_tensor(x), X)
    # tf.convert_to_tensor(x)
    # X = np.array(X, copy=True, subok=True)     # this is causing it to hang

    print("formatting y")
    y = np.array(encoder.transform([i for i in df["id"]]))

    return X, y


# split while keeping class balance
def split(X, y):
    splits = StratifiedShuffleSplit(test_size=0.3).split(X,y)

    for train_i, test_i in splits:
        X_train = X[train_i]
        y_train = y[train_i]
        X_test = X[test_i]
        y_test = y[test_i]

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    encoder = label_encoding.load()

    X, y = load_data(data_path, encoder)

    X_train, X_test, y_train, y_test = split(X, y)

    # X_train = tf.convert_to_tensor(X_train)

    input_shape = X[0].shape

    print("compiling model")
    model = cv2_simple.build(input_shape, len(y[0]))
    #model = cv2_big.build(X_train[0])

    model.summary()

    print("training model")
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test)) #30

    print("testing model")
    y_predict = model.predict(X_test)

    print("de-encoding labels")
    y_predict = encoder.inverse_transform(y_predict)

    y_test = encoder.inverse_transform(y_test)

    print("printing results")
    confusion_matrix.plot(y_test, y_predict, 'data/matrix.png')

    # histories.plot(history, 'data/history.png')

    print("saving model")
    model.save('data/model.h5')



