import pickle

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

index_path = 'data/index/gtzan_genre.csv'
encodings_path = 'data/encoding.pkl'


def load_labels(path):
    df = pd.read_csv(path)
    return df['id'].unique()


def create_and_save():
    labels = load_labels(index_path)

    enc = LabelBinarizer()

    enc.fit(labels)

    pickle.dump(enc, open(encodings_path, 'wb'))


def load():
    return pickle.load(open(encodings_path,'rb'))


if __name__ == "__main__":
    create_and_save()




