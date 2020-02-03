import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot(y_true, y_predict, dir):
    matrix = confusion_matrix(y_true, y_predict)

    df_cm = pd.DataFrame(matrix)\
        # , index=[i for i in "ABCDEFGHIJK"],
        #                  columns=[i for i in "ABCDEFGHIJK"])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(dir + "/matrix.png")