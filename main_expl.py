from TSB_UAD_code.feature import Window
from Techniques.distanceKR import static_kr as DOD
# Explainable Dyn (integrated version). For comparison, the naive version can be used instead:
# from Techniques.dynamic_expl3 import dynamic_kr as Dyn
from Techniques.dynamic_expl import dynamic_kr_expl as Dyn
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def data_and_labels(sequence_len,dataset):
    df = pd.read_csv(dataset)

    X_data = Window(window=sequence_len).convert(df.dropna().to_numpy()[:, 0].astype(float)).to_numpy()
    labels = df.values[:len(X_data), 1]

    templabel = labels.copy()

    for i in range(len(labels)):
        if labels[i] == 1:
            for q in range(max(0, i - sequence_len + 1), i):
                templabel[q] = 1
            for q in range(i, min(len(labels), i + sequence_len)):
                templabel[q] = 1
    label = templabel
    return X_data,label

def plot(DOD_score,Dyn_score,label):
    plt.subplot(211)
    plt.plot(DOD_score, color="green", label="DOD")
    plt.fill_between([i for i in range(len(label))], label, where=label, color="red", alpha=0.5, label="Real Anomalies")
    plt.xlabel("time")
    plt.legend()
    plt.subplot(212)
    plt.plot(Dyn_score, color="Blue", label="DYN")
    plt.fill_between([i for i in range(len(label))], label, where=label, color="red", alpha=0.5, label="Real Anomalies")
    plt.legend()
    plt.xlabel("time")
    plt.show()


if __name__ == "__main__" :
    X_data, label=data_and_labels(sequence_len=10,dataset="./data/YAHOO/Yahoo_A1real_53_data.out")


    # Application of generating score in online fashion for all data, applyig sliding window:
    dod_clf = DOD(k=50, R=10, window=200, slide=100)
    dyn_clf = Dyn(window=200, slide=100)

    DOD_anomalies=dod_clf.fit(X_data)
    Dyn_anomalies=dyn_clf.fit(X_data)
    
    # Detected outliers and their explanation labels
    print("--- Detected Anomalies and Explanations ---")
    for idx, (score, exp_label) in enumerate(zip(Dyn_anomalies, dyn_clf.outlier_labels)):
        if score > 0:
            print(f"Timestamp Index {idx}: Outlier Detected -> Explanation: {exp_label}")

    plot(DOD_anomalies, Dyn_anomalies, label)