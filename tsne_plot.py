from sklearn.manifold import TSNE
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_tsne(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X, y = data
    X_embedded = TSNE(n_components=3).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y)
    plt.show()


plot_tsne("/data/home/user/Aditya_Vallakatla_19CS30051/FewShot3D-MLOps/logs/pcnmlp/encodings.pkl")
