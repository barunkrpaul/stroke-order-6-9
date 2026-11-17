"""Download the two public datasets used in the paper into this folder.

Pendigits: the PMLB copy of the UCI dataset (doi:10.24432/C5MG6K), saved as pd.tsv.gz.
MNIST: the OpenML copy (all 70000 images), saved as mnist_X.npy and mnist_y.npy.
"""
import os, urllib.request, numpy as np

PEN = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
       "datasets/pendigits/pendigits.tsv.gz")

if not os.path.exists("pd.tsv.gz"):
    urllib.request.urlretrieve(PEN, "pd.tsv.gz")
    print("Pendigits saved to pd.tsv.gz")

if not os.path.exists("mnist_X.npy"):
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    np.save("mnist_X.npy", X.astype(np.uint8))
    np.save("mnist_y.npy", y.astype(np.int64))
    print("MNIST saved to mnist_X.npy and mnist_y.npy")
