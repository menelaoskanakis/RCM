import numpy as np
import os
import glob
import argparse

from sklearn.decomposition import PCA
from multiprocessing import Pool


def pca(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, 'batch*'))

    pca = PCA()

    activations = []
    for ind, file in enumerate(list_of_files):
        if len(activations) == 0:
            activations = np.load(file)
        else:
            activations = np.concatenate((activations, np.load(file)), axis=1)

    pca.fit(np.transpose(activations))

    np.save(os.path.join(folder_path, 'components.npy'), np.transpose(pca.components_))
    np.save(os.path.join(folder_path, 'singular_values.npy'), pca.singular_values_)
    np.save(os.path.join(folder_path, 'mean.npy'), pca.mean_)
    print(folder_path)


parser = argparse.ArgumentParser(description='ImageNet activation decomposition')
parser.add_argument('--root', type=str, help='define root')
args = parser.parse_args()


list_of_folders = [os.path.join(args.root, f) for f in os.listdir(args.root)]

p = Pool()
p.map(pca, list_of_folders)
