from zipfile import ZipFile
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from time import time
from tqdm import tqdm

def plot_gallery(images, titles=None, xscale=1, yscale=1, nrow=3, cmap='gray', output=None):
    ncol = math.ceil(len(images) / nrow)
    
    plt.figure(figsize=(xscale * ncol, yscale * nrow))

    for i in range(nrow * ncol):
        plt.subplot(nrow, ncol, i + 1)
        if i < len(images):
            plt.imshow(images[i], cmap=cmap)
            if titles is not None:
                # use size and y to adjust font size and position of title
                plt.title(titles[i], size=12, y=1)
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()

    if output is not None:
        plt.savefig(output)
    plt.show()

class DatasetWrapper(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]

def zip_sol():
    shutil.make_archive('A2-sol', 'zip', base_dir='supplements')
    with ZipFile('A2-sol.zip', 'a') as zipfile:
        zipfile.write('A2.ipynb', 'A2.ipynb')

    print('## created A2-sol.zip containing the following files ##')
    with ZipFile('A2-sol.zip', 'r') as zipObj:
       listOfiles = zipObj.namelist()
       for elem in listOfiles:
           print(elem)

if __name__ == '__main__':
    zip_sol()
