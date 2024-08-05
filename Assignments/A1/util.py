from zipfile import ZipFile
import shutil
import math
import matplotlib.pyplot as plt

def zip_sol():
    shutil.make_archive('A1-sol', 'zip', base_dir='supplements')
    with ZipFile('A1-sol.zip', 'a') as zipfile:
        zipfile.write('A1.ipynb', 'A1.ipynb')

    print('## created A1-sol.zip containing the following files ##')
    with ZipFile('A1-sol.zip', 'r') as zipObj:
       listOfiles = zipObj.namelist()
       for elem in listOfiles:
           print(elem)

def plot_gallery(images, titles=None, xscale=1.5, yscale=1.5, nrow=1, cmap='gray', output=None):
    ncol = math.ceil(len(images) / nrow)
    
    plt.figure(figsize=(xscale * ncol, yscale * nrow))

    for i in range(nrow * ncol):
        plt.subplot(nrow, ncol, i + 1)
        if i < len(images):
            plt.imshow(images[i], cmap=cmap)
            if titles is not None:
                # use size and y to adjust font size and position of title
                plt.title(titles[i], size=10, y=1)
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()

    if output is not None:
        plt.savefig(output)
    plt.show()


if __name__ == '__main__':
    zip_sol()
