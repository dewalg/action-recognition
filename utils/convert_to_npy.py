import imageio
import glob
import numpy as np
import os
import os.path

for image_path in glob.glob("*.jpg"):
    image = misc.imread(image_path)


def convert_files():

    class_folders = glob.glob(os.path.join("


def main():
    """
    Put all the images into a npy file
    """
    convert_files()


if __name__ == '__main__':
    main()
