import os
import glob
import random
import tempfile
import subprocess
import numpy as np
from scipy import misc
from datetime import datetime


### GLOBALS
dataset_dir = '/Users/dewalgupta/Documents/ucsd/291d/activitynet/data/'
###


def resize_crop(img: np.ndarray) -> np.ndarray:
    '''
    resize the image frame to a random 224 by 224
    '''

    aspect_ratio = float(img.shape[1]) / float(img.shape[0])
    new_w = 0
    new_h = 0
    if aspect_ratio <= 1.0:
        new_w = 256
        new_h = int(256 / aspect_ratio)
    else:
        new_h = 256
        new_w = int(256 * aspect_ratio)

    random.seed(datetime.now())
    resize = misc.imresize(img, (new_h, new_w), 'bilinear')
    wrange = resize.shape[1] - 224
    hrange = resize.shape[0] - 224
    w_crop = random.randint(0, wrange)
    h_crop = random.randint(0, hrange)

    return resize[h_crop:h_crop+224, w_crop:w_crop+224]


def createJPGs(video, dest):
    '''
    creates the jpegs by calling the ffmpeg
    '''
    video = str(video).replace(' ', '\ ')
    proc = subprocess.Popen(
        "ffmpeg -i " + video + " -r 25.0 " + dest,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd='.'
    )
    out, err = proc.communicate()
    # print(out)
    # print(err)


def main():
    '''
    1. move into the video directory
    2. extract the frames and resize them using
        bilinear extrapolation
    3. randomly select a 224 by 224 crop
    4. change pixel values to be [-1, 1]
    '''
    os.chdir(dataset_dir)
    for label in os.listdir():
        if label.startswith("."):
            continue

        print("===================== " + label + " ======================== ")
        label_path = os.path.join(dataset_dir, label)
        os.chdir(label_path)
        for video in glob.glob("*.mp4"):
            print("\tprocessing video: " + video)
            l = list()
            with tempfile.TemporaryDirectory() as dirpath:
                os.chdir(dirpath)
                video_path = os.path.join(label_path, video)
                createJPGs(video_path, "img%4d.jpg")
                for img in glob.glob("*.jpg"):
                    npimg = misc.imread(img)
                    cropped = resize_crop(npimg)
                    scaled = 2*(cropped/255) - 1
                    l.append(scaled)

                npy = np.array(l)
                save_fn = os.path.join(dataset_dir, label, os.path.splitext(video)[0])
                np.save(save_fn, npy)
                print("\tsaved video with shape: " + str(npy.shape))


if __name__ == '__main__':
    main()
