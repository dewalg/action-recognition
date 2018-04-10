import os
import glob
import subprocess
from tqdm import tqdm
from joblib import Parallel, delayed
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../config/config.ini')

### GLOBALS
# dataset_dir = '/datasets/home/71/671/cs291dag/MiniKinetics/train/'
dataset_dir = config['paths']['train_data']
###


def createJPGs(video, label_path):
    '''
    creates the jpegs by calling the ffmpeg
    '''
#    print('here: ' + video)
    dest_name = os.path.splitext(video)[0]
    if dest_name.startswith('-'):
        dest_name = dest_name[1:]

    if dest_name not in os.listdir():
        os.mkdir(dest_name)

    if len(os.listdir(dest_name)) != 0:
        return

    print('adding ' + video)
    video_path = os.path.join(label_path, video)
    dest_name = os.path.join(dest_name, "img%4d.jpg")

    video = str(video).replace(' ', '\ ')
    dest = str(dest_name).replace(' ', '\ ')
    command = "ffmpeg -i " + video + " -r 25.0 " + dest
    proc = subprocess.Popen(
        command,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd='.'
    )
    out, err = proc.communicate()


def main():
    '''
    1. move into the video directory
    2. extract the frames and resize them using
        bilinear extrapolation
    3. randomly select a 224 by 224 crop
    4. change pixel values to be [-1, 1]
    '''
    os.chdir(dataset_dir)
    for label in tqdm(os.listdir()):
        if label.startswith("."):
            continue

        print("===================== " + label + " ======================== ")
        label_path = os.path.join(dataset_dir, label)
        os.chdir(label_path)
        files_list = glob.glob("*.mp4")

        Parallel(n_jobs=-1, verbose=True)(delayed(createJPGs)(video, label_path) for video in files_list)

        # for video in glob.glob("*.mp4"):
        #     print("\tprocessing video: " + video)
        #     dest_name = os.path.splitext(video)[0]
        #     if dest_name not in os.listdir():
        #         os.mkdir(dest_name)
        #     video_path = os.path.join(label_path, video)
        #     dest_name = os.path.join(dest_name, "img%4d.jpg")
        #     createJPGs(video_path, dest_name)

if __name__ == '__main__':
    main()
