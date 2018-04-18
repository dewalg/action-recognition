import os
import subprocess
from tqdm import tqdm
from joblib import Parallel, delayed
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../config/config.ini')

dataset_dir = config['paths']['val_data']


def mod_dirs(dir, label_path):
    """
    creates the jpegs by calling the ffmpeg
    """
    print('changing ' + dir + ' under ' + label_path)
    mod_dir = dir[1:]
    dir_path = os.path.join(label_path, dir)
    mod_path = os.path.join(label_path, mod_dir)
    dir_path = str(dir_path).replace(' ', '\ ')
    mod_path = str(mod_path).replace(' ', '\ ')
    command = "mv " + dir_path + " " + mod_path
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
    change_count = 0
    for label in tqdm(os.listdir()):
        if label.startswith("."):
            continue

        print("===================== " + label + " ======================== ")
        label_path = os.path.join(dataset_dir, label)
        os.chdir(label_path)
        dirs_list = [dir_name for dir_name in os.listdir(label_path) if dir_name.startswith('-') and os.path.isdir(os.path.join(label_path, dir_name))]
        # Parallel(n_jobs=-1, verbose=True)(delayed(mod_dirs)(dir, label_path) for dir in dirs_list)
        change_count += len(dirs_list)
        print("CHANGE COUNT ",change_count)


if __name__ == '__main__':
    main()
