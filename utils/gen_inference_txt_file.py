import os
import glob
import random
from configparser import ConfigParser, ExtendedInterpolation
import sys

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("../config/config.ini")

def main(DATA_DIR, INF_OUT, inf_categories):
    OUTPUT_DIR = config['paths']['base_fp'] + '/config/'
    os.chdir(DATA_DIR)

    categories = []
    with open(inf_categories) as f:
        categories = [line.strip() for line in f]

    all_vids = []
    all_categories = os.listdir(DATA_DIR)
    for label in categories:
        if label not in all_categories:
            print(label + " not a category")
            sys.exit(0)

        for video in glob.glob(os.path.join(label, "*.mp4")):
            filepath = os.path.splitext(video)[0]
            filepath.replace("/-", "/")
            all_vids.append(filepath)

    random.shuffle(all_vids)
    with open(os.path.join(OUTPUT_DIR, INF_OUT), 'w') as f:
        for path in all_vids:
            if os.listdir(os.path.join(DATA_DIR, path)):
                f.write(os.path.join(DATA_DIR, path) + "\n")


if __name__ == '__main__':
    TRAIN_DIR = config['paths']['train_data']
    VAL_DIR = config['paths']['val_data']
    categories = config['paths']['inf_cat']
    main(VAL_DIR, 'inference.txt', categories)

