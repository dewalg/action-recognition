import os
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

        label_path = os.path.join(DATA_DIR, label)
        dirs_list = [dir_name for dir_name in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, dir_name))]
        for dir in dirs_list:
            dir_path = os.path.join(label_path, dir)
            all_vids.append(dir_path)

    random.shuffle(all_vids)
    with open(os.path.join(OUTPUT_DIR, INF_OUT), 'w') as f:
        for path in all_vids:
            if os.listdir(path):
                f.write(path + "\n")


if __name__ == '__main__':
    TRAIN_DIR = config['paths']['train_data']
    VAL_DIR = config['paths']['val_data']
    categories = config['paths']['inf_cat']
    main(VAL_DIR, 'inference.txt', categories)

