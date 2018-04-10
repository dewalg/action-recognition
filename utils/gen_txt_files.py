import os
import glob
import random
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("../config/config.ini")

def main(TRAIN_DIR, VAL_DIR, TRAIN_OUT, VAL_OUT, LABEL_OUT, limit=None):
    OUTPUT_DIR = config['paths']['base_fp'] + '/config/'
    os.chdir(TRAIN_DIR)

    categories = []
    all_vids = []
    val_vids = []
    for label in os.listdir(TRAIN_DIR):
        if label.startswith('.'):
            continue

        if len(categories) == limit:
            continue

        categories.append(label)
        for video in glob.glob(os.path.join(label, "*.mp4")):
            filepath = os.path.splitext(video)[0]
            all_vids.append(filepath)

    random.shuffle(all_vids)
    with open(os.path.join(OUTPUT_DIR, TRAIN_OUT), 'w') as f:
        for path in all_vids:
            if os.listdir(os.path.join(TRAIN_DIR, path)):
                f.write(os.path.join(TRAIN_DIR, path) + "\n")

    os.chdir(VAL_DIR)
    for label in categories:
        for video in glob.glob(os.path.join(label, "*.mp4")):
            filepath = os.path.splitext(video)[0]
            val_vids.append(filepath)

    random.shuffle(val_vids)
    with open(os.path.join(OUTPUT_DIR, VAL_OUT), 'w') as f:
        for path in all_vids:
            if os.listdir(os.path.join(VAL_DIR, path)):
                f.write(os.path.join(VAL_DIR, path) + "\n")


    with open(os.path.join(OUTPUT_DIR, LABEL_OUT), 'w') as f:
        for cat in categories:
            f.write(cat + "\n")


if __name__ == '__main__':
    TRAIN_DIR = config['paths']['train_data']
    VAL_DIR = config['paths']['val_data']
    main(TRAIN_DIR, VAL_DIR, 'train_micro.txt', 'val_micro.txt', 'label_map_micro.txt', 2)

