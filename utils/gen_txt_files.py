import os
import glob
import random
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("../config/config.ini")

def main(TRAIN_DIR, VAL_DIR, TRAIN_OUT, VAL_OUT, LABEL_OUT, cat_limit=None, vid_limit=None):
    OUTPUT_DIR = config['paths']['base_fp'] + '/config/'
    os.chdir(TRAIN_DIR)

    categories = []
    all_vids = []
    val_vids = []
    vid_list = os.listdir(TRAIN_DIR)
    random.shuffle(vid_list)
    for label in vid_list:
        if label.startswith('.'):
            continue

        if len(categories) == cat_limit:
            break

        categories.append(label)
        count = 0
        for video in glob.glob(os.path.join(label, "*.mp4")):
            filepath = os.path.splitext(video)[0]
            all_vids.append(filepath)
            count += 1
            if count == vid_limit:
                break

    random.shuffle(all_vids)
    with open(os.path.join(OUTPUT_DIR, TRAIN_OUT), 'w') as f:
        for path in all_vids:
            path = path.replace("/-", "/")
            if os.listdir(os.path.join(TRAIN_DIR, path)):
                f.write(os.path.join(TRAIN_DIR, path) + "\n")

    os.chdir(VAL_DIR)
    for label in categories:
        for video in glob.glob(os.path.join(label, "*.mp4")):
            filepath = os.path.splitext(video)[0]
            val_vids.append(filepath)

    random.shuffle(val_vids)
    with open(os.path.join(OUTPUT_DIR, VAL_OUT), 'w') as f:
        for path in val_vids:
            path = path.replace("/-", "/")
            if os.listdir(os.path.join(VAL_DIR, path)):
                f.write(os.path.join(VAL_DIR, path) + "\n")


    with open(os.path.join(OUTPUT_DIR, LABEL_OUT), 'w') as f:
        for cat in categories:
            f.write(cat + "\n")


if __name__ == '__main__':
    TRAIN_DIR = config['paths']['train_data']
    VAL_DIR = config['paths']['val_data']
    main(TRAIN_DIR, VAL_DIR, 'train_micro.txt', 'val_micro.txt', 'label_map_micro.txt', 5, 150)

