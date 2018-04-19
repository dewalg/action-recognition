import os
import random
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("../config/config.ini")


def main(DATA_DIR, INF_OUT, inf_categories):

    OUTPUT_DIR = config['paths']['base_fp'] + '/config/'

    with open(inf_categories) as f:
        categories = [line.strip() for line in f]

    all_categories = os.listdir('/nfs/data/train')
    categories = list(set.intersection(set(categories), set(all_categories)))

    with open(DATA_DIR) as f:
        all_vids = [line.strip() for line in f if line.split('/')[4] in categories]

    random.shuffle(all_vids)
    print("CONTENTS = ",len(all_vids))
    with open(os.path.join(OUTPUT_DIR, INF_OUT), 'w') as f:
        for path in all_vids:
            f.write(path + "\n")


if __name__ == '__main__':
    inf_categories = config['paths']['inf_cat']
    main('/nfs/action-recognition/config/train_pipeline_paths.txt', 'mini_train_pipeline_paths.txt', inf_categories)


