from pipeline import Pipeline
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from clocknet.clocknet import ClockNet
from clocknet.resnet import inception_resnet_v2_tf
import tensorflow as tf

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../config/config.ini')


NUM_FRAMES = config['hp'].getint('num_frames')
CROP_SIZE = config['hp'].getint('crop_size')
BATCH_SIZE = config['hp'].getint('batch_size')
STRIDE = NUM_FRAMES

VID_DIR = config['paths']['train_fp']
CLS_DICT_FP = config['paths']['cls_dict_fp']

pipe = Pipeline(VID_DIR, CLS_DICT_FP)

queue = pipe.get_dataset().batch(1)
iterator = tf.data.Iterator.from_structure(queue.output_types, queue.output_shapes)
init_op = iterator.make_initializer(queue)

rgb, labels = iterator.get_next()

with tf.variable_scope('model'):
    model = ClockNet(num_classes=10)
    mem = model._build(rgb)

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(tf.global_variables_initializer())
    mem = sess.run([mem])
    print(np.array(mem).shape)