from pipeline import Pipeline
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from clocknet.clock_step import ClockStep
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

#with tf.variable_scope('clocknet'):
model = ClockStep(num_classes=10)
mem = model._build(rgb)

vars_to_load_flow = tf.contrib.framework.get_variables_to_restore(include=['clock_flow'])

with tf.Session() as sess:
    model.init_flow(sess, vars_to_load_flow)
    sess.run(init_op)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./profiler", sess.graph)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    mem = sess.run([mem], options=run_options, run_metadata=run_metadata)
    writer.add_run_metadata(run_metadata, 'step001')
    print(np.array(mem).shape)
