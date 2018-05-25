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
# model = ClockStep(num_classes=10)
# mem = model._build(rgb)

all_vars = tf.all_variables()
model_vars = [k for k in all_vars if k.name.startswith("clock_flow")]

v_flow = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='clock_flow')
print("LEN OF VARS")
print(len(v_flow))
if len(v_flow) > 0:
    print(v_flow[0])
    print(v_flow[1])
    print(v_flow[2])

v_dict = {v.op.name: v for v in v_flow}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_op)
    model = ClockStep(num_classes=10)
    mem = model._build(rgb)
    model.init_flow(model_vars)
    model.init_rgb()

    writer = tf.summary.FileWriter("./profiler", sess.graph)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    mem = sess.run([mem], options=run_options, run_metadata=run_metadata)
    writer.add_run_metadata(run_metadata, 'step001')
    print(np.array(mem).shape)
