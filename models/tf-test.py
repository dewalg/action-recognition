from pipeline import Pipeline
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
from clocknet.clock_step import ClockStep
# from clocknet.clock_flow import ClockFlow
import tensorflow as tf
from clocknet.resnet import inception_resnet_v2_wrapper
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

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

rgb_reshaped = tf.reshape(rgb, [64, 299, 299, 3])
resnet = inception_resnet_v2_wrapper.InceptionResNetV2(input_tensor=rgb_reshaped)
resnet_out = tf.identity(resnet['mixed_6a'], name='rgb_resnet_out')

# flow = ClockFlow(inputs=rgb_reshaped)
# flow_out = flow.get_flow()
flow_out = tf.random_uniform([64, 17, 17, 2])

model = ClockStep(num_classes=10)
mem = model._build(rgb, resnet_out, flow_out)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=mem)

##### FOR MULTI-GPU ERROR, UNCOMMENT BELOW
# mem = []
# rgb_reshaped = tf.reshape(rgb, [64, 299, 299, 3])
# rnet = inception_resnet_v2_wrapper.InceptionResNetV2()
# for i in range(1):
#     with tf.name_scope('tower_%d' % i) as scope:
#         resnet = rnet._build_graph(rgb_reshaped)
#         resnet_out = tf.identity(resnet['mixed_6a'], name='rgb_resnet_out')
#         print('FINISHED ' + str(i) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#
#         # flow = ClockFlow(inputs=rgb_reshaped)
#         # flow_out = flow.get_flow()
#         flow_out = tf.random_uniform([64, 17, 17, 2])
#
#         model = ClockStep(num_classes=10)
#         mem.append(model._build(rgb, resnet_out, flow_out))

with tf.Session() as sess:

    sess.run(init_op)
    sess.run(tf.global_variables_initializer())
    resnet.load_weights()
    # flow.load_weights()

    # USE BELOW TO SEE PLOT OF RESNET LOADED WEIGHTS
    # w = resnet.irv2.get_layer('conv2d_78').get_weights()[0]
    # plt.hist(w.flatten())
    # plt.show()

    # USE BELOW TO SEE ALL LAYERS AND THEIR WEIGHTS
    # for layer in irv2.layers:
    #     w = layer.get_weights()
    #     print(layer.name + " " + str(len(w)))

    # USE BELOW TO RUN PROFILER ON CLOCKNET
    writer = tf.summary.FileWriter("./profiler", sess.graph)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    loss, mem = sess.run([loss, mem], options=run_options, run_metadata=run_metadata)
    writer.add_run_metadata(run_metadata, 'step001')
    print(mem)
    print(loss)
    writer.close()
