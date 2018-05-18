# using python 3

import tensorflow as tf
import sonnet as snt
from .flownet.src.flownet2 import flownet2

# debug flag for debugging outputs
_DEBUG = True

H_f = W_f = 17
D_f = 2


class Flownet:
    def __init__(self):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        # load FlowNet checkpoint.
        self.net = flownet2.FlowNet2()
        self.net.load_ckpt()

    def call(self, prev_frame, curr_frame):
        # should return a h_f x w_f x 2 tensor
        return self.net.compute_flow(prev_frame, curr_frame)


class ClockFlow(snt.AbstractModule):
    def __init__(self, num_classes, name='clockflow'):
        super(ClockFlow, self).__init__(name=name)
        self.num_classes = num_classes
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.flownet = Flownet()
        # self.prev_frame = tf.random_normal([399, 399, 3], mean=0, stddev=1)
        self.prev_frame = tf.multiply(tf.ones([399, 399, 3]), 255.0)

    def iterate(self, memory, frame):
        if _DEBUG: print("CLOCK_RGB debug: frame shape = ", frame.shape)
        features = self.flownet.call(self.prev_frame, frame)
        features = tf.reshape(features, [self.mem_h, self.mem_w, self.df])
        self.prev_frame = frame
        return features

    def _build(self, inputs):
        # ones = tf.ones([32, 399, 399, 3])
        # zeros = tf.zeros([32, 399, 399, 3])
        # ones = tf.expand_dims(ones, 0)
        # zeros = tf.expand_dims(zeros, 0)
        # inputs = tf.concat([ones, zeros], 0)
        # inputs = tf.reshape(inputs, [64, 399, 399, 3])
        self.prev_frame = inputs[0][0]
        if _DEBUG: print("CLOCK_RGB debug: inputs shape = ", inputs.shape)
        initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
        memory = tf.scan(self.iterate, inputs[0], initializer=initial_state)
        return memory
