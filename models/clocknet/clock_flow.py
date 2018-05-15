'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''

# using python 3

import numpy as np
import time
import tensorflow as tf
import sonnet as snt
from .flownet.src.flownet2 import flownet2

# debug flag for debugging outputs
_DEBUG = True

H_f = W_f = 17
D_f = 1088


class Flownet:
    def __init__(self):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f

    def call(self, prev_frame, curr_frame):
        """
        :param prev_frame: rgb input frame - should be 399 x 399 x 3
        :param curr_frame: rgb input frame - should be 399 x 399 x 3
        :return: 17 x 17 x 2 flow information.
        """
        # should return a h_f x w_f x 2 tensor
        net = flownet2.FlowNet2()
        return net.compute_flow(prev_frame, curr_frame)
        # return tf.random_normal([self.mem_h, self.mem_w, 2],
        #                                mean=0, stddev=1)


class ClockFlow(snt.AbstractModule):
    def __init__(self, num_classes, name='clockflow'):
        super(ClockFlow, self).__init__(name=name)
        self.num_classes = num_classes
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.flownet = Flownet()
        self.prev_frame = tf.random_normal([self.mem_h, self.mem_w, 3], mean=0, stddev=1)

    def iterate(self, memory, frame):
        features = self.flownet.call(self.prev_frame, frame)
        features = tf.reshape(features, [self.mem_h, self.mem_w , 2])
        return features

    def _build(self, inputs):
        if _DEBUG: print("INPUTS SHAPE...******", inputs.shape)
        initial_state = tf.zeros([self.mem_w, self.mem_h, 2])
        memory = tf.scan(self.iterate, inputs[0], initializer=initial_state)
        return memory


