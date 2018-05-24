# using python 3

import tensorflow as tf
import sonnet as snt
from .flownet.src.flownet2 import flownet2

# debug flag for debugging outputs
_DEBUG = True

H_f = W_f = 17
D_f = 2


class ClockFlow:
    def __init__(self):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        # load FlowNet checkpoint.
        self.net = flownet2.FlowNet2()
        self.prev_frame = tf.multiply(tf.ones([399, 399, 3]), 255.0)

    def load(self, tfsession, vars=None):
        self.net.load_ckpt(tfsession, vars)

    def iterate(self, memory, frame):
        if _DEBUG: print("CLOCK_FLOW debug: frame shape = ", frame.shape)
        features = self.net.compute_flow(self.prev_frame, frame)
        features = tf.reshape(features, [self.mem_h, self.mem_w, self.df])
        self.prev_frame = frame
        return features

    def _build(self, inputs):
        self.prev_frame = inputs[0][0]
        if _DEBUG: print("CLOCK_FLOW debug: inputs shape = ", inputs.shape)
        initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
        memory = tf.scan(self.iterate, inputs, initializer=initial_state)
        return memory

