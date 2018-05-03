'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''

# using python 3

import tensorflow as tf
import sonnet as snt

# debug flag for debugging outputs
_DEBUG = False

class Resnet:
    # TODO: plug in resnet
    def __init__(self):
        self.mem_h = 14
        self.mem_w = 14
        self.df = 256

    def call(self, inputs):
        # should return a 14x14x256 tensor
        return tf.random_normal([self.mem_h, self.mem_w, self.df],
                                mean=0, stddev=1)


class Flownet:
    # TODO: plug in Flownet
    def __init__(self):
        self.mem_h = 14
        self.mem_w = 14
        self.df = 256

    def call(self, inputs):
        # should return a 14x14x256 tensor
        return tf.random_normal([self.mem_h, self.mem_w, self.df],
                                mean=0, stddev=1)


class ClockNet(snt.AbstractModule):
    def __init__(self, num_classes, name='clocknet'):
        super(ClockNet, self).__init__(name=name)
        self.num_classes = num_classes

        # on resnet architecture, conv4_x spits out
        # a 14x14x256 feature map which is what we use
        self.mem_h = 14
        self.mem_w = 14
        self.df = 256

        # memory is initialized randomly
        self.memory = tf.random_normal([self.mem_h, self.mem_w, self.df],
                                       mean=0, stddev=1)

        self.resnet = Resnet()
        self.flownet = Flownet()
        self.prev_frame = tf.random_normal([self.mem_h, self.mem_w, self.df],
                                       mean=0, stddev=1)


    def compute_mem(self, curr_frame):
        # compute flow bewteen two frames
        flow = self.flownet.call(self.prev_frame, curr_frame)

        # calculate the new memory
        self.memory = self.phi(self.memory, flow)

        # save the prev frame to compute flow for next iteration
        self.prev_frame = self.curr_frame


    def iterate(self, frame, is_training):
        # computation per frame
        features = self.resnet.call(frame)

        # compute the new memory with flow from prev frame
        # (corresponds to 'w' function in the paper)
        self.compute_mem(frame)

        # compute the new memory
        # corresponds to the 'A' function
        self.aggregate(features)


    def build(self, inputs, is_training=False):
        """Connects the model to inputs.

        Args:
          inputs: Inputs to the model, which should have dimensions
              `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
          is_training: whether to use training mode for snt.BatchNorm (boolean).
          dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
              [0, 1)).

        Returns:
          A tuple consisting of:
            1. Network output at location `self._final_endpoint`.
            2. Dictionary containing all endpoints up to `self._final_endpoint`,
               indexed by endpoint name.
        """
        for frame in inputs.get_shape()[1]:
            self.iterate(frame, is_training)


