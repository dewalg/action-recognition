'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''

# using python 3

import numpy as np
import tensorflow as tf
import sonnet as snt
from .resnet import inception_resnet_v2

# debug flag for debugging outputs
_DEBUG = False
H_f = W_f = 17
D_f = 1088

class Resnet:
    def __init__(self):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.model = inception_resnet_v2.InceptionResNetV2()

    def call(self, inputs):
        # inputs is a 299x299x3 tensor
        # should return a 17x17x1088 tensor
        return self.model.predict(np.array([inputs]))[0]

    def callBatch(self, inputs):
        # inputs is a dx299x299x3 tensor
        # should return a dx17x17x1088 tensor
        return self.model.predict(inputs)

class Flownet:
    # TODO: plug in Flownet
    def __init__(self):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f

    def call(self, prev, after):
        # should return a W_f by H_f by D_f tensor
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
        self.prev_frame = tf.random_normal([self.mem_h, self.mem_w, 3],
                                       mean=0, stddev=1)

    def compute_mem(self, curr_frame):
        # compute flow bewteen two frames
        flow = self.flownet.call(self.prev_frame, curr_frame)

        # calculate the new memory
        self.memory = self.bl_sample(self.memory, flow)

        # save the prev frame to compute flow for next iteration
        self.prev_frame = self.curr_frame

    def iterate(self, frame):
        """
        Iterate does required computations on a frame level
        :param frame: the frame (or time step) currently being processed 
        :return: void - updates the memory 
        """
        # computation per frame
        features = self.resnet.call(frame)

        # compute the new memory with flow from prev frame
        # (corresponds to 'w' function in the paper)
        self.compute_mem(frame)

        # compute the new memory
        # corresponds to the 'A' function
        self.aggregate(features)

    @staticmethod
    def bl_kernel(a, b):
        """
        This function is the bilinear kernel. Calculates
        max(0, 1 - |a-b|) as described in https://arxiv.org/pdf/1611.07715.pdf
        """
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        return tf.maximum(0., 1 - tf.abs(a - b))

    def bl_sample(self, features, displ):
        """
        
        :param features: feature maps (aka the memory) that will be updated
                        Expected to be a H_f x W_f x D_f tensor
        :param displ: the displacement field
                        Expected to be a H_f x W_f x 2 tensor
        :return: returns the new feature map determine by a 
                        bilinear interpolation as descrbied in:
                        https://arxiv.org/pdf/1611.07715.pdf     
        """

        memory = tf.zeros([W_f, H_f, D_f])
        for c in range(D_f):
            # do BL-interpolation per channel
            for px in range(W_f):
                for py in range(H_f):
                    d_px = displ[px, py, 0]
                    d_py = displ[px, py, 1]
                    results = 0
                    for qx in range(W_f):
                        for qy in range(H_f):
                            prev = features[qx, qy, c]
                            if prev == 0:
                                continue

                            gx = self.bl_kernel(qx, px+d_px)
                            gy = self.bl_kernel(qy, py+d_py)
                            results += prev * gx * gy

                    memory[px, py, c] = results

        return memory

    def _build(self, inputs):
        """
        Connects the model to inputs.

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
        self.memory = tf.random_normal([self.mem_h, self.mem_w, self.df],
                                       mean=0, stddev=1)

        # print(inputs.shape)
        # for frame in inputs.get_shape()[1]:
        for frame in range(inputs.shape[1]):
            self.iterate(inputs[0][frame])

        return self.memory


