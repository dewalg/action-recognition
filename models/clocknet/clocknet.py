'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''

# using python 3

import numpy as np
import tensorflow as tf
import sonnet as snt

from .resnet import inception_resnet_v2_keras
from .resnet import inception_resnet_v2_tf
from .flownet.src import flownet2

# debug flag for debugging outputs
_DEBUG = True

H_f = W_f = 17
D_f = 1088


class Resnet:
    def __init__(self, num_classes = 10):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.num_classes = num_classes
        # self.model = inception_resnet_v2_keras.InceptionResNetV2()

    def call(self, inputs):
        # inputs is a 299x299x3 tensor
        # should return a 17x17x1088 tensor
        if _DEBUG: print('resnet call inputs:')
        if _DEBUG: print(inputs)
        # return self.model.predict(np.array([inputs]))[0]
        inputs = tf.reshape(inputs, [1, 299, 299, 3])
        out, _ = inception_resnet_v2_tf.inception_resnet_v2(inputs, num_classes=self.num_classes, dropout_keep_prob=0.5)
        out = tf.reshape(out, [17, 17, 1088])
        return out

    def call_batch(self, inputs):
        # inputs is a dx299x299x3 tensor
        # should return a dx17x17x1088 tensor
        return self.model.predict(inputs)


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
        # net = flownet2.FlowNet2()
        # return net.compute_flow(prev_frame, curr_frame)
        return tf.random_normal([self.mem_h, self.mem_w, 2],
                                       mean=0, stddev=1)

class ClockNet(snt.AbstractModule):
    def __init__(self, num_classes, name='clocknet'):
        super(ClockNet, self).__init__(name=name)
        self.num_classes = num_classes

        # on resnet architecture, conv4_x spits out
        # a 14x14x256 feature map which is what we use
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f

        # memory is initialized randomly
        # self.memory = tf.random_normal([self.mem_h, self.mem_w, self.df],
        #                                mean=0, stddev=1)

        self.resnet = Resnet(self.num_classes)
        self.flownet = Flownet()
        self.prev_frame = tf.random_normal([self.mem_h, self.mem_w, 3],
                                       mean=0, stddev=1)

    def compute_mem(self, memory, curr_frame):
        # compute flow bewteen two frames
        flow = self.flownet.call(self.prev_frame, curr_frame)

        # calculate the new memory
        memory = self.bl_sample(memory, flow)

        # save the prev frame to compute flow for next iteration
        self.prev_frame = curr_frame

        return memory

    @staticmethod
    def aggregate(memory, features):
        """
        Aggregates the memory with the features map of the
        current frame
        :param memory: current memory (already merged with flow)
        :param features: features of the frame from resnet
        :return: the new memory 
        """
        return tf.multiply(0.5, tf.add(memory, features))

    def iterate(self, memory, frame):
        """
        Iterate does required computations on a frame level
        :param memory: the current memory 
        :param frame: the frame (or time step) currently being processed 
        :return: void - updates the memory 
        """

        if _DEBUG: print(memory)
        if _DEBUG: print(frame)

        # computation per frame
        features = self.resnet.call(frame)

        # compute the new memory with flow from prev frame
        # (corresponds to 'w' function in the paper)
        memory = self.compute_mem(memory, frame)

        # compute the new memory
        # corresponds to the 'A' function
        memory = self.aggregate(memory, features)

        return memory

    # @staticmethod
    # def bl_kernel(a, b):
    #     """
    #     This function is the bilinear kernel. Calculates
    #     max(0, 1 - |a-b|) as described in https://arxiv.org/pdf/1611.07715.pdf
    #     """
    #     return tf.maximum(0., 1 - tf.abs(a - b))

    # def bl_sample(self, features, displ):
    #     """
    #
    #     :param features: feature maps (aka the memory) that will be updated
    #                     Expected to be a H_f x W_f x D_f tensor
    #     :param displ: the displacement field
    #                     Expected to be a H_f x W_f x 2 tensor
    #     :return: returns the new feature map determine by a
    #                     bilinear interpolation as descrbied in:
    #                     https://arxiv.org/pdf/1611.07715.pdf
    #     """
    #
    #     memory = tf.zeros([W_f, H_f, D_f])
    #     for c in range(D_f):
    #         # do BL-interpolation per channel
    #         for px in range(W_f):
    #             for py in range(H_f):
    #                 d_px = displ[px, py, 0]
    #                 d_py = displ[px, py, 1]
    #                 results = 0
    #                 for qx in range(W_f):
    #                     for qy in range(H_f):
    #                         prev = features[qx, qy, c]
    #                         if prev == 0:
    #                             continue
    #
    #                         gx = self.bl_kernel(qx, px+d_px)
    #                         gy = self.bl_kernel(qy, py+d_py)
    #                         results += prev * gx * gy
    #
    #                 memory[px, py, c] = results
    #
    #     return memory

    def bl_sample(self, features, displ):
        a = tf.fill([self.mem_w, self.mem_h, self.df], True)
        ind_p = tf.where(a)
        out = tf.map_fn(lambda p: self.bl_sample_mapped(p, features, displ),
                        ind_p,
                        dtype=tf.float32)
        return out

    def bl_kernel(self, q, p, channel, features):
        prev = features[q[0], q[1], channel]
        q = tf.to_float(q)
        p = tf.to_float(p)
        g1 = tf.maximum(0., 1 - tf.abs(q[0] - p[0]))
        g2 = tf.maximum(0., 1 - tf.abs(q[1] - p[1]))
        return prev*g1*g2

    def bl_sample_mapped(self, p, features, displ):
        b = tf.fill([self.mem_w, self.mem_h], True)
        ind_q = tf.where(b)
        dp = displ[p[0], p[1]]
        channel = p[2]
        p = tf.to_float(p)
        upd = tf.add(p[:2], dp)
        out = tf.map_fn(lambda q:
                        self.bl_kernel(q, upd, channel, features),
                        ind_q,
                        dtype=tf.float32)
        return tf.reduce_sum(out)

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
        # self.memory = tf.random_normal([self.mem_h, self.mem_w, self.df],
        #                                mean=0, stddev=1)

        # print(inputs.shape)
        # for frame in inputs.get_shape()[1]:
        # for frame in range(inputs.shape[1]):
        #     self.iterate(inputs[0][frame])

        if _DEBUG: print(inputs)
        initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
        memory = tf.scan(self.iterate, inputs[0], initializer=initial_state)
        return memory


