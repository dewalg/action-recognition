'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''

# using python 3

import numpy as np
import time
import tensorflow as tf
import sonnet as snt

from .resnet import inception_resnet_v2_keras

_DEBUG = True

H_f = W_f = 17
D_f = 1088


class Resnet:
    def __init__(self, num_classes = 10):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.num_classes = num_classes
        self.model = inception_resnet_v2_keras.InceptionResNetV2()

    def call(self, inputs):
        # inputs is a 299x299x3 tensor
        # should return a 17x17x1088 tensor
        return self.model.predict(np.array([inputs]))[0]
        # inputs = tf.reshape(inputs, [1, 299, 299, 3])
        # out, _ = inception_resnet_v2_tf.inception_resnet_v2(inputs, num_classes=self.num_classes, dropout_keep_prob=0.5, final_endpoint='Mixed_6a')
        # out = tf.reshape(out, [17, 17, 1088])
        # print("RETURNING FROM RESNET...********")
        # return out

    def call_batch(self, inputs):
        # inputs is a dx299x299x3 tensor
        # should return a dx17x17x1088 tensor
        return self.model.predict(inputs)


class ClockRgb(snt.AbstractModule):
    def __init__(self, num_classes, name='clocknet'):
        super(ClockRgb, self).__init__(name=name)
        self.num_classes = num_classes
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.resnet = Resnet(self.num_classes)

    def iterate(self, memory, frame):
        """
        Iterate does required computations on a frame level
        :param memory: the current memory
        :param frame: the frame (or time step) currently being processed
        :return: void - updates the memory
        """
        if _DEBUG: print("CLOCK_RGB debug: frame shape = ", frame.shape)

        start_time = time.time()
        if _DEBUG: print("CLOCK_RGB debug: starting iteration")
        features = self.resnet.call(frame)
        if _DEBUG: print("CLOCK_RGB debug: = %s : finished resnet features" % (time.time() - start_time))
        return features

    def _build(self, inputs):
        if _DEBUG: print("CLOCK_RGB debug: inputs shape = ", inputs.shape)
        initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
        memory = tf.scan(self.iterate, inputs[0], initializer=initial_state)
        return memory


