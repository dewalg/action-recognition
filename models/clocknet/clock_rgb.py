'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''

# using python 3
import numpy as np
import tensorflow as tf
from .resnet import inception_resnet_v2_wrapper

_DEBUG = False

H_f = W_f = 17
D_f = 1088


class ClockRgb:
    def __init__(self, num_classes = 10):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        self.num_classes = num_classes
        self.input_tensor = tf.placeholder(tf.float32, (64, 299, 299, 3), name='rgb_image')
        self.model = inception_resnet_v2_wrapper.InceptionResNetV2(input_tensor=self.input_tensor)

    def load_ckpt(self):
        sess = tf.get_default_session()
        self.model.load_weights(sess)

    # def call(self, inputs):
    #     # inputs is a 299x299x3 tensor
    #     # should return a 17x17x1088 tensor
    #     inputs = tf.reshape(inputs, [1, 299, 299, 3])
    #     out = self.model(inputs)
    #     out = tf.reshape(out, [17, 17, 1088])
    #     return out

    def call_batch(self):
        # inputs is a dx299x299x3 tensor
        # should return a dx17x17x1088 tensor
        # print(inputs)
        return tf.identity(self.model['mixed_6a'], name='my_output')

    def _build(self, inputs):
        if _DEBUG: print("CLOCK_RGB debug: inputs shape = ", inputs.shape)
        out = self.call_batch()
        if _DEBUG: print("CLOCK_RGB debug: outputs shape = ", out.shape)
        sess = tf.get_default_session()
        np_inputs = inputs.eval(session=sess)
        np_inputs = np_inputs.astype(np.float32)
        print("CLOCK_RGB debug: outputs shape ", np_inputs.shape)
        print("CLOCK_RGB debug: outputs shape ", type(np_inputs))
        fd = {self.input_tensor: np_inputs}
        sess.run(out, fd)
        return out


# class ClockRgb_orig:
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.mem_h = H_f
#         self.mem_w = W_f
#         self.df = D_f
#         self.resnet = Resnet(self.num_classes)
#
#     def iterate(self, memory, frame):
#         """
#         Iterate does required computations on a frame level
#         :param memory: the current memory
#         :param frame: the frame (or time step) currently being processed
#         :return: void - updates the memory
#         """
#         if _DEBUG: print("CLOCK_RGB debug: frame shape = ", frame.shape)
#
#         start_time = time.time()
#         if _DEBUG: print("CLOCK_RGB debug: starting iteration")
#         features = self.resnet.call(frame)
#         if _DEBUG: print("CLOCK_RGB debug: = %s : finished resnet features" % (time.time() - start_time))
#         return features
#
#     def _build(self, inputs):
#         if _DEBUG: print("CLOCK_RGB debug: inputs shape = ", inputs.shape)
#         initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
#         # memory = tf.scan(self.iterate, inputs[0], initializer=initial_state)
#         return self.res
#         return memory
#

