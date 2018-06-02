# using python 3

import tensorflow as tf
from .flownet.src.flownet2 import flownet2
import tempfile

# debug flag for debugging outputs
_DEBUG = False

H_f = W_f = 17
D_f = 2


class ClockFlow:
    def __init__(self, inputs=None):
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f
        # load FlowNet checkpoint.
        # self.net = flownet2.FlowNet2()
        self.prev_frame = tf.multiply(tf.ones([299, 299, 3]), 255.0)
        self._build(inputs)

    def load_weights(self):
        sess = tf.get_default_session()
        self.net.load_ckpt(sess, self.flow_weights)

    def iterate(self, memory, frame):
        if _DEBUG: print("CLOCK_FLOW debug: frame shape = ", frame.shape)
        features = self.net.compute_flow(self.prev_frame, frame)
        features = tf.reshape(features, [self.mem_h, self.mem_w, self.df])
        self.prev_frame = frame
        return features

    def _build(self, inputs):
        with tf.Session() as sess:
            with tf.variable_scope('Flow'):
                with tf.name_scope('inputs'):
                    if inputs is None:
                        inputs = tf.placeholder(tf.float32, shape=self._shape, name='input_flow')

                    self.inputs = inputs
                    self.prev_frame = self.inputs[0]

                with tf.variable_scope('model'):
                    self.net = flownet2.FlowNet2()
                    if _DEBUG: print("CLOCK_FLOW debug: inputs shape = ", inputs.shape)
                    initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
                    memory = tf.scan(self.iterate, inputs, initializer=initial_state)
                    self.out = memory

            self.flow_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Flow/model')
            # with tempfile.NamedTemporaryFile() as f:
            #     self.tf_flow_ckpt_file = tf.train.Saver(self.flow_weights).save(sess, f.name)

    def save_weights(self):
        sess = tf.get_default_session()
        with tempfile.NamedTemporaryFile() as f:
            return tf.train.Saver(self.flow_weights).save(sess, f.name)

    def get_flow(self):
        return self.out
