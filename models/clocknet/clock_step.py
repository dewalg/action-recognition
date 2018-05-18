'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''
# using python 3

import time
import tensorflow as tf
import sonnet as snt

from clocknet.clock_flow import ClockFlow
from clocknet.clock_rgb import ClockRgb

# debug flag for debugging outputs
_DEBUG = True

H_f = W_f = 17
D_f = 1088


class ClockStep(snt.AbstractModule):
    def __init__(self, num_classes, name='clockstep'):
        super(ClockStep, self).__init__(name=name)
        self.num_classes = num_classes
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f

    def compute_mem(self, memory, flow):
        memory = self.bl_sample(memory, flow)
        return memory

    @staticmethod
    def aggregate(memory, features):
        return tf.multiply(0.5, tf.add(memory, features))

    def iterate(self, memory, flow_rgb):

        (flow, rgb) = flow_rgb
        if _DEBUG: print("Debug: ClockStep = FLOW SHAPE = ", flow.shape)
        if _DEBUG: print("Debug: ClockStep = RGBS SHAPE = ", rgb.shape)
        start_time = time.time()
        memory = self.compute_mem(memory, flow)
        if _DEBUG: print("Debug: ClockStep = AFTER COMPUTE MEM = ", memory.shape)
        if _DEBUG: print("Debug: ClockStep = %s : finished memory computation" % (time.time() - start_time))

        start_time = time.time()
        memory = self.aggregate(memory, rgb)
        if _DEBUG: print("Debug: ClockStep = %s : finished memory aggregation" % (time.time() - start_time))
        return memory

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
        return prev * g1 * g2

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
        if _DEBUG: print("DEBUG: ClockStep = INPUTS SHAPE = ", inputs.shape)
        clock_flow = ClockFlow(num_classes=10)
        flows = clock_flow._build(inputs)
        
        # clock_rgb = ClockRgb(num_classes=10)
        # rgbs = clock_rgb._build(inputs)
        rgbs = tf.random_normal([64, 17, 17, 1088], mean=0, stddev=1)
        initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
        memory = tf.scan(self.iterate, (flows, rgbs), initializer=initial_state)
        return memory
