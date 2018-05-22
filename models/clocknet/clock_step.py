'''
Clocknet is based on the paper "Memory Warps for Learning Long-Term Online
Video Representations" by Vu et al.
'''
# using python 3

import time
import tensorflow as tf
import sonnet as snt

# from clock_flow import ClockFlow
from clock_rgb import ClockRgb

# debug flag for debugging outputs
_DEBUG = True

H_f = W_f = 17
D_f = 1088


class ClockStep():
    def __init__(self, num_classes, name='clockstep'):
        self.num_classes = num_classes
        self.mem_h = H_f
        self.mem_w = W_f
        self.df = D_f

        # self.clock_flow = ClockFlow(num_classes=10)
        self.clock_rgb = ClockRgb(num_classes=10)

    def compute_mem(self, memory, flow):
        x_flow = tf.slice(flow, [0, 0, 0], [-1, -1, 1])
        y_flow = tf.slice(flow, [0, 0, 1], [-1, -1, 1])
        print("FLOW DIM")
        print(x_flow)
        memory = self.bilinear_sampler(memory, x_flow, y_flow)
        return memory

    @staticmethod
    def aggregate(memory, features):
        return tf.multiply(0.5, tf.add(memory, features))

    def iterate(self, memory, flow_rgb):

        (flow, rgb) = flow_rgb
        if _DEBUG: print("Debug: ClockStep = FLOW SHAPE = ", flow.shape)
        if _DEBUG: print("Debug: ClockStep = RGBS SHAPE = ", rgb.shape)
        start_time = time.time()
        # memory = tf.random_normal([17, 17, 1088], mean=0, stddev=1)
        memory = self.compute_mem(memory, flow)
        if _DEBUG: print("Debug: ClockStep = AFTER COMPUTE MEM = ", memory.shape)
        if _DEBUG: print("Debug: ClockStep = %s : finished memory computation" % (time.time() - start_time))

        start_time = time.time()
        memory = self.aggregate(memory, rgb)
        if _DEBUG: print("Debug: ClockStep = %s : finished memory aggregation" % (time.time() - start_time))
        return memory

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def bilinear_sampler(self, im, x, y):
        """
        BL sampler
        :param im: expects a [batch x height x width x channels] tensor 
        :param x: flow in x direction [height x width x 1]
        :param y: flow in y direction [height x width x 1]
        :return: BL sampled tensor in shape [batch x height x width x channels] tensor 
        """
        im = tf.expand_dims(im, 0)
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = height
        out_width = width
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = self._repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, [-1, channels])
        # im_flat = tf.reshape(im, [-1, 10])
        # im_flat = tf.Print(im_flat, [im_flat.shape])
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return tf.reshape(output, [out_height, out_width, channels])

    def _build(self, inputs):
        if _DEBUG: print("DEBUG: ClockStep = INPUTS SHAPE = ", inputs.shape)
        # flows = self.clock_flow._build(inputs)
        flows = tf.random_uniform([64, 17, 17, 2], maxval=1)
        rgbs = self.clock_rgb._build(inputs)
        initial_state = tf.zeros([self.mem_w, self.mem_h, self.df])
        memory = tf.scan(self.iterate, (flows, rgbs), initializer=initial_state)
        return memory

if __name__ == '__main__':
    model = ClockStep(num_classes=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer)
        summary_writer = tf.summary.FileWriter("./tmp", sess.graph)
