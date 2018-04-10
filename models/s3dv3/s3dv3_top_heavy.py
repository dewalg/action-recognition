from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

DEBUG = True

class Unit3D(snt.AbstractModule):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=snt.SAME,
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self.padding = padding
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        """Connects the module to inputs.

        Args:
          inputs: Inputs to the Unit3D component.
          is_training: whether to use training mode for snt.BatchNorm (boolean).

        Returns:
          Outputs from the module.
        """
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=self.padding,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class SepConv(snt.AbstractModule):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=[1, 1, 1],
                 padding=snt.SAME,
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='sep_conv'):
        """Initializes SepConv module."""
        super(SepConv, self).__init__(name=name)
        self._output_channels = output_channels

        # build appropriate kernels for stride and convolution
        # for the spatial domain [1, x, y]
        self._sp_kernel_shape = [1] + kernel_shape[1:]
        self._sp_stride_shape = [1] + stride[1:]

        # do the same for the temporal domain convolution
        self._temp_kernel_shape = [kernel_shape[0]] + [1, 1]
        self._temp_stride_shape = [stride[0]] + [1, 1]

        self.padding = padding
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        """Connects the module to inputs.

        Args:
          inputs: Inputs to the SepConv component.
          is_training: whether to use training mode for snt.BatchNorm (boolean).

        Returns:
          Outputs from the module.
        """

        intermediate = snt.Conv3D(output_channels=self._output_channels,
                                  kernel_shape=self._sp_kernel_shape,
                                  stride=self._sp_stride_shape,
                                  padding=self.padding,
                                  use_bias=self._use_bias)(inputs)
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._temp_kernel_shape,
                         stride=self._temp_stride_shape,
                         padding=self.padding,
                         use_bias=self._use_bias)(intermediate)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class s3d(snt.AbstractModule):
    """Inception-v1 I3D architecture.

    The model is introduced in:

      Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
      Joao Carreira, Andrew Zisserman
      https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception architecture, introduced in:

      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_0_1x3x3',
        'Conv3d_1_1x3x3',
        'Conv3d_2_3x3x3',
        'MaxPool3d_1_3x3x3',
        'Conv3d_3_1x1x1',
        'Conv3d_4_3x3x3',
        'MaxPool3d_2_1x3x3'
        'Mixed0_35x35x256',
        'Mixed1_35x35x288',
        'Mixed2_35x35x288',
        'Mixed3_17x17x480',
        'Mixed4_17x17x768',
        'Mixed5_17x17x768',
        'Mixed6_17x17x768',
        'Mixed7_17x17x768',
        #'AuxLogits',
        'Mixed8_8x8x1280',
        'Mixed9_8x8x1280',
        'Mixed10_8x8x1280',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d'):
        """Initializes I3D model instance.

        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.

        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(s3d, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, is_training, dropout_keep_prob=1.0):
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

        Raises:
          ValueError: if `self._final_endpoint` is not recognized.
        """
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}

        end_point = 'Conv3d_0_1x3x3'
        net = SepConv(output_channels=32, kernel_shape=[1, 3, 3], stride=[2, 2, 2], name=end_point, padding=snt.VALID)(net, is_training=is_training)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Conv3d_1_1x3x3'
        net = SepConv(output_channels=32, kernel_shape=[1, 3, 3], name=end_point, padding=snt.VALID)(net, is_training=is_training)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Conv3d_2_3x3x3'
        net = SepConv(output_channels=64, kernel_shape=[3, 3, 3], name=end_point, padding=snt.SAME)(net, is_training=is_training)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_1_1x3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding=snt.VALID, name=end_point)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Conv3d_3_1x1x1'
        net = Unit3D(output_channels=80, kernel_shape=[1, 1, 1], name=end_point, padding=snt.VALID)(net, is_training=is_training)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Conv3d_4_3x3x3'
        net = SepConv(output_channels=192, kernel_shape=[3, 3, 3], name=end_point, padding=snt.VALID)(net, is_training=is_training)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_2_1x3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding=snt.VALID, name=end_point)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        # inception blocks

        end_point = 'Mixed0_35x35x256'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=64, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0c_3x3x3')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed1_35x35x288'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=64, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0c_3x3x3')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed2_35x35x288'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=64, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0c_3x3x3')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed3_17x17x480'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = SepConv(output_channels=384, kernel_shape=[3, 3, 3], stride=[2, 2, 2], padding=snt.VALID, name='Conv3d_0a_3x3x3')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3x3')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=96, kernel_shape=[3, 3, 3], stride=[2, 2, 2], padding=snt.VALID, name='Conv3d_0c_3x3x3')(branch_1, is_training=is_training)
            net = tf.concat([branch_0, branch_1], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed4_17x17x768'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=128, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x1x7')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x7x1')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=128, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=128, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x1x7')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=128, kernel_shape=[1, 1, 7], name='Conv3d_0d_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0e_1x1x7')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed5_17x17x768'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=160, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x1x7')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x7x1')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=160, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=160, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x1x7')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=160, kernel_shape=[1, 1, 7], name='Conv3d_0d_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0e_1x1x7')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed6_17x17x768'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=160, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x1x7')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x7x1')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=160, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=160, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x1x7')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=160, kernel_shape=[1, 1, 7], name='Conv3d_0d_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0e_1x1x7')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed7_17x17x768'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x1x7')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x7x1')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 1, 7], name='Conv3d_0b_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0c_1x1x7')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 1, 7], name='Conv3d_0d_1x7x1')(branch_2, is_training=is_training)
                branch_2 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0e_1x1x7')(branch_2, is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        # end_point = 'AuxLogits'
        # aux_logits = tf.identity(net)
        # with tf.variable_scope(end_point):
        #     aux_logits = tf.nn.avg_pool3d(aux_logits, ksize=[1, 1, 5, 5, 1], strides=[1, 1, 3, 3, 1], padding=snt.VALID)
        #     aux_logits = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(aux_logits, is_training=is_training)
        #     aux_shape = [1]
        #     aux_shape.extend(aux_logits.get_shape()[1:3])
        #     aux_logits = Unit3D(output_channels=768, kernel_shape=aux_shape, padding=snt.VALID)
        #     aux_logits = tf.layers.flatten(aux_logits)
        #     aux_logits = tf.contrib.layers.fully_connected(aux_logits, self._num_classes)
        # end_points[end_point] = aux_logits
        # if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed8_8x8x1280'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_0 = SepConv(output_channels=320, kernel_shape=[3, 3, 3], stride=[2, 2, 2], padding=snt.VALID, name='Conv3d_0b_3x3x3')(branch_0, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 7, 1], name='Conv3d_0b_1x7x1')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[1, 1, 7], name='Conv3d_0c_1x1x7')(branch_1, is_training=is_training)
                branch_1 = SepConv(output_channels=192, kernel_shape=[3, 3, 3], stride=[2, 2, 2], padding=snt.VALID, name='Conv3d_0d_3x3x3')(branch_1, is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding=snt.VALID, name='MaxPool3d_0a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed9_8x8x1280'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=320, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1_1 = SepConv(output_channels=384, kernel_shape=[1, 1, 3], name='Conv3d_0b_1x1x3')(branch_1, is_training=is_training)
                branch_1_2 = SepConv(output_channels=384, kernel_shape=[1, 3, 1], name='Conv3d_0c_1x3x1')(branch_1, is_training=is_training)
                branch_1 = tf.concat([branch_1_1, branch_1_2], 4)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=448, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=384, kernel_shape=[3, 3, 3], name='Conv3d_0b_1x1x3')(branch_2, is_training=is_training)
                branch_2_1 = SepConv(output_channels=384, kernel_shape=[1, 1, 3], name='Conv3d_0c_1x1x3')(branch_2, is_training=is_training)
                branch_2_2 = SepConv(output_channels=384, kernel_shape=[1, 3, 1], name='Conv3d_0c_1x3x1')(branch_2, is_training=is_training)
                branch_2 = tf.concat([branch_2_1, branch_2_2], 4)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed10_8x8x1280'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=320, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_1_1 = SepConv(output_channels=384, kernel_shape=[1, 1, 3], name='Conv3d_0b_1x1x3')(branch_1, is_training=is_training)
                branch_1_2 = SepConv(output_channels=384, kernel_shape=[1, 3, 1], name='Conv3d_0c_1x3x1')(branch_1, is_training=is_training)
                branch_1 = tf.concat([branch_1_1, branch_1_2], 4)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=448, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1x1')(net, is_training=is_training)
                branch_2 = SepConv(output_channels=384, kernel_shape=[3, 3, 3], name='Conv3d_0b_1x1x3')(branch_2, is_training=is_training)
                branch_2_1 = SepConv(output_channels=384, kernel_shape=[1, 1, 3], name='Conv3d_0c_1x1x3')(branch_2, is_training=is_training)
                branch_2_2 = SepConv(output_channels=384, kernel_shape=[1, 3, 1], name='Conv3d_0c_1x3x1')(branch_2, is_training=is_training)
                branch_2 = tf.concat([branch_2_1, branch_2_2], 4)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.avg_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding=snt.SAME)
                branch_3 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1x1')(branch_3, is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        if DEBUG: print(end_point + ":\t\t" + str(net.shape))
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool3d(net, ksize=[1, 2, 5, 5, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)
            net = tf.nn.dropout(net, dropout_keep_prob)
            if DEBUG: print(end_point + ":\t\t" + str(net.shape))
            logits = Unit3D(output_channels=self._num_classes, kernel_shape=[1, 1, 1], activation_fn=None, use_batch_norm=False, use_bias=True, name='Conv3d_0c_1x1x1', padding=snt.VALID)(net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        end_points[end_point] = averaged_logits
        if self._final_endpoint == end_point: return averaged_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averaged_logits)
        end_points[end_point] = predictions
        return predictions, end_points
