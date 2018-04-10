from __future__ import absolute_import
import sonnet as snt
import tensorflow as tf
from tensorflow.python.keras import layers

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


class ConvBlock(snt.AbstractModule):
    def __init__(self, output_channels=(1, 1, 1),
                 kernel_shape=(1, 1, 1),
                 stride=[2, 2, 2],
                 padding=snt.SAME,
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='conv_block'):
        """Initializes SepConv module."""
        super(ConvBlock, self).__init__(name=name)
        self.name = name
        self._output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        net = Unit3D(output_channels=self._output_channels[0],
                     kernel_shape=[1, 1, 1],
                     stride=self.stride,
                     padding=self.padding,
                     use_bias=self._use_bias,
                     name=self.name+"_1")(inputs, is_training=is_training)

        net = SepConv(output_channels=self._output_channels[1],
                      kernel_shape=self.kernel_shape,
                      padding=snt.SAME,
                      name=self.name+"_2")(net, is_training=is_training)

        net = snt.Conv3D(output_channels=self._output_channels[2],
                         kernel_shape=[1, 1, 1],
                         padding=self.padding,
                         use_bias=self._use_bias,
                         name=self.name+"_3")(net)
        net = snt.BatchNorm()(net, is_training=is_training, test_local_stats=False)

        proj = snt.Conv3D(output_channels=self._output_channels[2],
                          kernel_shape=[1, 1, 1],
                          stride=self.stride,
                          padding=self.padding,
                          use_bias=self._use_bias,
                          name=self.name+"_4")(inputs)

        proj = snt.BatchNorm()(proj, is_training=is_training, test_local_stats=False)
        net = layers.add([net, proj])
        net = self._activation_fn(net)

        return net


class IdentityBlock(snt.AbstractModule):
    def __init__(self, output_channels=(1, 1, 1),
                 kernel_shape=(1, 1, 1),
                 padding=snt.SAME,
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='id_block'):
        super(IdentityBlock, self).__init__(name=name)
        self.name = name
        self._output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.padding = padding
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        net = Unit3D(output_channels=self._output_channels[0],
                     kernel_shape=[1, 1, 1],
                     padding=self.padding,
                     use_bias=self._use_bias,
                     name=self.name+"_1")(inputs, is_training=is_training)

        net = SepConv(output_channels=self._output_channels[1],
                      kernel_shape=self.kernel_shape,
                      padding=snt.SAME,
                      name=self.name+"_2")(net, is_training=is_training)

        net = snt.Conv3D(output_channels=self._output_channels[2],
                         kernel_shape=[1, 1, 1],
                         padding=self.padding,
                         use_bias=self._use_bias,
                         name=self.name+"_3")(net)

        net = snt.BatchNorm()(net, is_training=is_training, test_local_stats=False)
        net = layers.add([net, inputs])
        net = self._activation_fn(net)

        return net


class ResNet(snt.AbstractModule):

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_0_1x7x7',
        'MaxPool3d_1x3x3',
        'conv_block_2a',
        'id_block_2b',
        'id_block_2c',
        'conv_block_3a',
        'id_block_3b',
        'id_block_3c',
        'id_block_3d',
        'conv_block_4a',
        'id_block_4b',
        'id_block_4c',
        'id_block_4d',
        'id_block_4e',
        'id_block_4f',
        'conv_block_5a',
        'id_block_5b',
        'id_block_5c',
        'AvgPool3d_1x7x7',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='ResNet-50'):
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

        super(ResNet, self).__init__(name=name)
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
        end_point = 'Conv3d_0_1x7x7'
        net = SepConv(output_channels=64, kernel_shape=[7, 7, 7], stride=[2, 2, 2], name=end_point, padding=snt.SAME)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_1x3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding=snt.SAME, name=end_point)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv_block_2a'
        net = ConvBlock(output_channels=[64, 64, 256], kernel_shape=[5, 3, 3], stride=[1, 1, 1], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_2b'
        net = IdentityBlock(output_channels=[64, 64, 256], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_2c'
        net = IdentityBlock(output_channels=[64, 64, 256], kernel_shape=[1, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv_block_3a'
        net = ConvBlock(output_channels=[128, 128, 512], kernel_shape=[7, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_3b'
        net = IdentityBlock(output_channels=[128, 128, 512], kernel_shape=[5, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_3c'
        net = IdentityBlock(output_channels=[128, 128, 512], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_3d'
        net = IdentityBlock(output_channels=[128, 128, 512], kernel_shape=[1, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'conv_block_4a'
        net = ConvBlock(output_channels=[256, 256, 1024], kernel_shape=[7, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_4b'
        net = IdentityBlock(output_channels=[256, 256, 1024], kernel_shape=[5, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_4c'
        net = IdentityBlock(output_channels=[256, 256, 1024], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_4d'
        net = IdentityBlock(output_channels=[256, 256, 1024], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_4e'
        net = IdentityBlock(output_channels=[256, 256, 1024], kernel_shape=[5, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_4f'
        net = IdentityBlock(output_channels=[256, 256, 1024], kernel_shape=[7, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points


        end_point = 'conv_block_5a'
        net = ConvBlock(output_channels=[512, 512, 2048], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_5b'
        net = IdentityBlock(output_channels=[512, 512, 2048], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'id_block_5c'
        net = IdentityBlock(output_channels=[512, 512, 2048], kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'AvgPool3d_1x7x7'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID, name=end_point)

        if DEBUG: print(end_point + ":\t\t" + str(net.shape))

        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            logits = Unit3D(output_channels=self._num_classes,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        end_points[end_point] = averaged_logits
        if self._final_endpoint == end_point: return averaged_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averaged_logits)
        end_points[end_point] = predictions
        return predictions, end_points

