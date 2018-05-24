import tensorflow as tf
import tempfile
import numpy as np

class InceptionResNetV2():
    """
    A class that builds a TF graph with a pre-trained VGG19 model (on imagenet)
    Also takes care of preprocessing. Input should be a regular RGB image (0-255)
    """

    def __init__(self, input_tensor=None):
        self._build_graph(input_tensor)

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope('IRV2'):
                with tf.name_scope('inputs'):
                    self.input_tensor = input_tensor

                with tf.variable_scope('model'):
                    self.irv2 = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                                                        input_tensor=self.input_tensor)

                self.outputs = {l.name: l.output for l in self.irv2.layers}

            self.irv2_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='IRV2/model')

            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(self.irv2_weights).save(sess, f.name)

        self.model_weights_tensors = set(self.irv2_weights)

    def load_weights(self):
        sess = tf.get_default_session()
        tf.train.Saver(self.irv2_weights).restore(sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.outputs[key]


if __name__ == '__main__':
    my_img = tf.random_uniform([64, 299, 299, 3], maxval=255.0)
    irv2 = InceptionResNetV2(input_tensor=my_img)
    output = tf.identity(irv2['mixed_6a'], name='my_output')
    with tf.Session() as sess:
        irv2.load_weights()
        output_val = sess.run(output)