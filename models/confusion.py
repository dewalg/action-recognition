import numpy as np
import sys
import os
import time
import i3d
import tensorflow as tf
from pipeline import Pipeline
from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../config/config.ini')

NUM_CLASSES = 50

CLS_DICT_FP = config['paths']['inf_cat']
DATA = config['paths']['inference_fp']

rgb_ckpt = config['checkpoint']['rgb']
rgb_imgnet_ckpt = config['checkpoint']['rgb_imagenet']

# build the model
def inference(rgb_inputs):
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(rgb_inputs, is_training=False, dropout_keep_prob=1.0)

    model_predictions = tf.nn.softmax(rgb_logits)
    return rgb_logits, model_predictions


if __name__ == '__main__':
    data_pipeline = Pipeline(DATA, CLS_DICT_FP)
    NUM_VAL_VIDS = data_pipeline.getNumVids()

    train_queue = data_pipeline.get_dataset()
    train_iterator = tf.contrib.data.Iterator.from_structure(train_queue.output_types, train_queue.output_shapes)
    train_init_op = train_iterator.make_initializer(train_queue)

    rgbs, label = train_iterator.get_next()
    logits, preds = inference(rgbs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_saver.restore(sess, rgb_ckpt)

        sess.run(train_init_op)
        preds = []
        labels = []
        while True:
            try:
                out_logits, out_preds, label = sess.run([logits, preds, label])

                out_logits = out_logits[0]
                out_preds = out_preds[0]
                sorted = np.argsort(out_preds)[::-1]
                cls_idx = sorted[0]
                preds.append(cls_idx)
                labels.append(label)

            except tf.errors.OutOfRangeError as e:
                break
            except Exception as e:
                print(e)
                sys.exit(1)

        confusion = tf.confusion_matrix(labels=labels, predictions=preds, num_classes=NUM_CLASSES)
        con = sess.run(confusion)
        print(con)
        np.save('confusion.npy', con)
        np.save('preds.npy', preds)
        np.save('labels.npy', labels)


