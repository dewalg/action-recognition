import sys
import os
import time
from comet_ml import Experiment
from clocknet.clock_step import ClockStep
from clocknet.resnet.inception_resnet_v2_wrapper import InceptionResNetV2
from clocknet.clock_flow import ClockFlow
import tensorflow as tf
from pipeline import Pipeline
from configparser import ConfigParser, ExtendedInterpolation

#
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('../config/config.ini')

NUM_CLASSES = config['hp'].getint('num_classes')
NUM_FRAMES = config['hp'].getint('num_frames')
CROP_SIZE = config['hp'].getint('crop_size')
BATCH_SIZE = config['hp'].getint('batch_size')
STRIDE = NUM_FRAMES
CLS_DICT_FP = config['paths']['cls_dict_fp']
DROPOUT_KEEP_PROB = config['hp'].getfloat('dropout_keep_prob')
MAX_EPOCH = config['hp'].getint('max_epoch')
NUM_GPUS = config['hp'].getint('num_gpus')

TRAIN_DATA = config['paths']['train_fp']
VAL_DATA = config['paths']['val_fp']

'''
CHECKPOINT_PATHS = {
    'rgb': './checkpoints/rgb_scratch/model.ckpt',
    'rgb_imagenet': './checkpoints/rgb_imagenet/model.ckpt',
}
'''

LR = config['hp'].getfloat('lr')
TMPDIR = config['paths']['tmpdir']
LOGDIR = config['paths']['logdir']
SAVE_ITER = config['iter'].getint('save_iter')
VAL_ITER = config['iter'].getint('val_iter')
DISPLAY_ITER = config['iter'].getint('display_iter')
SHUFFLE_SIZE = config['iter'].getint('shuffle_buffer')


# build the model
def get_true_counts(tower_logits_labels):
    true_count = 0
    for logits, labels in tower_logits_labels:
        true_count += tf.reduce_sum(
            tf.cast(
                tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels),
                tf.int32
            )
        )
    return true_count


if __name__ == '__main__':

    hyper_params = {"learning_rate": LR,
                    "num_classes": NUM_CLASSES,
                    "num_frames": NUM_FRAMES,
                    "dropout_keep_prob": DROPOUT_KEEP_PROB,
                    "batch_size": BATCH_SIZE,
                    "shuffle_size": SHUFFLE_SIZE,
                    }

    """  this is user-sensitive API key. Change it to see logs in your comet-ml """
    experiment = Experiment(api_key="5t7sqKGYmr76wqaEHqwN0Sqcg", project_name="lsar")
    experiment.log_multiple_params(hyper_params)
    """ =================================================================== """
    train_pipeline = Pipeline(TRAIN_DATA, CLS_DICT_FP)
    val_pipeline = Pipeline(VAL_DATA, CLS_DICT_FP)
    NUM_VAL_VIDS = val_pipeline.getNumVids()

    is_training = tf.placeholder(tf.bool)

    # opt = tf.train.GradientDescentOptimizer(LR)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    tower_grads = []
    tower_losses = []
    tower_logits_labels = []

    train_queue = train_pipeline.get_dataset().shuffle(buffer_size=SHUFFLE_SIZE).batch(BATCH_SIZE)
    train_iterator = tf.data.Iterator.from_structure(train_queue.output_types, train_queue.output_shapes)
    train_init_op = train_iterator.make_initializer(train_queue)

    val_queue = val_pipeline.get_dataset().shuffle(buffer_size=SHUFFLE_SIZE).batch(BATCH_SIZE)
    val_iterator = tf.data.Iterator.from_structure(val_queue.output_types, val_queue.output_shapes)
    val_init_op = val_iterator.make_initializer(val_queue)

    model = ClockStep(num_classes=NUM_CLASSES)

    rgbs, labels = tf.cond(is_training, lambda: train_iterator.get_next(),
                           lambda: val_iterator.get_next())
    rgb_reshaped = tf.reshape(rgbs, [64, 299, 299, 3])

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        with tf.device('/gpu:0'):
            resnet = InceptionResNetV2(input_tensor=rgb_reshaped)
            resnet_out = tf.identity(resnet['mixed_6a'], name='rgb_resnet_out')

            flow = ClockFlow(inputs=rgb_reshaped)
            flow_out = flow.get_flow()
            # flow_out = tf.random_uniform([64, 17, 17, 2])

            model = ClockStep(num_classes=10)
            logits = model._build(rgbs, resnet_out, flow_out)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            tf.get_variable_scope().reuse_variables()
            grads = opt.compute_gradients(loss)
            tower_logits_labels.append((logits, labels))

    true_count_op = get_true_counts(tower_logits_labels)

    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    train_op = opt.apply_gradients(grads)
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    summary_op = tf.summary.merge(summaries)

    if not os.path.exists(TMPDIR):
        os.mkdir(TMPDIR)
    saver = tf.train.Saver(max_to_keep=3)
    ckpt_path = os.path.join(TMPDIR, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    with tf.Session() as sess:
        # saver for fine tuning

        sess.run(train_init_op)
        sess.run(val_init_op)
        sess.run(tf.global_variables_initializer())
        resnet.load_weights()
        flow.load_weights()
        experiment.set_model_graph(sess.graph)

        # rgb_def_state = get_pretrained_save_state()
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('Restoring from: %s', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
        else:
            tf.logging.info('No checkpoint file found, restoring pretrained weights...')
            # rgb_def_state.restore(sess, CHECKPOINT_PATHS['rgb_imagenet'])
            # rgb_def_state.restore(sess, CHECKPOINT_PATHS['rgb'])
            # tf.logging.info('Restore Complete.')

        # prefetch_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        tf.logging.set_verbosity(tf.logging.INFO)

        it = 0
        last_time = time.time()
        last_step = 0
        val_time = 0
        for epoch in range(MAX_EPOCH):
            # sess.run(train_init_op)
            while True:
                # print('==== EPOCH : ' + str(epoch) + ' || iter : ' + str(it))
                try:
                    _, loss_val = sess.run([train_op, loss], feed_dict={is_training: True})

                    # logging into comet-ml
                    experiment.log_metric("train_loss", loss_val, step=it)

                    if it % DISPLAY_ITER == 0:
                        tf.logging.info('step %d, loss = %f', it, loss_val)
                        loss_summ = tf.Summary(value=[
                            tf.Summary.Value(tag="train_loss", simple_value=loss_val)
                        ])

                        summary_writer.add_summary(loss_summ, it)
                        summary_str = sess.run(summary_op, feed_dict={is_training: True})
                        summary_writer.add_summary(summary_str, it)

                    if it % SAVE_ITER == 0 and it > 0:
                        saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)

                    if it % VAL_ITER == 0 and it > 0:
                        sess.run(val_init_op)
                        val_start = time.time()
                        tf.logging.info('validating...')
                        true_count = 0
                        val_loss = 0
                        while True:
                            try:
                                c, l = sess.run([true_count_op, loss], {is_training: False})
                                true_count += c
                                val_loss += l
                            except tf.errors.OutOfRangeError as e:
                                break
                        # add val accuracy to summary
                        acc = true_count / NUM_VAL_VIDS
                        tf.logging.info('val accuracy: %f', acc)
                        acc_summ = tf.Summary(value=[
                            tf.Summary.Value(tag="val_acc", simple_value=acc)
                        ])
                        summary_writer.add_summary(acc_summ, it)
                        # logging into comet-ml
                        experiment.log_metric("val_acc", acc, step=it)
                        # add val loss to summary
                        val_loss = val_loss / int(NUM_VAL_VIDS / NUM_GPUS / BATCH_SIZE)
                        tf.logging.info('val loss: %f', val_loss)
                        val_loss_summ = tf.Summary(value=[
                            tf.Summary.Value(tag="val_loss", simple_value=val_loss)
                        ])
                        summary_writer.add_summary(val_loss_summ, it)

                        # logging into comet-ml
                        # experiment.log_metric("val_loss", val_loss, step=it)
                        val_time = time.time() - val_start
                        saver.save(sess, os.path.join(ckpt_path, 'model_ckpt'), it)

                    it += 1

                except tf.errors.OutOfRangeError as e:
                    break
                except Exception as e:
                    print(e)
                    sys.exit(1)

        summary_writer.close()