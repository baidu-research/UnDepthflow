# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from monodepth_dataloader import MonodepthDataloader
from models import *

from eval.evaluate_flow import load_gt_flow_kitti
from eval.evaluate_mask import load_gt_mask
from loss_utils import average_gradients

from test import test

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

FLAGS = flags.FLAGS

flags.DEFINE_string('trace', "./", 'directory for model checkpoints.')
flags.DEFINE_integer('num_iterations', 300000,
                     'number of training iterations.')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')
flags.DEFINE_string(
    'mode', '',
    'selection from four modes of ["flow", "depth", "depthflow", "stereo"]')
flags.DEFINE_string('train_test', 'train', 'whether to train or test')
flags.DEFINE_boolean("retrain", True, "whether to reset the iteration counter")

flags.DEFINE_string('data_dir', '', 'root filepath of data.')
flags.DEFINE_string('train_file',
                    './filenames/kitti_train_files_png_4frames.txt',
                    'training file')
flags.DEFINE_string('gt_2012_dir', '',
                    'directory of ground truth of kitti 2012')
flags.DEFINE_string('gt_2015_dir', '',
                    'directory of ground truth of kitti 2015')

flags.DEFINE_integer('batch_size', 4, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.0001,
                   'the base learning rate of the generator')
flags.DEFINE_integer('num_gpus', 1, 'the number of gpu to use')

flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 832, "Image width")

flags.DEFINE_float("depth_smooth_weight", 10.0, "Weight for depth smoothness")
flags.DEFINE_float("ssim_weight", 0.85,
                   "Weight for using ssim loss in pixel loss")
flags.DEFINE_float("flow_smooth_weight", 10.0, "Weight for flow smoothness")
flags.DEFINE_float("flow_consist_weight", 0.01, "Weight for flow consistent")
flags.DEFINE_float("flow_diff_threshold", 4.0,
                   "threshold when comparing optical flow and rigid flow ")

flags.DEFINE_string('eval_pose', '', 'pose seq to evaluate')

FLAGS.num_scales = 4
opt = FLAGS


def main(unused_argv):
    if FLAGS.trace == "":
        raise Exception("OUT_DIR must be specified")

    print 'Constructing models and inputs.'

    if FLAGS.mode == "depthflow":  # stage 3: train depth and flow together
        Model = Model_depthflow
        Model_eval = Model_eval_depthflow

        opt.eval_flow = True
        opt.eval_depth = True
        opt.eval_mask = True
    elif FLAGS.mode == "depth":  # stage 2: train depth
        Model = Model_depth
        Model_eval = Model_eval_depth

        opt.eval_flow = True
        opt.eval_depth = True
        opt.eval_mask = False
    elif FLAGS.mode == "flow":  # stage 1: train flow
        Model = Model_flow
        Model_eval = Model_eval_flow

        opt.eval_flow = True
        opt.eval_depth = False
        opt.eval_mask = False
    elif FLAGS.mode == "stereo":
        Model = Model_stereo
        Model_eval = Model_eval_stereo

        opt.eval_flow = False
        opt.eval_depth = True
        opt.eval_mask = False
    else:
        raise "mode must be one of flow, depth, depthflow or stereo"

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate)

        tower_grads = []

        image1, image_r, image2, image2_r, proj_cam2pix, proj_pix2cam = MonodepthDataloader(
            FLAGS).data_batch

        split_image1 = tf.split(
            axis=0, num_or_size_splits=FLAGS.num_gpus, value=image1)
        split_image2 = tf.split(
            axis=0, num_or_size_splits=FLAGS.num_gpus, value=image2)
        split_cam2pix = tf.split(
            axis=0, num_or_size_splits=FLAGS.num_gpus, value=proj_cam2pix)
        split_pix2cam = tf.split(
            axis=0, num_or_size_splits=FLAGS.num_gpus, value=proj_pix2cam)
        split_image_r = tf.split(
            axis=0, num_or_size_splits=FLAGS.num_gpus, value=image_r)
        split_image_r_next = tf.split(
            axis=0, num_or_size_splits=FLAGS.num_gpus, value=image2_r)

        summaries_cpu = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                          tf.get_variable_scope().name)

        with tf.variable_scope(tf.get_variable_scope()) as vs:
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    if i == FLAGS.num_gpus - 1:
                        scopename = "model"
                    else:
                        scopename = '%s_%d' % ("tower", i)
                    with tf.name_scope(scopename) as ns:
                        if i == 0:
                            model = Model(
                                split_image1[i],
                                split_image2[i],
                                split_image_r[i],
                                split_image_r_next[i],
                                split_cam2pix[i],
                                split_pix2cam[i],
                                reuse_scope=False,
                                scope=vs)
                            var_pose = list(
                                set(
                                    tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=".*pose_net.*")))
                            var_depth = list(
                                set(
                                    tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=".*(depth_net|feature_net_disp).*"
                                    )))
                            var_flow = list(
                                set(
                                    tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=".*(flow_net|feature_net_flow).*"
                                    )))

                            if FLAGS.mode == "depthflow":
                                var_train_list = var_pose + var_depth + var_flow
                            elif FLAGS.mode == "depth":
                                var_train_list = var_pose + var_depth
                            elif FLAGS.mode == "flow":
                                var_train_list = var_flow
                            else:
                                var_train_list = var_depth

                        else:
                            model = Model(
                                split_image1[i],
                                split_image2[i],
                                split_image_r[i],
                                split_image_r_next[i],
                                split_cam2pix[i],
                                split_pix2cam[i],
                                reuse_scope=True,
                                scope=vs)

                        loss = model.loss
                        # Retain the summaries from the final tower.
                        if i == FLAGS.num_gpus - 1:
                            summaries = tf.get_collection(
                                tf.GraphKeys.SUMMARIES, ns)
                            eval_model = Model_eval(scope=vs)
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = train_op.compute_gradients(
                            loss, var_list=var_train_list)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = train_op.apply_gradients(
                grads, global_step=global_step)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=10)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries + summaries_cpu)

        # Make training session.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        summary_writer = tf.summary.FileWriter(
            FLAGS.trace, graph=sess.graph, flush_secs=10)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.pretrained_model:
            if FLAGS.train_test == "test" or (not FLAGS.retrain):
                saver.restore(sess, FLAGS.pretrained_model)
            elif FLAGS.mode == "depthflow":
                saver_rest = tf.train.Saver(
                    list(
                        set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) -
                        set(
                            tf.get_collection(
                                tf.GraphKeys.GLOBAL_VARIABLES,
                                scope=".*(Adam_1|Adam).*"))),
                    max_to_keep=1)
                saver_rest.restore(sess, FLAGS.pretrained_model)
            elif FLAGS.mode == "depth":
                saver_flow = tf.train.Saver(
                    tf.get_collection(
                        tf.GraphKeys.MODEL_VARIABLES,
                        scope=".*(flow_net|feature_net_flow).*"),
                    max_to_keep=1)
                saver_flow.restore(sess, FLAGS.pretrained_model)
            else:
                raise Exception(
                    "pretrained_model not used. Please set train_test=test or retrain=False"
                )
            if FLAGS.retrain:
                sess.run(global_step.assign(0))

        start_itr = global_step.eval(session=sess)
        tf.train.start_queue_runners(sess)

        if opt.eval_flow:
            gt_flows_2012, noc_masks_2012 = load_gt_flow_kitti("kitti_2012")
            gt_flows_2015, noc_masks_2015 = load_gt_flow_kitti("kitti")
            gt_masks = load_gt_mask()
        else:
            gt_flows_2012, noc_masks_2012, gt_flows_2015, noc_masks_2015, gt_masks = \
              None, None, None, None, None

        # Run training.
        for itr in range(start_itr, FLAGS.num_iterations):
            if FLAGS.train_test == "train":
                _, summary_str, summary_scalar_str = sess.run(
                    [apply_gradient_op, summary_op, model.summ_op])

                if (itr) % (SUMMARY_INTERVAL) == 2:
                    summary_writer.add_summary(summary_scalar_str, itr)

                if (itr) % (SUMMARY_INTERVAL * 10) == 2:
                    summary_writer.add_summary(summary_str, itr)

                if (itr) % (SAVE_INTERVAL) == 2:
                    saver.save(
                        sess, FLAGS.trace + '/model', global_step=global_step)

            if (itr) % (VAL_INTERVAL) == 2 or FLAGS.train_test == "test":
                test(sess, eval_model, itr, gt_flows_2012, noc_masks_2012,
                     gt_flows_2015, noc_masks_2015, gt_masks)


if __name__ == '__main__':
    app.run()
