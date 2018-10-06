from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
"""
Adopted from https://github.com/tinghuiz/SfMLearner
"""


def pose_exp_net(tgt_image, src_image):
    inputs = tf.concat([tgt_image, src_image], axis=3)
    with tf.variable_scope('pose_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
                normalizer_fn=None,
                weights_regularizer=slim.l2_regularizer(0.0004),
                activation_fn=tf.nn.relu,
                outputs_collections=end_points_collection):
            cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1, 16, [7, 7], stride=1, scope='cnv1b')
            cnv2 = slim.conv2d(cnv1b, 32, [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2, 32, [5, 5], stride=1, scope='cnv2b')
            cnv3 = slim.conv2d(cnv2b, 64, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3, 64, [3, 3], stride=1, scope='cnv3b')
            cnv4 = slim.conv2d(cnv3b, 128, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4, 128, [3, 3], stride=1, scope='cnv4b')
            cnv5 = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5, 256, [3, 3], stride=1, scope='cnv5b')

            # Pose specific layers
            cnv6 = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 256, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 256, [3, 3], stride=1, scope='cnv7b')
            pose_pred = slim.conv2d(
                cnv7b,
                6, [1, 1],
                scope='pred',
                stride=1,
                normalizer_fn=None,
                activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            # Empirically we found that scaling by a small constant on the 
            # translations facilitates training.
            pose_final = tf.reshape(pose_avg, [-1, 6])
            pose_final = tf.concat(
                [pose_final[:, 0:3], 0.01 * pose_final[:, 3:6]], axis=1)
    return pose_final
