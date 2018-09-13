# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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

import tensorflow.contrib.slim as slim
from optical_flow_warp_old import transformer_old


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_bilinear(inputs, [rH.value, rW.value])


def leaky_relu(_x, alpha=0.1):
    pos = tf.nn.relu(_x)
    neg = alpha * (_x - abs(_x)) * 0.5

    return pos + neg


def feature_pyramid_flow(image, reuse):
    with tf.variable_scope('feature_net_flow'):
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
                weights_regularizer=slim.l2_regularizer(0.0004),
                activation_fn=leaky_relu,
                variables_collections=["flownet"],
                reuse=reuse):
            cnv1 = slim.conv2d(image, 16, [3, 3], stride=2, scope="cnv1")
            cnv2 = slim.conv2d(cnv1, 16, [3, 3], stride=1, scope="cnv2")
            cnv3 = slim.conv2d(cnv2, 32, [3, 3], stride=2, scope="cnv3")
            cnv4 = slim.conv2d(cnv3, 32, [3, 3], stride=1, scope="cnv4")
            cnv5 = slim.conv2d(cnv4, 64, [3, 3], stride=2, scope="cnv5")
            cnv6 = slim.conv2d(cnv5, 64, [3, 3], stride=1, scope="cnv6")
            cnv7 = slim.conv2d(cnv6, 96, [3, 3], stride=2, scope="cnv7")
            cnv8 = slim.conv2d(cnv7, 96, [3, 3], stride=1, scope="cnv8")
            cnv9 = slim.conv2d(cnv8, 128, [3, 3], stride=2, scope="cnv9")
            cnv10 = slim.conv2d(cnv9, 128, [3, 3], stride=1, scope="cnv10")
            cnv11 = slim.conv2d(cnv10, 192, [3, 3], stride=2, scope="cnv11")
            cnv12 = slim.conv2d(cnv11, 192, [3, 3], stride=1, scope="cnv12")

            return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12


def cost_volumn(feature1, feature2, d=4):
    batch_size, H, W, feature_num = map(int, feature1.get_shape()[0:4])
    feature2 = tf.pad(feature2, [[0, 0], [d, d], [d, d], [0, 0]], "CONSTANT")
    cv = []
    for i in range(2 * d + 1):
        for j in range(2 * d + 1):
            cv.append(
                tf.reduce_mean(
                    feature1 * feature2[:, i:(i + H), j:(j + W), :],
                    axis=3,
                    keep_dims=True))
    return tf.concat(cv, axis=3)


def optical_flow_decoder_dc(inputs, level):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(0.0004),
            activation_fn=leaky_relu):
        cnv1 = slim.conv2d(
            inputs, 128, [3, 3], stride=1, scope="cnv1_fd_" + str(level))
        cnv2 = slim.conv2d(
            cnv1, 128, [3, 3], stride=1, scope="cnv2_fd_" + str(level))
        cnv3 = slim.conv2d(
            tf.concat(
                [cnv1, cnv2], axis=3),
            96, [3, 3],
            stride=1,
            scope="cnv3_fd_" + str(level))
        cnv4 = slim.conv2d(
            tf.concat(
                [cnv2, cnv3], axis=3),
            64, [3, 3],
            stride=1,
            scope="cnv4_fd_" + str(level))
        cnv5 = slim.conv2d(
            tf.concat(
                [cnv3, cnv4], axis=3),
            32, [3, 3],
            stride=1,
            scope="cnv5_fd_" + str(level))
        flow = slim.conv2d(
            tf.concat(
                [cnv4, cnv5], axis=3),
            2, [3, 3],
            stride=1,
            scope="cnv6_fd_" + str(level),
            activation_fn=None)

        return flow, cnv5


def context_net(inputs):
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(0.0004),
            activation_fn=leaky_relu):
        cnv1 = slim.conv2d(inputs, 128, [3, 3], rate=1, scope="cnv1_cn")
        cnv2 = slim.conv2d(cnv1, 128, [3, 3], rate=2, scope="cnv2_cn")
        cnv3 = slim.conv2d(cnv2, 128, [3, 3], rate=4, scope="cnv3_cn")
        cnv4 = slim.conv2d(cnv3, 96, [3, 3], rate=8, scope="cnv4_cn")
        cnv5 = slim.conv2d(cnv4, 64, [3, 3], rate=16, scope="cnv5_cn")
        cnv6 = slim.conv2d(cnv5, 32, [3, 3], rate=1, scope="cnv6_cn")

        flow = slim.conv2d(
            cnv6, 2, [3, 3], rate=1, scope="cnv7_cn", activation_fn=None)
        return flow


def construct_model_pwc_full(image1, image2, feature1, feature2):
    with tf.variable_scope('flow_net'):
        batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])

        #############################
        feature1_1, feature1_2, feature1_3, feature1_4, feature1_5, feature1_6 = feature1
        feature2_1, feature2_2, feature2_3, feature2_4, feature2_5, feature2_6 = feature2

        cv6 = cost_volumn(feature1_6, feature2_6, d=4)
        flow6, _ = optical_flow_decoder_dc(cv6, level=6)

        flow6to5 = tf.image.resize_bilinear(flow6,
                                            [H / (2**5), (W / (2**5))]) * 2.0
        feature2_5w = transformer_old(feature2_5, flow6to5, [H / 32, W / 32])
        cv5 = cost_volumn(feature1_5, feature2_5w, d=4)
        flow5, _ = optical_flow_decoder_dc(
            tf.concat(
                [cv5, feature1_5, flow6to5], axis=3), level=5)
        flow5 = flow5 + flow6to5

        flow5to4 = tf.image.resize_bilinear(flow5,
                                            [H / (2**4), (W / (2**4))]) * 2.0
        feature2_4w = transformer_old(feature2_4, flow5to4, [H / 16, W / 16])
        cv4 = cost_volumn(feature1_4, feature2_4w, d=4)
        flow4, _ = optical_flow_decoder_dc(
            tf.concat(
                [cv4, feature1_4, flow5to4], axis=3), level=4)
        flow4 = flow4 + flow5to4

        flow4to3 = tf.image.resize_bilinear(flow4,
                                            [H / (2**3), (W / (2**3))]) * 2.0
        feature2_3w = transformer_old(feature2_3, flow4to3, [H / 8, W / 8])
        cv3 = cost_volumn(feature1_3, feature2_3w, d=4)
        flow3, _ = optical_flow_decoder_dc(
            tf.concat(
                [cv3, feature1_3, flow4to3], axis=3), level=3)
        flow3 = flow3 + flow4to3

        flow3to2 = tf.image.resize_bilinear(flow3,
                                            [H / (2**2), (W / (2**2))]) * 2.0
        feature2_2w = transformer_old(feature2_2, flow3to2, [H / 4, W / 4])
        cv2 = cost_volumn(feature1_2, feature2_2w, d=4)
        flow2_raw, f2 = optical_flow_decoder_dc(
            tf.concat(
                [cv2, feature1_2, flow3to2], axis=3), level=2)
        flow2_raw = flow2_raw + flow3to2

        flow2 = context_net(tf.concat([flow2_raw, f2], axis=3)) + flow2_raw

        flow0_enlarge = tf.image.resize_bilinear(flow2 * 4.0, [H, W])
        flow1_enlarge = tf.image.resize_bilinear(flow3 * 4.0, [H // 2, W // 2])
        flow2_enlarge = tf.image.resize_bilinear(flow4 * 4.0, [H // 4, W // 4])
        flow3_enlarge = tf.image.resize_bilinear(flow5 * 4.0, [H // 8, W // 8])

        return flow0_enlarge, flow1_enlarge, flow2_enlarge, flow3_enlarge
