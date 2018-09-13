# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.platform import app
import numpy as np
import pdb


def transformerFwd(U,
                   flo,
                   out_size,
                   name='SpatialTransformerFwd',
                   backprop=False,
                   **kwargs):
    """Forward Warping Layer described in 
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
        The optical flow used for forward warping 
        having the shape of [num_batch, height, width, 2].
    backprop: boolean
        Indicates whether to back-propagate through forward warping layer
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(
                    tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f - 1) / 2.0
            y = (y + 1.0) * (height_f - 1) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0_c = tf.clip_by_value(x0, zero, max_x)
            x1_c = tf.clip_by_value(x1, zero, max_x)
            y0_c = tf.clip_by_value(y0, zero, max_y)
            y1_c = tf.clip_by_value(y1, zero, max_y)

            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)

            base_y0 = base + y0_c * dim2
            base_y1 = base + y1_c * dim2
            idx_a = base_y0 + x0_c
            idx_b = base_y1 + x0_c
            idx_c = base_y0 + x1_c
            idx_d = base_y1 + x1_c

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

            zerof = tf.zeros_like(wa)

            wa = tf.where(
                tf.logical_and(tf.equal(x0_c, x0), tf.equal(y0_c, y0)), wa,
                zerof)
            wb = tf.where(
                tf.logical_and(tf.equal(x0_c, x0), tf.equal(y1_c, y1)), wb,
                zerof)
            wc = tf.where(
                tf.logical_and(tf.equal(x1_c, x1), tf.equal(y0_c, y0)), wc,
                zerof)
            wd = tf.where(
                tf.logical_and(tf.equal(x1_c, x1), tf.equal(y1_c, y1)), wd,
                zerof)

            if not backprop:
                zeros = tf.zeros(
                    shape=[
                        int(im.get_shape()[0]) * int(im.get_shape()[1]) *
                        int(im.get_shape()[2]), int(im.get_shape()[3])
                    ],
                    dtype='float32')
                output = tf.Variable(
                    zeros,
                    trainable=False,
                    collections=[tf.GraphKeys.LOCAL_VARIABLES])
                init = tf.assign(output, zeros)

                # tf.scatter_add will not back-propagate gradients
                with tf.control_dependencies([init]):
                    output = tf.scatter_add(output, idx_a, im_flat * wa)
                    output = tf.scatter_add(output, idx_b, im_flat * wb)
                    output = tf.scatter_add(output, idx_c, im_flat * wc)
                    output = tf.scatter_add(output, idx_d, im_flat * wd)
            else:
                shape = [
                    int(im.get_shape()[0]) * int(im.get_shape()[1]) *
                    int(im.get_shape()[2]), int(im.get_shape()[3])
                ]
                output = tf.scatter_nd(tf.expand_dims(idx_a, -1), im_flat*wa, shape) + \
                         tf.scatter_nd(tf.expand_dims(idx_b, -1), im_flat*wb, shape) + \
                         tf.scatter_nd(tf.expand_dims(idx_c, -1), im_flat*wc, shape) + \
                         tf.scatter_nd(tf.expand_dims(idx_d, -1), im_flat*wd, shape)

            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(
                tf.ones(shape=tf.stack([height, 1])),
                tf.transpose(
                    tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(
                tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                tf.ones(shape=tf.stack([1, width])))

            return x_t, y_t

    def _transform(flo, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            x_s, y_s = _meshgrid(out_height, out_width)
            x_s = tf.expand_dims(x_s, 0)
            x_s = tf.tile(x_s, [num_batch, 1, 1])

            y_s = tf.expand_dims(y_s, 0)
            y_s = tf.tile(y_s, [num_batch, 1, 1])

            x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
            y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

            x_t_flat = tf.reshape(x_t, [-1])
            y_t_flat = tf.reshape(y_t, [-1])

            input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat,
                                             out_size)

            output = tf.reshape(
                input_transformed,
                tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(flo, U, out_size)
        return output


def main(unused_argv):
    # Some test cases
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    #   
    image = tf.constant(range(16), shape=[1, 4, 4, 1], dtype="float32")

    flo = np.zeros((1, 4, 4, 2))
    flo[0, 1, 1, 0] = 1.0
    flo = tf.constant(flo, dtype="float32")

    image2 = transformerFwd(image, flo, [4, 4])

    image2 = sess.run(image2)
    loss = tf.reduce_mean(tf.abs(image2 - 1.0))

    var_grad = tf.gradients(loss, [flo])[0]

    sess.run(tf.global_variables_initializer())
    print(image2.eval(session=sess))
    pdb.set_trace()


if __name__ == '__main__':
    app.run()
