import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import cv2


# Some were adopted from 
# https://github.com/tensorflow/models/tree/master/research/video_prediction
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


def weighted_mean_squared_error(true, pred, weight):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """

    tmp = tf.reduce_sum(
        weight * tf.square(true - pred), axis=[1, 2],
        keep_dims=True) / tf.reduce_sum(
            weight, axis=[1, 2], keep_dims=True)
    return tf.reduce_mean(tmp)


def mean_L1_error(true, pred):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
    return tf.reduce_sum(tf.abs(true - pred)) / tf.to_float(tf.size(pred))


def weighted_mean_L1_error(true, pred, weight):
    """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
    return tf.reduce_sum(tf.abs(true - pred) *
                         weight) / tf.to_float(tf.size(pred))


def cal_grad2_error(flo, image, beta):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo 
    """

    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = tf.exp(-10.0 * tf.reduce_mean(
        tf.abs(img_grad_x), 3, keep_dims=True))
    weights_y = tf.exp(-10.0 * tf.reduce_mean(
        tf.abs(img_grad_y), 3, keep_dims=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (tf.reduce_mean(beta*weights_x[:,:, 1:, :]*tf.abs(dx2)) + \
           tf.reduce_mean(beta*weights_y[:, 1:, :, :]*tf.abs(dy2))) / 2.0


def cal_grad2_error_mask(flo, image, beta, mask):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo
    within the given mask 
    """

    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy

    img_grad_x, img_grad_y = gradient(image)
    weights_x = tf.exp(-10.0 * tf.reduce_mean(
        tf.abs(img_grad_x), 3, keep_dims=True))
    weights_y = tf.exp(-10.0 * tf.reduce_mean(
        tf.abs(img_grad_y), 3, keep_dims=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (tf.reduce_mean(beta*weights_x[:,:, 1:, :]*tf.abs(dx2) * mask[:, :, 1:-1, :]) + \
           tf.reduce_mean(beta*weights_y[:, 1:, :, :]*tf.abs(dy2) * mask[:, 1:-1, :, :])) / 2.0


def SSIM(x, y):
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
    sigma_y = slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def deprocess_image(image):
    # Assuming input image is float32
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


def preprocess_image(image):
    # Assuming input image is uint8
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def charbonnier_loss(x,
                     mask=None,
                     truncate=None,
                     alpha=0.45,
                     beta=1.0,
                     epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    batch, height, width, channels = tf.unstack(tf.shape(x))
    normalization = tf.cast(batch * height * width * channels, tf.float32)

    error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

    if mask is not None:
        error = tf.multiply(mask, error)

    if truncate is not None:
        error = tf.minimum(error, truncate)

    return tf.reduce_sum(error) / normalization
