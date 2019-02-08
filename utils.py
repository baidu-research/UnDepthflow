from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from optical_flow_warp_old import transformer_old
import pdb
from tensorflow.python.platform import app
"""
Adopted from https://github.com/tinghuiz/SfMLearner
"""


def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_depth_for_display(depth,
                                pc=95,
                                crop_percent=0,
                                normalizer=None,
                                cmap='gray'):
    # convert to disparity
    depth = 1. / (depth + 1e-6)
    if normalizer is not None:
        depth = depth / normalizer
    else:
        depth = depth / (np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1 - crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


def inverse_warp(depth,
                 pose,
                 intrinsics,
                 intrinsics_inv,
                 pose_mat_inverse=False):
    """Inverse warp a source image to the target image plane
       Part of the code modified from  
       https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
    Args:
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        optical flow induced by the given depth and pose, 
        pose matrix
    """

    def _pixel2cam(depth, pixel_coords, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame"""
        cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth
        return cam_coords

    def _cam2pixel(cam_coords, proj_c2p):
        """Transform coordinates in the camera frame to the pixel frame"""
        pcoords = tf.matmul(proj_c2p, cam_coords)
        X = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
        Y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
        Z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
        # Not tested if adding a small number is necessary
        X_norm = X / (Z + 1e-10)
        Y_norm = Y / (Z + 1e-10)
        pixel_coords = tf.concat([X_norm, Y_norm], axis=1)
        return pixel_coords

    def _meshgrid_abs(height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _meshgrid_abs_xy(batch, height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        return tf.tile(tf.expand_dims(x_t, 0), [batch, 1, 1]), tf.tile(
            tf.expand_dims(y_t, 0), [batch, 1, 1])

    def _euler2mat(z, y, x):
        """Converts euler angles to rotation matrix
         TODO: remove the dimension for 'N' (deprecated for converting all source
               poses altogether)
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            z: rotation angle along z axis (in radians) -- size = [B, N]
            y: rotation angle along y axis (in radians) -- size = [B, N]
            x: rotation angle along x axis (in radians) -- size = [B, N]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
        """
        B = tf.shape(z)[0]
        N = 1
        z = tf.clip_by_value(z, -np.pi, np.pi)
        y = tf.clip_by_value(y, -np.pi, np.pi)
        x = tf.clip_by_value(x, -np.pi, np.pi)

        # Expand to B x N x 1 x 1
        z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        x = tf.expand_dims(tf.expand_dims(x, -1), -1)

        zeros = tf.zeros([B, N, 1, 1])
        ones = tf.ones([B, N, 1, 1])

        cosz = tf.cos(z)
        sinz = tf.sin(z)
        rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
        rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
        rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
        zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

        cosy = tf.cos(y)
        siny = tf.sin(y)
        roty_1 = tf.concat([cosy, zeros, siny], axis=3)
        roty_2 = tf.concat([zeros, ones, zeros], axis=3)
        roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
        ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

        cosx = tf.cos(x)
        sinx = tf.sin(x)
        rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
        rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
        rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
        xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

        rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
        return rotMat

    def _pose_vec2mat(vec):
        """Converts 6DoF parameters to transformation matrix
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 4, 4]
        """
        translation = tf.slice(vec, [0, 0], [-1, 3])
        translation = tf.expand_dims(translation, -1)
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = _euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([rot_mat, translation], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        return transform_mat

    dims = tf.shape(depth)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsics_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)
    if len(pose.get_shape().as_list()) == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    if pose_mat_inverse:
        pose_mat = tf.matrix_inverse(pose_mat)

    # Get projection matrix for tgt camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, hom_filler], axis=1)
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat)
    src_pixel_coords = _cam2pixel(cam_coords_hom, proj_cam_to_src_pixel)
    src_pixel_coords = tf.reshape(src_pixel_coords,
                                  [batch_size, 2, img_height, img_width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1])

    tgt_pixel_coords_x, tgt_pixel_coords_y = _meshgrid_abs_xy(
        batch_size, img_height, img_width)
    flow_x = src_pixel_coords[:, :, :, 0] - tgt_pixel_coords_x
    flow_y = src_pixel_coords[:, :, :, 1] - tgt_pixel_coords_y
    flow = tf.concat(
        [tf.expand_dims(flow_x, -1), tf.expand_dims(flow_y, -1)], axis=-1)
    return flow, pose_mat


def calculate_pose_basis(cam_coords1, cam_coords2, weights, batch_size):
    '''
    Given two point clouds and weights, find the transformation that 
    minimizes the distance between the two clouds
    Args:
        cam_coords1: point cloud 1 -- [B, 3, -1]
        cam_coords2: point cloud 2 -- [B, 3, -1]
        weights: weights to specify which points in the point cloud are 
                 used for alignment -- [B, 1, -1]
    return:
        transformation matrix -- [B, 4, 4]
    '''
    centroids1 = tf.reduce_mean(
        cam_coords1 * weights, axis=2, keep_dims=True) / tf.reduce_mean(
            weights, axis=2, keep_dims=True)
    centroids2 = tf.reduce_mean(
        cam_coords2 * weights, axis=2, keep_dims=True) / tf.reduce_mean(
            weights, axis=2, keep_dims=True)

    cam_coords1_shifted = tf.expand_dims(
        tf.transpose(cam_coords1 - centroids1, [0, 2, 1]), -1)
    cam_coords2_shifted = tf.expand_dims(
        tf.transpose(cam_coords2 - centroids2, [0, 2, 1]), -2)

    weights_trans = tf.expand_dims(tf.transpose(weights, [0, 2, 1]), -1)
    H = tf.reduce_sum(
        tf.matmul(cam_coords1_shifted, cam_coords2_shifted) * weights_trans,
        axis=1,
        keep_dims=False)
    S, U, V = tf.svd(H)
    R = tf.matmul(V, U, transpose_a=False, transpose_b=True)

    T = -tf.matmul(R, centroids1) + centroids2

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    rigid_pose_mat = tf.concat([tf.concat([R, T], axis=2), filler], axis=1)

    return rigid_pose_mat


def inverse_warp_new(depth1,
                     depth2,
                     pose,
                     intrinsics,
                     intrinsics_inv,
                     flow_input,
                     occu_mask,
                     pose_mat_inverse=False):
    """
    Inverse warp a source image to the target image plane after refining the 
    pose by rigid alignment described in 
    'Joint Unsupervised Learning of Optical Flow and Depth by Watching 
    Stereo Videos by Yang Wang et al.'
    Args:
        depth1: depth map of the target image -- [B, H, W]
        depth2: depth map of the source image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
        flow_input: flow between target and source image -- [B, H, W, 2]
        occu_mask: occlusion mask of target image -- [B, H, W, 1]
    Returns:
        [optical flow induced by refined pose, 
         refined pose matrix,
         disparity of the target frame transformed by refined pose,
         the mask for areas used for rigid alignment]
    """

    def _pixel2cam(depth, pixel_coords, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame"""
        cam_coords = tf.matmul(intrinsics_inv, pixel_coords) * depth
        return cam_coords

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(
                    tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _cam2pixel(cam_coords, proj_c2p):
        """Transform coordinates in the camera frame to the pixel frame"""
        pcoords = tf.matmul(proj_c2p, cam_coords)
        X = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
        Y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
        Z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
        # Not tested if adding a small number is necessary
        X_norm = X / (Z + 1e-10)
        Y_norm = Y / (Z + 1e-10)
        pixel_coords = tf.concat([X_norm, Y_norm], axis=1)
        return pixel_coords

    def _meshgrid_abs(height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    def _meshgrid_abs_xy(batch, height, width):
        """Meshgrid in the absolute coordinates"""
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(
                tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * tf.cast(width, tf.float32)
        y_t = (y_t + 1.0) * 0.5 * tf.cast(height, tf.float32)
        return tf.tile(tf.expand_dims(x_t, 0), [batch, 1, 1]), tf.tile(
            tf.expand_dims(y_t, 0), [batch, 1, 1])

    def _euler2mat(z, y, x):
        """Converts euler angles to rotation matrix
         TODO: remove the dimension for 'N' (deprecated for converting all source
               poses altogether)
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            z: rotation angle along z axis (in radians) -- size = [B, N]
            y: rotation angle along y axis (in radians) -- size = [B, N]
            x: rotation angle along x axis (in radians) -- size = [B, N]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
        """
        B = tf.shape(z)[0]
        N = 1
        z = tf.clip_by_value(z, -np.pi, np.pi)
        y = tf.clip_by_value(y, -np.pi, np.pi)
        x = tf.clip_by_value(x, -np.pi, np.pi)

        # Expand to B x N x 1 x 1
        z = tf.expand_dims(tf.expand_dims(z, -1), -1)
        y = tf.expand_dims(tf.expand_dims(y, -1), -1)
        x = tf.expand_dims(tf.expand_dims(x, -1), -1)

        zeros = tf.zeros([B, N, 1, 1])
        ones = tf.ones([B, N, 1, 1])

        cosz = tf.cos(z)
        sinz = tf.sin(z)
        rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
        rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
        rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
        zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

        cosy = tf.cos(y)
        siny = tf.sin(y)
        roty_1 = tf.concat([cosy, zeros, siny], axis=3)
        roty_2 = tf.concat([zeros, ones, zeros], axis=3)
        roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
        ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

        cosx = tf.cos(x)
        sinx = tf.sin(x)
        rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
        rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
        rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
        xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

        rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
        return rotMat

    def _pose_vec2mat(vec):
        """Converts 6DoF parameters to transformation matrix
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 4, 4]
        """
        translation = tf.slice(vec, [0, 0], [-1, 3])
        translation = tf.expand_dims(translation, -1)
        rx = tf.slice(vec, [0, 3], [-1, 1])
        ry = tf.slice(vec, [0, 4], [-1, 1])
        rz = tf.slice(vec, [0, 5], [-1, 1])
        rot_mat = _euler2mat(rz, ry, rx)
        rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])
        transform_mat = tf.concat([rot_mat, translation], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        return transform_mat

    dims = tf.shape(depth1)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth1 = tf.reshape(depth1, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    # Point Cloud Q_1
    cam_coords1 = _pixel2cam(depth1, grid, intrinsics_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords1_hom = tf.concat([cam_coords1, ones], axis=1)
    if len(pose.get_shape().as_list()) == 3:
        pose_mat = pose
    else:
        pose_mat = _pose_vec2mat(pose)

    if pose_mat_inverse:
        pose_mat = tf.matrix_inverse(pose_mat)
    # Point Cloud \hat{Q_1}
    cam_coords1_trans = tf.matmul(pose_mat, cam_coords1_hom)[:, 0:3, :]

    depth2 = tf.reshape(depth2, [batch_size, 1, img_height * img_width])
    # Point Cloud Q_2
    cam_coords2 = _pixel2cam(depth2, grid, intrinsics_inv)
    cam_coords2 = tf.reshape(cam_coords2,
                             [batch_size, 3, img_height, img_width])
    cam_coords2 = tf.transpose(cam_coords2, [0, 2, 3, 1])
    cam_coords2_trans = transformer_old(cam_coords2, flow_input,
                                        [img_height, img_width])
    # Point Cloud \tilda{Q_1}
    cam_coords2_trans = tf.reshape(
        tf.transpose(cam_coords2_trans, [0, 3, 1, 2]), [batch_size, 3, -1])

    occu_mask = tf.reshape(occu_mask, [batch_size, 1, -1])
    # To eliminate occluded area from the small_mask
    occu_mask = tf.where(occu_mask < 0.75,
                         tf.ones_like(occu_mask) * 10000.0,
                         tf.ones_like(occu_mask))

    diff2 = tf.sqrt(
        tf.reduce_sum(
            tf.square(cam_coords1_trans - cam_coords2_trans),
            axis=1,
            keep_dims=True)) * occu_mask
    small_mask = tf.where(
        diff2 < tf.contrib.distributions.percentile(
            diff2, 25.0, axis=2, keep_dims=True),
        tf.ones_like(diff2),
        tf.zeros_like(diff2))

    # Delta T
    rigid_pose_mat = calculate_pose_basis(cam_coords1_trans, cam_coords2_trans,
                                          small_mask, batch_size)
    # T' = deltaT x T
    pose_mat2 = tf.matmul(rigid_pose_mat, pose_mat)

    # Get projection matrix for tgt camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, hom_filler], axis=1)
    proj_cam_to_src_pixel = tf.matmul(intrinsics, pose_mat2)
    src_pixel_coords = _cam2pixel(cam_coords1_hom, proj_cam_to_src_pixel)
    src_pixel_coords = tf.reshape(src_pixel_coords,
                                  [batch_size, 2, img_height, img_width])
    src_pixel_coords = tf.transpose(src_pixel_coords, perm=[0, 2, 3, 1])

    tgt_pixel_coords_x, tgt_pixel_coords_y = _meshgrid_abs_xy(
        batch_size, img_height, img_width)
    flow_x = src_pixel_coords[:, :, :, 0] - tgt_pixel_coords_x
    flow_y = src_pixel_coords[:, :, :, 1] - tgt_pixel_coords_y
    flow = tf.concat(
        [tf.expand_dims(flow_x, -1), tf.expand_dims(flow_y, -1)], axis=-1)

    cam_coords1_trans_z = tf.matmul(pose_mat2, cam_coords1_hom)[:, 2:3, :]
    cam_coords1_trans_z = tf.reshape(cam_coords1_trans_z,
                                     [batch_size, img_height, img_width, 1])
    disp1_trans = 1.0 / cam_coords1_trans_z

    return flow, pose_mat2, disp1_trans, tf.reshape(
        small_mask,
        [batch_size, img_height, img_width, 1]), cam_coords1_trans, tf.matmul(
            pose_mat2, cam_coords1_hom)[:, 0:3, :], cam_coords2


def main(unused_argv):
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))

    # Random rotation and translation
    R = np.mat(np.random.rand(3, 3))
    t = np.mat(np.random.rand(3, 1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U * Vt

    # remove reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U * Vt

    # number of points
    n = 1000

    A = np.mat(np.random.rand(n, 3))
    B = R * A.T + np.tile(t, (1, n))
    B = B.T

    weights = np.mat(np.random.randint(2, size=(n, 1)), dtype=np.float64)

    tfA = tf.expand_dims(tf.transpose(tf.convert_to_tensor(A)), 0)
    tfB = tf.expand_dims(tf.transpose(tf.convert_to_tensor(B)), 0)
    tfWeights = tf.expand_dims(tf.transpose(tf.convert_to_tensor(weights)), 0)

    tfR, tfT = calculate_pose_basis(tfA, tfB, tfWeights)
    pdb.set_trace()


if __name__ == '__main__':
    app.run()
