import tensorflow as tf
from tensorflow.python.platform import flags

from nets.pose_net import pose_exp_net
from monodepth_model import disp_godard
from nets.pwc_flow import construct_model_pwc_full, feature_pyramid_flow
from nets.pwc_disp import feature_pyramid_disp
from optical_flow_warp_fwd import transformerFwd
from optical_flow_warp_old import transformer_old
from monodepth_dataloader import get_multi_scale_intrinsics
from utils import inverse_warp, inverse_warp_new
from loss_utils import SSIM, deprocess_image, preprocess_image,\
  cal_grad2_error_mask, charbonnier_loss, cal_grad2_error

opt = flags.FLAGS


class Model_stereo(object):
    def __init__(self,
                 image1=None,
                 image2=None,
                 image1r=None,
                 image2r=None,
                 cam2pix=None,
                 pix2cam=None,
                 reuse_scope=False,
                 scope=None):
        summaries = []

        with tf.variable_scope(scope, reuse=reuse_scope):
            feature1_disp = feature_pyramid_disp(image1, reuse=False)
            feature1r_disp = feature_pyramid_disp(image1r, reuse=True)

            pred_disp, stereo_smooth_loss = disp_godard(
                image1,
                image1r,
                feature1_disp,
                feature1r_disp,
                opt,
                is_training=True)

            pred_depth = [1. / d for d in pred_disp]

        self.loss = stereo_smooth_loss

        summaries.append(tf.summary.scalar("total_loss", self.loss))
        summaries.append(
            tf.summary.scalar("stereo_smooth_loss", stereo_smooth_loss))

        tf.summary.image("pred_disp", pred_disp[0][:, :, :, 0:1])
        s = 0
        tf.summary.image('scale%d_depth_image' % s,
                         pred_depth[s][:, :, :, 0:1])
        tf.summary.image('scale%d_right_disparity_image' % s,
                         pred_disp[s][:, :, :, 1:2])
        #     
        self.summ_op = tf.summary.merge(summaries)


class Model_eval_stereo(object):
    def __init__(self, scope=None):
        with tf.variable_scope(scope, reuse=True):
            input_uint8_1 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1')
            input_uint8_1r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1r')
            input_uint8_2 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2')
            input_uint8_2r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2r')
            input_intrinsic = tf.placeholder(tf.float32, [3, 3])

            input_1 = preprocess_image(input_uint8_1)
            input_2 = preprocess_image(input_uint8_2)
            input_1r = preprocess_image(input_uint8_1r)
            input_2r = preprocess_image(input_uint8_2r)

            feature1_disp = feature_pyramid_disp(input_1, reuse=True)
            feature1r_disp = feature_pyramid_disp(input_1r, reuse=True)

            pred_disp = disp_godard(
                input_1,
                input_1r,
                feature1_disp,
                feature1r_disp,
                opt,
                is_training=False)

        self.input_1 = input_uint8_1
        self.input_2 = input_uint8_2
        self.input_r = input_uint8_1r
        self.input_2r = input_uint8_2r
        self.input_intrinsic = input_intrinsic

        self.pred_disp = pred_disp[0][:, :, :, 0:1]

        # Placeholder created for interface consistency
        self.pred_flow_optical = tf.constant(0.0)
        self.pred_flow_rigid = tf.constant(0.0)
        self.pred_disp2 = tf.constant(0.0)
        self.pred_mask = tf.constant(0.0)


class Model_flow(object):
    def __init__(self,
                 image1=None,
                 image2=None,
                 image1r=None,
                 image2r=None,
                 cam2pix=None,
                 pix2cam=None,
                 reuse_scope=False,
                 scope=None):
        summaries = []

        batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])

        with tf.variable_scope(scope, reuse=reuse_scope):
            feature1 = feature_pyramid_flow(image1, reuse=False)
            feature2 = feature_pyramid_flow(image2, reuse=True)

            optical_flows = construct_model_pwc_full(image1, image2, feature1,
                                                     feature2)

        with tf.variable_scope(scope, reuse=True):
            optical_flows_rev = construct_model_pwc_full(image2, image1,
                                                         feature2, feature1)

        occu_masks = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[batch_size, H / (2**s), W / (2**s), 1],
                        dtype='float32'),
                    flowr, [H / (2**s), W / (2**s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            for s, flowr in enumerate(optical_flows_rev)
        ]

        pixel_loss_depth = 0
        pixel_loss_optical = 0
        exp_loss = 0
        flow_smooth_loss = 0
        tgt_image_all = []
        src_image_all = []
        proj_image_depth_all = []
        proj_error_depth_all = []
        exp_mask_stack_all = []
        flyout_map_all = []

        for s in range(opt.num_scales):
            # Scale the source and target images for computing loss at the 
            # according scale.
            curr_tgt_image = tf.image.resize_area(
                image1,
                [int(opt.img_height / (2**s)), int(opt.img_width / (2**s))])
            curr_src_image = tf.image.resize_area(
                image2,
                [int(opt.img_height / (2**s)), int(opt.img_width / (2**s))])

            occu_mask = occu_masks[s]
            occu_mask_avg = tf.reduce_mean(occu_mask)

            curr_proj_image_optical = transformer_old(
                curr_src_image, optical_flows[s], [H / (2**s), W / (2**s)])
            curr_proj_error_optical = tf.abs(curr_proj_image_optical -
                                             curr_tgt_image)
            pixel_loss_optical += (1.0 - opt.ssim_weight) * tf.reduce_mean(
                curr_proj_error_optical * occu_mask) / occu_mask_avg

            curr_flyout_map = occu_mask

            if opt.ssim_weight > 0:
                pixel_loss_optical += opt.ssim_weight * tf.reduce_mean(
                    SSIM(curr_proj_image_optical * occu_mask, curr_tgt_image *
                         occu_mask)) / occu_mask_avg

            flow_smooth_loss += opt.flow_smooth_weight * cal_grad2_error(
                optical_flows[s] / 20.0, curr_tgt_image, 1.0)

            tgt_image_all.append(curr_tgt_image)
            src_image_all.append(curr_src_image)
            proj_image_depth_all.append(curr_proj_image_optical)
            proj_error_depth_all.append(curr_proj_error_optical)

            flyout_map_all.append(curr_flyout_map)

        self.loss = (pixel_loss_optical + flow_smooth_loss)

        summaries.append(tf.summary.scalar("total_loss", self.loss))
        summaries.append(
            tf.summary.scalar("pixel_loss_depth", pixel_loss_depth))
        summaries.append(
            tf.summary.scalar("pixel_loss_optical", pixel_loss_optical))
        summaries.append(tf.summary.scalar("exp_loss", exp_loss))
        tf.summary.image('scale%d_target_image' % s, \
                         deprocess_image(tgt_image_all[s]))
        tf.summary.image('scale%d_src_image' % s, \
                         deprocess_image(src_image_all[s]))

        tf.summary.image('scale_projected_image',
                         deprocess_image(proj_image_depth_all[s]))
        tf.summary.image('scale_proj_error_error', proj_error_depth_all[s])
        tf.summary.image('scale_flyout_mask', flyout_map_all[s])

        self.summ_op = tf.summary.merge(summaries)


class Model_eval_flow(object):
    def __init__(self, scope=None):
        with tf.variable_scope(scope, reuse=True):
            input_uint8_1 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1')
            input_uint8_1r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1r')
            input_uint8_2 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2')
            input_uint8_2r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2r')
            input_intrinsic = tf.placeholder(tf.float32, [3, 3])

            cam2pix, pix2cam = get_multi_scale_intrinsics(input_intrinsic,
                                                          opt.num_scales)
            cam2pix = tf.expand_dims(cam2pix, axis=0)
            pix2cam = tf.expand_dims(pix2cam, axis=0)

            input_1 = preprocess_image(input_uint8_1)
            input_2 = preprocess_image(input_uint8_2)
            input_1r = preprocess_image(input_uint8_1r)
            input_2r = preprocess_image(input_uint8_2r)

            feature1 = feature_pyramid_flow(input_1, reuse=True)
            feature2 = feature_pyramid_flow(input_2, reuse=True)

            optical_flows = construct_model_pwc_full(input_1, input_2,
                                                     feature1, feature2)

        self.input_1 = input_uint8_1
        self.input_2 = input_uint8_2
        self.input_r = input_uint8_1r
        self.input_2r = input_uint8_2r
        self.input_intrinsic = input_intrinsic
        self.pred_flow_optical = optical_flows[0]

        # Placeholder created for interface consistency
        self.pred_flow_rigid = tf.constant(0.0)
        self.pred_disp = tf.constant(0.0)
        self.pred_disp2 = tf.constant(0.0)
        self.pred_mask = tf.constant(0.0)


class Model_depth(object):
    def __init__(self,
                 image1=None,
                 image2=None,
                 image1r=None,
                 image2r=None,
                 cam2pix=None,
                 pix2cam=None,
                 reuse_scope=False,
                 scope=None):
        summaries = []

        batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])

        with tf.variable_scope(scope, reuse=reuse_scope):
            feature1_flow = feature_pyramid_flow(image1, reuse=False)
            feature2_flow = feature_pyramid_flow(image2, reuse=True)

            feature1_disp = feature_pyramid_disp(image1, reuse=False)
            feature1r_disp = feature_pyramid_disp(image1r, reuse=True)

            pred_disp, stereo_smooth_loss = disp_godard(
                image1,
                image1r,
                feature1_disp,
                feature1r_disp,
                opt,
                is_training=True)

            pred_depth = [1. / d for d in pred_disp]
            pred_poses = pose_exp_net(image1, image2)

            optical_flows_rev = construct_model_pwc_full(
                image2, image1, feature2_flow, feature1_flow)

        occu_masks = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[batch_size, H / (2**s), W / (2**s), 1],
                        dtype='float32'),
                    flowr, [H / (2**s), W / (2**s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            for s, flowr in enumerate(optical_flows_rev)
        ]

        pixel_loss_depth = 0
        pixel_loss_optical = 0
        exp_loss = 0
        flow_smooth_loss = 0
        tgt_image_all = []
        src_image_all = []
        proj_image_depth_all = []
        proj_error_depth_all = []
        exp_mask_stack_all = []
        flyout_map_all = []

        for s in range(opt.num_scales):
            # Scale the source and target images for computing loss at the 
            # according scale.
            curr_tgt_image = tf.image.resize_area(
                image1,
                [int(opt.img_height / (2**s)), int(opt.img_width / (2**s))])
            curr_src_image = tf.image.resize_area(
                image2,
                [int(opt.img_height / (2**s)), int(opt.img_width / (2**s))])

            depth_flow, pose_mat = inverse_warp(
                pred_depth[s][:, :, :, 0:1],
                pred_poses,
                cam2pix[:, s, :, :],  ## [batchsize, scale, 3, 3]
                pix2cam[:, s, :, :])

            occu_mask = occu_masks[s]
            occu_mask_avg = tf.reduce_mean(occu_mask)

            curr_proj_image_depth = transformer_old(curr_src_image, depth_flow,
                                                    [H / (2**s), W / (2**s)])
            curr_proj_error_depth = tf.abs(curr_proj_image_depth -
                                           curr_tgt_image)
            pixel_loss_depth += (1.0 - opt.ssim_weight) * tf.reduce_mean(
                curr_proj_error_depth * occu_mask) / occu_mask_avg

            curr_flyout_map = occu_mask

            if opt.ssim_weight > 0:
                pixel_loss_depth += opt.ssim_weight * tf.reduce_mean(
                    SSIM(curr_proj_image_depth * occu_mask, curr_tgt_image *
                         occu_mask)) / occu_mask_avg

            tgt_image_all.append(curr_tgt_image)
            src_image_all.append(curr_src_image)
            proj_image_depth_all.append(curr_proj_image_depth)
            proj_error_depth_all.append(curr_proj_error_depth)

            flyout_map_all.append(curr_flyout_map)

        self.loss = (10.0 * pixel_loss_depth + stereo_smooth_loss)

        summaries.append(tf.summary.scalar("total_loss", self.loss))
        summaries.append(
            tf.summary.scalar("pixel_loss_depth", pixel_loss_depth))
        summaries.append(
            tf.summary.scalar("pixel_loss_optical", pixel_loss_optical))
        summaries.append(tf.summary.scalar("exp_loss", exp_loss))
        summaries.append(
            tf.summary.scalar("stereo_smooth_loss", stereo_smooth_loss))

        tf.summary.image("pred_disp", pred_disp[0][:, :, :, 0:1])
        # for s in range(opt.num_scales):
        s = 0
        tf.summary.histogram("pose_0-2", pred_poses[:, 0:3])
        tf.summary.histogram("pose_3-5", pred_poses[:, 3:6])
        tf.summary.image('scale%d_depth_image' % s,
                         pred_depth[s][:, :, :, 0:1])
        tf.summary.image('scale%d_right_disparity_image' % s,
                         pred_disp[s][:, :, :, 1:2])
        tf.summary.image('scale%d_target_image' % s, \
                         deprocess_image(tgt_image_all[s]))
        tf.summary.image('scale%d_src_image' % s, \
                         deprocess_image(src_image_all[s]))

        tf.summary.image('scale_projected_image',
                         deprocess_image(proj_image_depth_all[s]))
        tf.summary.image('scale_proj_error_error', proj_error_depth_all[s])
        tf.summary.image('scale_flyout_mask', flyout_map_all[s])
        self.summ_op = tf.summary.merge(summaries)


class Model_eval_depth(object):
    def __init__(self, scope=None):
        with tf.variable_scope(scope, reuse=True):
            input_uint8_1 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1')
            input_uint8_1r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1r')
            input_uint8_2 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2')
            input_uint8_2r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2r')
            input_intrinsic = tf.placeholder(tf.float32, [3, 3])

            cam2pix, pix2cam = get_multi_scale_intrinsics(input_intrinsic,
                                                          opt.num_scales)
            cam2pix = tf.expand_dims(cam2pix, axis=0)
            pix2cam = tf.expand_dims(pix2cam, axis=0)

            input_1 = preprocess_image(input_uint8_1)
            input_2 = preprocess_image(input_uint8_2)
            input_1r = preprocess_image(input_uint8_1r)
            input_2r = preprocess_image(input_uint8_2r)

            feature1_disp = feature_pyramid_disp(input_1, reuse=True)
            feature1r_disp = feature_pyramid_disp(input_1r, reuse=True)

            feature1_flow = feature_pyramid_flow(input_1, reuse=True)
            feature2_flow = feature_pyramid_flow(input_2, reuse=True)

            pred_disp = disp_godard(
                input_1,
                input_1r,
                feature1_disp,
                feature1r_disp,
                opt,
                is_training=False)
            pred_poses = pose_exp_net(input_1, input_2)

            optical_flows = construct_model_pwc_full(
                input_1, input_2, feature1_flow, feature2_flow)

            s = 0
            depth_flow, pose_mat = inverse_warp(
                1.0 / pred_disp[s][:, :, :, 0:1],
                pred_poses,
                cam2pix[:, s, :, :],  ## [batchsize, scale, 3, 3]
                pix2cam[:, s, :, :])

        self.input_1 = input_uint8_1
        self.input_2 = input_uint8_2
        self.input_r = input_uint8_1r
        self.input_2r = input_uint8_2r
        self.input_intrinsic = input_intrinsic

        self.pred_flow_rigid = depth_flow
        self.pred_flow_optical = optical_flows[0]
        self.pred_disp = pred_disp[0][:, :, :, 0:1]
        self.pred_pose_mat = pose_mat[0, :, :]

        # Placeholder created for interface consistency
        self.pred_disp2 = tf.constant(0.0)
        self.pred_mask = tf.constant(0.0)


class Model_depthflow(object):
    def __init__(self,
                 image1=None,
                 image2=None,
                 image1r=None,
                 image2r=None,
                 cam2pix=None,
                 pix2cam=None,
                 reuse_scope=False,
                 scope=None):
        summaries = []

        batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])

        with tf.variable_scope(scope, reuse=reuse_scope):
            feature1_flow = feature_pyramid_flow(image1, reuse=False)
            feature2_flow = feature_pyramid_flow(image2, reuse=True)

            feature1_disp = feature_pyramid_disp(image1, reuse=False)
            feature1r_disp = feature_pyramid_disp(image1r, reuse=True)

            pred_disp, stereo_smooth_loss = disp_godard(
                image1,
                image1r,
                feature1_disp,
                feature1r_disp,
                opt,
                is_training=True)

            pred_depth = [1. / d for d in pred_disp]
            pred_poses = pose_exp_net(image1, image2)

            optical_flows_rev = construct_model_pwc_full(
                image2, image1, feature2_flow, feature1_flow)

        with tf.variable_scope(scope, reuse=True):
            feature2_disp = feature_pyramid_disp(image2, reuse=True)
            feature2r_disp = feature_pyramid_disp(image2r, reuse=True)
            pred_disp_rev = disp_godard(
                image2,
                image2r,
                feature2_disp,
                feature2r_disp,
                opt,
                is_training=False)

            optical_flows = construct_model_pwc_full(
                image1, image2, feature1_flow, feature2_flow)

        occu_masks = [
            tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[batch_size, H / (2**s), W / (2**s), 1],
                        dtype='float32'),
                    flowr, [H / (2**s), W / (2**s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)
            for s, flowr in enumerate(optical_flows_rev)
        ]

        _, pose_mat, _, _ = inverse_warp_new(
            1.0 / pred_disp[0][:, :, :, 0:1], 1.0 /
            pred_disp_rev[0][:, :, :, 0:1], pred_poses, cam2pix[:, 0, :, :],
            pix2cam[:, 0, :, :], optical_flows[0], occu_masks[0])

        pixel_loss_depth = 0
        pixel_loss_optical = 0
        exp_loss = 0
        flow_smooth_loss = 0
        flow_consist_loss = 0
        tgt_image_all = []
        src_image_all = []
        proj_image_depth_all = []
        proj_error_depth_all = []
        flyout_map_all = []

        for s in range(opt.num_scales):
            occu_mask = occu_masks[s]
            # Scale the source and target images for computing loss at the 
            # according scale.
            curr_tgt_image = tf.image.resize_area(
                image1,
                [int(opt.img_height / (2**s)), int(opt.img_width / (2**s))])
            curr_src_image = tf.image.resize_area(
                image2,
                [int(opt.img_height / (2**s)), int(opt.img_width / (2**s))])

            depth_flow, pose_mat = inverse_warp(
                pred_depth[s][:, :, :, 0:1],
                tf.stop_gradient(pose_mat),
                cam2pix[:, s, :, :],  ## [batchsize, scale, 3, 3]
                pix2cam[:, s, :, :])

            depth_flow_orig, _ = inverse_warp(
                tf.stop_gradient(pred_depth[s][:, :, :, 0:1]),
                pred_poses,
                cam2pix[:, s, :, :],  ## [batchsize, scale, 3, 3]
                pix2cam[:, s, :, :])

            flow_diff = tf.sqrt(
                tf.reduce_sum(
                    tf.square(depth_flow - optical_flows[s]),
                    axis=3,
                    keep_dims=True))
            flow_diff_mask = tf.cast(
                flow_diff < (opt.flow_diff_threshold / 2**s), tf.float32)
            occu_region = tf.cast(occu_mask < 0.5, tf.float32)
            ref_exp_mask = tf.clip_by_value(
                flow_diff_mask + occu_region,
                clip_value_min=0.0,
                clip_value_max=1.0)

            occu_mask_avg = tf.reduce_mean(occu_mask)

            curr_proj_image_depth = transformer_old(curr_src_image, depth_flow,
                                                    [H / (2**s), W / (2**s)])
            curr_proj_error_depth = tf.abs(curr_proj_image_depth -
                                           curr_tgt_image) * ref_exp_mask
            pixel_loss_depth += (1.0 - opt.ssim_weight) * tf.reduce_mean(
                curr_proj_error_depth * occu_mask) / occu_mask_avg

            curr_proj_image_depth_orig = transformer_old(
                curr_src_image, depth_flow_orig, [H / (2**s), W / (2**s)])
            curr_proj_error_depth_orig = tf.abs(curr_proj_image_depth_orig -
                                                curr_tgt_image) * ref_exp_mask
            pixel_loss_depth += (1.0 - opt.ssim_weight) * tf.reduce_mean(
                curr_proj_error_depth_orig * occu_mask) / occu_mask_avg

            curr_proj_image_optical = transformer_old(
                curr_src_image, optical_flows[s], [H / (2**s), W / (2**s)])
            curr_proj_error_optical = tf.abs(curr_proj_image_optical -
                                             curr_tgt_image)
            pixel_loss_optical += (1.0 - opt.ssim_weight) * tf.reduce_mean(
                curr_proj_error_optical * occu_mask) / occu_mask_avg

            curr_flyout_map = occu_mask

            if opt.ssim_weight > 0:
                pixel_loss_depth += opt.ssim_weight * tf.reduce_mean(
                    SSIM(curr_proj_image_depth * occu_mask * ref_exp_mask,
                         curr_tgt_image * occu_mask *
                         ref_exp_mask)) / occu_mask_avg
                pixel_loss_depth += opt.ssim_weight * tf.reduce_mean(
                    SSIM(curr_proj_image_depth_orig * occu_mask * ref_exp_mask,
                         curr_tgt_image * occu_mask *
                         ref_exp_mask)) / occu_mask_avg
                pixel_loss_optical += opt.ssim_weight * tf.reduce_mean(
                    SSIM(curr_proj_image_optical * occu_mask, curr_tgt_image *
                         occu_mask)) / occu_mask_avg

#         
            flow_smooth_loss += opt.flow_smooth_weight * cal_grad2_error_mask(
                optical_flows[s] / 20.0, curr_tgt_image, 1.0,
                1.0 - ref_exp_mask)
            depth_flow_stop = tf.stop_gradient(depth_flow)
            flow_consist_loss += opt.flow_consist_weight * charbonnier_loss(
                depth_flow_stop - optical_flows[s], ref_exp_mask)

            tgt_image_all.append(curr_tgt_image)
            src_image_all.append(curr_src_image)
            proj_image_depth_all.append(curr_proj_image_depth)
            proj_error_depth_all.append(curr_proj_error_depth)

            flyout_map_all.append(curr_flyout_map)

        self.loss = (
            10.0 * pixel_loss_depth + stereo_smooth_loss
        ) + pixel_loss_optical + flow_smooth_loss + flow_consist_loss

        summaries.append(tf.summary.scalar("total_loss", self.loss))
        summaries.append(
            tf.summary.scalar("pixel_loss_depth", pixel_loss_depth))
        summaries.append(
            tf.summary.scalar("pixel_loss_optical", pixel_loss_optical))
        summaries.append(tf.summary.scalar("exp_loss", exp_loss))
        summaries.append(
            tf.summary.scalar("stereo_smooth_loss", stereo_smooth_loss))

        tf.summary.image("pred_disp", pred_disp[0][:, :, :, 0:1])
        s = 0
        tf.summary.histogram("pose_0-2", pred_poses[:, 0:3])
        tf.summary.histogram("pose_3-5", pred_poses[:, 3:6])
        tf.summary.image('scale%d_depth_image' % s,
                         pred_depth[s][:, :, :, 0:1])
        tf.summary.image('scale%d_right_disparity_image' % s,
                         pred_disp[s][:, :, :, 1:2])
        tf.summary.image('scale%d_target_image' % s, \
                         deprocess_image(tgt_image_all[s]))
        tf.summary.image('scale%d_src_image' % s, \
                         deprocess_image(src_image_all[s]))

        tf.summary.image('scale_projected_image',
                         deprocess_image(proj_image_depth_all[s]))
        tf.summary.image('scale_proj_error_error', proj_error_depth_all[s])
        tf.summary.image('scale_flyout_mask', flyout_map_all[s])
        self.summ_op = tf.summary.merge(summaries)


class Model_eval_depthflow(object):
    def __init__(self, scope=None):
        with tf.variable_scope(scope, reuse=True):
            input_uint8_1 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1')
            input_uint8_1r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_1r')
            input_uint8_2 = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2')
            input_uint8_2r = tf.placeholder(
                tf.uint8, [1, opt.img_height, opt.img_width, 3],
                name='raw_input_2r')
            input_intrinsic = tf.placeholder(tf.float32, [3, 3])

            cam2pix, pix2cam = get_multi_scale_intrinsics(input_intrinsic,
                                                          opt.num_scales)
            cam2pix = tf.expand_dims(cam2pix, axis=0)
            pix2cam = tf.expand_dims(pix2cam, axis=0)

            input_1 = preprocess_image(input_uint8_1)
            input_2 = preprocess_image(input_uint8_2)
            input_1r = preprocess_image(input_uint8_1r)
            input_2r = preprocess_image(input_uint8_2r)

            feature1_disp = feature_pyramid_disp(input_1, reuse=True)
            feature1r_disp = feature_pyramid_disp(input_1r, reuse=True)

            feature2_disp = feature_pyramid_disp(input_2, reuse=True)
            feature2r_disp = feature_pyramid_disp(input_2r, reuse=True)

            feature1_flow = feature_pyramid_flow(input_1, reuse=True)
            feature2_flow = feature_pyramid_flow(input_2, reuse=True)

            pred_disp = disp_godard(
                input_1,
                input_1r,
                feature1_disp,
                feature1r_disp,
                opt,
                is_training=False)
            pred_disp_rev = disp_godard(
                input_2,
                input_2r,
                feature2_disp,
                feature2r_disp,
                opt,
                is_training=False)

            pred_poses = pose_exp_net(input_1, input_2)

            optical_flows = construct_model_pwc_full(
                input_1, input_2, feature1_flow, feature2_flow)
            optical_flows_rev = construct_model_pwc_full(
                input_2, input_1, feature2_flow, feature1_flow)

            s = 0
            occu_mask = tf.clip_by_value(
                transformerFwd(
                    tf.ones(
                        shape=[
                            1, opt.img_height // (2**s),
                            opt.img_width // (2**s), 1
                        ],
                        dtype='float32'),
                    optical_flows_rev[s],
                    [opt.img_height // (2**s), opt.img_width // (2**s)]),
                clip_value_min=0.0,
                clip_value_max=1.0)

            depth_flow, pose_mat, disp1_trans, small_mask = inverse_warp_new(
                1.0 / pred_disp[0][:, :, :, 0:1],
                1.0 / pred_disp_rev[0][:, :, :, 0:1], pred_poses,
                cam2pix[:, 0, :, :], pix2cam[:, 0, :, :], optical_flows[0],
                occu_mask)

            flow_diff = tf.sqrt(
                tf.reduce_sum(
                    tf.square(depth_flow - optical_flows[0]),
                    axis=3,
                    keep_dims=True))
            flow_diff_mask = tf.cast(flow_diff < (opt.flow_diff_threshold),
                                     tf.float32)
            occu_region = tf.cast(occu_mask < 0.5, tf.float32)
            ref_exp_mask = tf.clip_by_value(
                flow_diff_mask + occu_region,
                clip_value_min=0.0,
                clip_value_max=1.0)

        self.input_1 = input_uint8_1
        self.input_2 = input_uint8_2
        self.input_r = input_uint8_1r
        self.input_2r = input_uint8_2r
        self.input_intrinsic = input_intrinsic
        self.pred_pose_mat = pose_mat[0, :, :]

        self.pred_flow_rigid = depth_flow
        self.pred_flow_optical = optical_flows[0]
        self.pred_disp = pred_disp[0][:, :, :, 0:1]
        self.pred_disp2 = disp1_trans*0.0 + \
                          transformer_old(pred_disp_rev[0][:,:,:,0:1], optical_flows[0], [opt.img_height, opt.img_width])*(1.0-0.0)
        self.pred_mask = 1.0 - ref_exp_mask
