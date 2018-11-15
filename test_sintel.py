import numpy as np
import tensorflow as tf
import scipy.misc as sm

from eval.evaluate_flow import get_scaled_intrinsic_matrix, eval_flow_avg,\
  scale_intrinsics
from eval.evaluate_disp import eval_disp_avg
from eval.sintel_io import disparity_read, cam_read

import cv2

import re, os
import sys

from tensorflow.python.platform import flags
from flowlib import read_flow, write_flow
import pdb
opt = flags.FLAGS


def cal_err_rate(pred_disp, gt_disp):
    epe_map = np.abs(pred_disp - gt_disp)
    bad_pixels = np.logical_and(epe_map >= 3,
                                epe_map / np.maximum(gt_disp, 1.0e-10) >= 0.05)

    return 1.0 * bad_pixels.sum() / (gt_disp.shape[0] * gt_disp.shape[1])


def test_sintel(sess, eval_model, itr, gt_dir, prefix):
    with tf.name_scope("evaluation"):
        sys.stderr.write("Evaluation at iter [" + str(itr) + "]: \n")
        flow_epes, disp_err_rates = [], []

        scenes = os.listdir(os.path.join(gt_dir, "flow"))

        for scene in scenes:
            frames = sorted(
                os.listdir(os.path.join(gt_dir, "final_left", scene)))
            for i in range(len(frames) - 1):
                img1 = sm.imread(
                    os.path.join(gt_dir, prefix + "_left", scene, frames[i]))
                orig_H, orig_W = img1.shape[0:2]
                img1 = sm.imresize(img1, (opt.img_height, opt.img_width))

                img2 = sm.imread(
                    os.path.join(gt_dir, prefix + "_left", scene, frames[i +
                                                                         1]))
                img2 = sm.imresize(img2, (opt.img_height, opt.img_width))

                imgr = sm.imread(
                    os.path.join(gt_dir, prefix + "_right", scene, frames[i]))
                imgr = sm.imresize(imgr, (opt.img_height, opt.img_width))

                img2r = sm.imread(
                    os.path.join(gt_dir, prefix + "_right", scene, frames[i +
                                                                          1]))
                img2r = sm.imresize(img2r, (opt.img_height, opt.img_width))

                img1 = np.expand_dims(img1, axis=0)
                img2 = np.expand_dims(img2, axis=0)
                imgr = np.expand_dims(imgr, axis=0)
                img2r = np.expand_dims(img2r, axis=0)

                gt_flow = read_flow(
                    os.path.join(gt_dir, "flow", scene, frames[i].replace(
                        "png", "flo")))
                gt_disp = disparity_read(
                    os.path.join(gt_dir, "disparities", scene, frames[i]))

                input_intrinsic = cam_read(
                    os.path.join(gt_dir, "depth/camdata_left", scene,
                                 "frame_0001.cam"))[0]
                input_intrinsic = scale_intrinsics(
                    input_intrinsic,
                    sx=1.0 * opt.img_width / orig_W,
                    sy=1.0 * opt.img_height / orig_H)

                pred_flow_rigid, pred_flow_optical, \
                pred_disp, pred_disp2, pred_mask= sess.run([eval_model.pred_flow_rigid,
                                                         eval_model.pred_flow_optical,
                                                         eval_model.pred_disp,
                                                         eval_model.pred_disp2,
                                                         eval_model.pred_mask],
                                                          feed_dict = {eval_model.input_1: img1,
                                                                       eval_model.input_2: img2,
                                                                       eval_model.input_r: imgr,
                                                                       eval_model.input_2r:img2r,
                                                                       eval_model.input_intrinsic: input_intrinsic})

                if opt.eval_flow:
                    pred_flow = np.squeeze(pred_flow_optical)
                    pred_flow[:, :, 0] = pred_flow[:, :,
                                                   0] / opt.img_width * orig_W
                    pred_flow[:, :, 1] = pred_flow[:, :,
                                                   1] / opt.img_height * orig_H

                    pred_flow = cv2.resize(
                        pred_flow, (orig_W, orig_H),
                        interpolation=cv2.INTER_LINEAR)
                    flow_epe = np.mean(
                        np.sqrt(
                            np.sum(np.square(pred_flow - gt_flow), axis=2)))
                    flow_epes.append(flow_epe)

                if opt.eval_depth:
                    disp_err_rate = cal_err_rate(
                        gt_disp,
                        orig_W * cv2.resize(
                            np.squeeze(pred_disp), (orig_W, orig_H),
                            interpolation=cv2.INTER_LINEAR))
                    disp_err_rates.append(disp_err_rate)

        if opt.eval_flow:
            sys.stderr.write("sintel flow err is %.4f \n" %
                             (sum(flow_epes) / len(flow_epes)))
        if opt.eval_depth:
            sys.stderr.write("sintel disp err is %.4f \n" %
                             (sum(disp_err_rates) / len(disp_err_rates)))


def test_sintel_flow(sess, eval_model, itr, gt_dir, prefix):
    with tf.name_scope("evaluation"):
        sys.stderr.write("Evaluation at iter [" + str(itr) + "]: \n")
        scenes = os.listdir(os.path.join(gt_dir, "clean"))

        for scene in scenes:
            if not os.path.exists(
                    os.path.join(opt.trace, "sintel_test", scene)):
                os.makedirs(os.path.join(opt.trace, "sintel_test", scene))
            frames = sorted(os.listdir(os.path.join(gt_dir, prefix, scene)))
            for i in range(len(frames) - 1):
                img1 = sm.imread(
                    os.path.join(gt_dir, prefix, scene, frames[i]))
                orig_H, orig_W = img1.shape[0:2]
                img1 = sm.imresize(img1, (opt.img_height, opt.img_width))

                img2 = sm.imread(
                    os.path.join(gt_dir, prefix, scene, frames[i + 1]))
                img2 = sm.imresize(img2, (opt.img_height, opt.img_width))

                imgr = sm.imread(
                    os.path.join(gt_dir, prefix, scene, frames[i]))
                imgr = sm.imresize(imgr, (opt.img_height, opt.img_width))

                img2r = sm.imread(
                    os.path.join(gt_dir, prefix, scene, frames[i + 1]))
                img2r = sm.imresize(img2r, (opt.img_height, opt.img_width))

                img1 = np.expand_dims(img1, axis=0)
                img2 = np.expand_dims(img2, axis=0)
                imgr = np.expand_dims(imgr, axis=0)
                img2r = np.expand_dims(img2r, axis=0)

                input_intrinsic = np.zeros([3, 3])

                pred_flow_rigid, pred_flow_optical, \
                pred_disp, pred_disp2, pred_mask= sess.run([eval_model.pred_flow_rigid,
                                                         eval_model.pred_flow_optical,
                                                         eval_model.pred_disp,
                                                         eval_model.pred_disp2,
                                                         eval_model.pred_mask],
                                                          feed_dict = {eval_model.input_1: img1,
                                                                       eval_model.input_2: img2,
                                                                       eval_model.input_r: imgr,
                                                                       eval_model.input_2r:img2r,
                                                                       eval_model.input_intrinsic: input_intrinsic})

                if opt.eval_flow:
                    pred_flow = np.squeeze(pred_flow_optical)
                    pred_flow[:, :, 0] = pred_flow[:, :,
                                                   0] / opt.img_width * orig_W
                    pred_flow[:, :, 1] = pred_flow[:, :,
                                                   1] / opt.img_height * orig_H

                    pred_flow = cv2.resize(
                        pred_flow, (orig_W, orig_H),
                        interpolation=cv2.INTER_LINEAR)
                    write_flow(pred_flow,
                               os.path.join(opt.trace, "sintel_test", scene,
                                            frames[i].replace("png", "flo")))

    sys.exit()
