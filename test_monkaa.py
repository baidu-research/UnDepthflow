import numpy as np
import tensorflow as tf
import scipy.misc as sm

from eval.evaluate_flow import get_scaled_intrinsic_matrix, eval_flow_avg,\
  scale_intrinsics
from eval.evaluate_disp import eval_disp_avg
from eval.monkaa_io import readPFM

import cv2
import pdb

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


def test_monkaa(sess, eval_model, itr, gt_dir, test_files):
    with tf.name_scope("evaluation"):
        sys.stderr.write("Evaluation at iter [" + str(itr) + "]: \n")
        flow_epes, flow_rigid_epes, disp_err_rates = [], [], []

        with open(test_files, "r") as f:
            filenames = f.readlines()

        for file in filenames:
            left_1, right_1, left_2, right_2, _ = file.strip().split()
            scene = left_1.split("/")[1]
            frame_no = left_1.split("/")[-1][:-4]
            gt_flow_file = os.path.join(
                gt_dir, "optical_flow", scene, "into_future", "left",
                "OpticalFlowIntoFuture_%s_L.pfm" % frame_no)
            gt_disp_file = os.path.join(gt_dir, "disparity", scene, "left",
                                        "%s.pfm" % frame_no)

            img1 = sm.imread(os.path.join(gt_dir, left_1))
            orig_H, orig_W = img1.shape[0:2]
            img1 = sm.imresize(img1, (opt.img_height, opt.img_width))

            img2 = sm.imread(os.path.join(gt_dir, left_2))
            img2 = sm.imresize(img2, (opt.img_height, opt.img_width))

            imgr = sm.imread(os.path.join(gt_dir, right_1))
            imgr = sm.imresize(imgr, (opt.img_height, opt.img_width))

            img2r = sm.imread(os.path.join(gt_dir, right_2))
            img2r = sm.imresize(img2r, (opt.img_height, opt.img_width))

            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)
            imgr = np.expand_dims(imgr, axis=0)
            img2r = np.expand_dims(img2r, axis=0)

            gt_flow = readPFM(gt_flow_file)[0][:, :, 0:2]
            gt_disp = readPFM(gt_disp_file)[0]

            input_intrinsic = np.array([[1050.0, 0.0, 479.5],
                                        [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])
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
                    np.sqrt(np.sum(np.square(pred_flow - gt_flow), axis=2)))
                flow_epes.append(flow_epe)

                if opt.mode in ["depth", "depthflow"]:
                    pred_flow = np.squeeze(pred_flow_rigid)
                    pred_flow[:, :, 0] = pred_flow[:, :,
                                                   0] / opt.img_width * orig_W
                    pred_flow[:, :, 1] = pred_flow[:, :,
                                                   1] / opt.img_height * orig_H

                    pred_flow = cv2.resize(
                        pred_flow, (orig_W, orig_H),
                        interpolation=cv2.INTER_LINEAR)
                    flow_rigid_epe = np.mean(
                        np.sqrt(
                            np.sum(np.square(pred_flow - gt_flow), axis=2)))
                    flow_rigid_epes.append(flow_rigid_epe)

            if opt.eval_depth:
                disp_err_rate = cal_err_rate(
                    gt_disp,
                    orig_W * cv2.resize(
                        np.squeeze(pred_disp), (orig_W, orig_H),
                        interpolation=cv2.INTER_LINEAR))
                disp_err_rates.append(disp_err_rate)

        if opt.eval_flow:
            sys.stderr.write("monkaa flow err is %.4f \n" %
                             (sum(flow_epes) / len(flow_epes)))
            if opt.mode in ["depth", "depthflow"]:
                sys.stderr.write("monkaa rigid flow err is %.4f \n" %
                                 (sum(flow_rigid_epes) / len(flow_rigid_epes)))

        if opt.eval_depth:
            sys.stderr.write("monkaa disp err is %.4f \n" %
                             (sum(disp_err_rates) / len(disp_err_rates)))
