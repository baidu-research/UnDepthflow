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
from flowlib import read_flow, write_flow, flow_to_image
import pdb
from plot import normalize_disp_for_display
import pdb
import matplotlib.pyplot as plt

opt = flags.FLAGS


def mkdir_p(path):
    if not os.path.exists(path):
        os.mkdir(path)


def cal_err_rate(pred_disp, gt_disp):
    epe_map = np.abs(pred_disp - gt_disp)
    bad_pixels = np.logical_and(epe_map >= 3,
                                epe_map / np.maximum(gt_disp, 1.0e-10) >= 0.05)

    return 1.0 * bad_pixels.sum() / (gt_disp.shape[0] * gt_disp.shape[1])


def test_monkaa(sess,
                eval_model,
                itr,
                gt_dir,
                test_files,
                output_dir,
                write_file=False):
    grey_cmap = plt.get_cmap("Greys")
    with tf.name_scope("evaluation"):
        sys.stderr.write("Evaluation at iter [" + str(itr) + "]: \n")
        flow_epes, flow_rigid_epes, disp_err_rates = [], [], []
        pred_flows, pred_disps, pred_masks, img1s, img2s = [], [], [], [], []
        gt_flows, gt_disps = [], []

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
            img1s.append(img1)
            orig_H, orig_W = img1.shape[0:2]
            img1 = sm.imresize(img1, (opt.img_height, opt.img_width))

            img2 = sm.imread(os.path.join(gt_dir, left_2))
            img2s.append(img2)
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
            pred_mask = grey_cmap(
                cv2.resize(
                    np.squeeze(pred_mask), (orig_W, orig_H),
                    interpolation=cv2.INTER_LINEAR))[:, :, 0:3]
            pred_masks.append(0.7 * img1s[-1] + 1.0 * (pred_mask) * 255.0)

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
                pred_flows.append(pred_flow)
                gt_flows.append(gt_flow)

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
                pred_disp = orig_W * cv2.resize(
                    np.squeeze(pred_disp), (orig_W, orig_H),
                    interpolation=cv2.INTER_LINEAR)
                disp_err_rate = cal_err_rate(gt_disp, pred_disp)
                disp_err_rates.append(disp_err_rate)
                pred_disps.append(pred_disp)
                gt_disps.append(gt_disp)

        if opt.eval_flow:
            sys.stderr.write("monkaa flow err is %.4f \n" %
                             (sum(flow_epes) / len(flow_epes)))
            if opt.mode in ["depth", "depthflow"]:
                sys.stderr.write("monkaa rigid flow err is %.4f \n" %
                                 (sum(flow_rigid_epes) / len(flow_rigid_epes)))
            if write_file:
                mkdir_p(os.path.join(output_dir, "flow"))
                mkdir_p(os.path.join(output_dir, "flow_gt"))
                for i, pred_flow, gt_flow in zip(
                        range(len(pred_flows)), pred_flows, gt_flows):
                    sm.imsave(
                        os.path.join(output_dir, "flow",
                                     str(i).zfill(3) + ".png"),
                        flow_to_image(pred_flow))
                    sm.imsave(
                        os.path.join(output_dir, "flow_gt",
                                     str(i).zfill(3) + ".png"),
                        flow_to_image(gt_flow))

                mkdir_p(os.path.join(output_dir, "img1"))
                mkdir_p(os.path.join(output_dir, "img2"))
                for i, img1, img2 in zip(range(len(img1s)), img1s, img2s):
                    sm.imsave(
                        os.path.join(output_dir, "img1",
                                     str(i).zfill(3) + "_1.png"), img1)
                    sm.imsave(
                        os.path.join(output_dir, "img2",
                                     str(i).zfill(3) + "_2.png"), img2)

        if opt.eval_depth:
            sys.stderr.write("monkaa disp err is %.4f \n" %
                             (sum(disp_err_rates) / len(disp_err_rates)))
            if write_file:
                mkdir_p(os.path.join(output_dir, "disp"))
                mkdir_p(os.path.join(output_dir, "disp_gt"))
                for i, pred_disp, gt_disp in zip(
                        range(len(pred_disps)), pred_disps, gt_disps):
                    disp_plot = normalize_disp_for_display(
                        pred_disp, cmap="plasma")
                    gt_disp_plot = normalize_disp_for_display(
                        gt_disp, cmap="plasma")
                    sm.imsave(
                        os.path.join(output_dir, "disp",
                                     str(i).zfill(3) + ".png"), disp_plot)
                    sm.imsave(
                        os.path.join(output_dir, "disp_gt",
                                     str(i).zfill(3) + ".png"), gt_disp_plot)

        if opt.eval_mask and write_file:
            mkdir_p(os.path.join(output_dir, "mask"))
            for i, pred_mask in enumerate(pred_masks):
                sm.imsave(
                    os.path.join(output_dir, "mask", str(i).zfill(3) + ".png"),
                    pred_mask)
