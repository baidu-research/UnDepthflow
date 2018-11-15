import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc as sm
from flowlib import read_flow_png, flow_to_image
from PIL import Image


def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def normalize_disp_for_display(disp,
                               pc=95,
                               crop_percent=0,
                               normalizer=None,
                               cmap='gray'):
    # convert to disparity
    if normalizer is not None:
        disp = disp / normalizer
    else:
        disp = disp / (np.percentile(disp, pc) + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_H = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_H]
    disp = disp
    return disp


def plot_disp(input_path, output_path, cmap):
    files = os.listdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        disp = sm.imread(os.path.join(input_path, file), -1)
        disp = disp.astype(np.float32) / 256.0
        disp_plot = normalize_disp_for_display(disp, cmap=cmap)
        sm.imsave(os.path.join(output_path, file), disp_plot)


def plot_flow(input_path, output_path):
    files = os.listdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        flow = read_flow_png(os.path.join(input_path, file))
        sm.imsave(os.path.join(output_path, file), flow_to_image(flow))


def combine_png():
    output_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/all_plot"

    gt_flow_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/flow_occ_plot"
    pred_flow_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/flow_plot"

    gt_disp_dir = "/mnt/data/wangyang59/kitti_zhenheng/results/depth_gt_kitti"
    pred_disp_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/disp0_plot"

    mask_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/pred_mask"

    img_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/image_2"

    for i in range(200):
        img_no = str(i).zfill(6)
        gt_flow = Image.open(os.path.join(gt_flow_dir, img_no + "_10.png"))
        pred_flow = Image.open(os.path.join(pred_flow_dir, img_no + "_10.png"))

        gt_disp = Image.open(os.path.join(gt_disp_dir, img_no + "_10.png"))
        pred_disp = Image.open(os.path.join(pred_disp_dir, img_no + "_10.png"))

        gt_mask = Image.open(os.path.join(mask_dir, img_no + "_10_gt.png"))
        pred_mask = Image.open(os.path.join(mask_dir, img_no + "_10.png"))

        img_1 = Image.open(os.path.join(img_dir, img_no + "_10.png"))
        img_2 = Image.open(os.path.join(img_dir, img_no + "_11.png"))

        (width, height) = gt_flow.size

        result_width = 4 * width
        result_height = 2 * height

        result = Image.new('RGB', (result_width, result_height))

        result.paste(im=img_1, box=(0, 0))
        result.paste(im=img_2, box=(0, height))

        result.paste(im=gt_flow, box=(width, 0))
        result.paste(im=pred_flow, box=(width, height))

        result.paste(im=gt_disp, box=(2 * width, 0))
        result.paste(im=pred_disp, box=(2 * width, height))

        result.paste(im=gt_mask, box=(3 * width, 0))
        result.paste(im=pred_mask, box=(3 * width, height))

        result.save(os.path.join(output_dir, img_no + "_10.png"))


def combine_png2():
    output_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/compare_plot_plasma"

    gt_flow_dir = "/mnt/home/wangyang59/Experiments/flow_exp/fn2_unsup_test/pred_flow_plot"
    pred_flow_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/flow_plot"

    #gt_disp_dir = "/mnt/home/wangyang59/Projects/baselines/monodepth/model/disp_plot"
    gt_disp_dir = "/mnt/home/wangyang59/best_sssmnet_2015_training/plot"
    pred_disp_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/disp0_plot_plasma"

    img_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/image_2"

    for i in range(200):
        img_no = str(i).zfill(6)

        img_1 = Image.open(os.path.join(img_dir, img_no + "_10.png"))
        img_2 = Image.open(os.path.join(img_dir, img_no + "_11.png"))

        gt_flow = Image.open(os.path.join(gt_flow_dir, img_no + "_10.png"))
        pred_flow = Image.open(os.path.join(pred_flow_dir, img_no + "_10.png"))

        gt_disp = Image.open(os.path.join(gt_disp_dir, img_no + "_10.png"))
        pred_disp = Image.open(os.path.join(pred_disp_dir, img_no + "_10.png"))

        (width, height) = gt_flow.size

        result_width = 3 * width
        result_height = 2 * height

        result = Image.new('RGB', (result_width, result_height))

        result.paste(im=img_1, box=(0, 0))
        result.paste(im=img_2, box=(0, height))

        result.paste(im=gt_flow, box=(width, 0))
        result.paste(im=pred_flow, box=(width, height))

        result.paste(im=gt_disp, box=(width * 2, 0))
        result.paste(im=pred_disp, box=(width * 2, height))

        result.save(os.path.join(output_dir, img_no + "_10.png"))
