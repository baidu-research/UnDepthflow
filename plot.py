import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc as sm
from flowlib import read_flow_png, flow_to_image, read_flow_png_fill
from PIL import Image
import cv2
import skimage.io


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


def plot_mask(input_path, output_path):
    grey_cmap = plt.get_cmap("Greys")
    files = os.listdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        mask = sm.imread(os.path.join(input_path, file))
        sm.imsave(os.path.join(output_path, file), grey_cmap(mask))


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
    output_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/compare_plot_plasma_godard"

    gt_flow_dir = "/mnt/home/wangyang59/Experiments/flow_exp/fn2_unsup_test/pred_flow_plot"
    pred_flow_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/flow_plot"

    gt_disp_dir = "/mnt/home/wangyang59/Projects/baselines/monodepth/model/disp_plot_plasma"
    #gt_disp_dir = "/mnt/home/wangyang59/best_sssmnet_2015_training/plot"
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


def plot_godard(data, output_dir):
    img_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/image_2"
    for i in range(200):
        img_no = str(i).zfill(6)
        img1 = Image.open(os.path.join(img_dir, img_no + "_10.png"))
        W, H = img1.size
        disp = data[i]
        disp = W * cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR)
        skimage.io.imsave(
            os.path.join(output_dir, str(i).zfill(6) + "_10.png"),
            (disp * 256).astype('uint16'))


def combine_png_compare():
    output_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/all_plot_disp"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    gt_flow_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/flow_occ_plot"
    pred_flow_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/flow_plot"
    baseline_flow_dir = "/mnt/home/wangyang59/Experiments/flow_exp/fn2_unsup_test/pred_flow_plot"

    gt_disp_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/disp_occ_0_fill_plasma"
    pred_disp_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/kitti_2015_val/disp0_plot_plasma"
    baseline_disp_dir = "/mnt/home/wangyang59/Projects/baselines/monodepth/model/disp_plot_plasma"

    mask_dir = "/mnt/home/wangyang59/Experiments/depth_exp/depthflow_ml_depix10_flow10_floerrmask3_floconsist0.01_corr_trainpose2_test/pred_mask"
    baseline_mask_dir = "/mnt/home/wangyang59/Projects/baselines/kitti_seg_res_plot"

    img_dir = "/mnt/data/wangyang59/kitti_zhenheng/training/image_2"

    for i in range(200):
        img_no = str(i).zfill(6)
        gt_flow = Image.open(os.path.join(gt_flow_dir, img_no + "_10.png"))
        pred_flow = Image.open(os.path.join(pred_flow_dir, img_no + "_10.png"))
        bl_flow = Image.open(
            os.path.join(baseline_flow_dir, img_no + "_10.png"))

        gt_disp = Image.open(os.path.join(gt_disp_dir, img_no + "_10.png"))
        pred_disp = Image.open(os.path.join(pred_disp_dir, img_no + "_10.png"))
        bl_disp = Image.open(
            os.path.join(baseline_disp_dir, img_no + "_10.png"))

        gt_mask = Image.open(os.path.join(mask_dir, img_no + "_10_gt.png"))
        pred_mask = Image.open(os.path.join(mask_dir, img_no + "_10_plot.png"))
        bl_mask = Image.open(
            os.path.join(baseline_mask_dir, img_no + "_10.png"))

        img_1 = Image.open(os.path.join(img_dir, img_no + "_10.png"))
        img_2 = Image.open(os.path.join(img_dir, img_no + "_11.png"))

        (width, height) = gt_flow.size

        result_width = 4 * width
        result_height = 1 * height

        result = Image.new('RGB', (result_width, result_height))

        result.paste(im=img_1, box=(0, 0))

        result.paste(im=gt_mask, box=(width, 0))
        result.paste(im=pred_mask, box=(width * 2, 0))
        result.paste(im=bl_mask, box=(3 * width, 0))

        result.save(os.path.join(output_dir, img_no + "_10.png"))


def combine_errormap_compare():
    output_dir = "/mnt/home/wangyang59/Experiments/depth_exp/error_maps_compare/disp"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    flow_only_dir = "/mnt/scratch/wangyang59/cpp_test/results/stereo_only/errors_disp_img_0"
    ego_dir = "/mnt/scratch/wangyang59/cpp_test/results/ego/errors_disp_img_0"
    ego_rdvo_dir = "/mnt/scratch/wangyang59/cpp_test/results/ego+rdvo/errors_disp_img_0"
    full_dir = "/mnt/scratch/wangyang59/cpp_test/results/full/errors_disp_img_0"

    for i in range(200):
        img_no = str(i).zfill(6)
        flow_only = Image.open(os.path.join(flow_only_dir, img_no + "_10.png"))
        ego_map = Image.open(os.path.join(ego_dir, img_no + "_10.png"))
        ego_rdvo_map = Image.open(
            os.path.join(ego_rdvo_dir, img_no + "_10.png"))
        full_map = Image.open(os.path.join(full_dir, img_no + "_10.png"))

        (width, height) = flow_only.size

        result_width = 2 * width
        result_height = 2 * height

        result = Image.new('RGB', (result_width, result_height))

        result.paste(im=flow_only, box=(0, 0))

        result.paste(im=ego_map, box=(width, 0))
        result.paste(im=ego_rdvo_map, box=(width, height))
        result.paste(im=full_map, box=(0, height))

        result.save(os.path.join(output_dir, img_no + "_10.png"))


def combine_png_monkaa():
    root_dir = "/mnt/home/wangyang59/Experiments/depth_exp/test_clean_code_monkaa_depthflow_ft_small_pixdepth1.0_test/"
    output_dir = os.path.join(root_dir, "combo_plot")

    gt_flow_dir = os.path.join(root_dir, "flow_gt")
    pred_flow_dir = os.path.join(root_dir, "flow")

    gt_disp_dir = os.path.join(root_dir, "disp_gt")
    pred_disp_dir = os.path.join(root_dir, "disp")

    mask_dir = os.path.join(root_dir, "mask")

    img1_dir = os.path.join(root_dir, "img1")
    img2_dir = os.path.join(root_dir, "img2")

    for i in range(438):
        img_no = str(i).zfill(3)

        img_1 = Image.open(os.path.join(img1_dir, img_no + "_1.png"))
        img_2 = Image.open(os.path.join(img2_dir, img_no + "_2.png"))

        gt_flow = Image.open(os.path.join(gt_flow_dir, img_no + ".png"))
        pred_flow = Image.open(os.path.join(pred_flow_dir, img_no + ".png"))

        gt_disp = Image.open(os.path.join(gt_disp_dir, img_no + ".png"))
        pred_disp = Image.open(os.path.join(pred_disp_dir, img_no + ".png"))

        mask = Image.open(os.path.join(mask_dir, img_no + ".png"))

        (width, height) = gt_flow.size

        result_width = 3 * width
        result_height = 2 * height

        result = Image.new('RGB', (result_width, result_height))

        result.paste(im=img_1, box=(0, 0))
        result.paste(im=mask, box=(0, height))

        result.paste(im=gt_flow, box=(width, 0))
        result.paste(im=pred_flow, box=(width, height))

        result.paste(im=gt_disp, box=(width * 2, 0))
        result.paste(im=pred_disp, box=(width * 2, height))

        result.save(os.path.join(output_dir, img_no + ".png"))
