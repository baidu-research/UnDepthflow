import numpy as np
import scipy.misc as sm
import cv2


def calculate_error_rate(epe_map, gt_disp, mask):
    bad_pixels = np.logical_and(
        epe_map * mask >= 3,
        epe_map * mask / np.maximum(gt_disp, 1.0 - mask) >= 0.05)
    if mask.sum() > 0:
        return bad_pixels.sum() / mask.sum()
    else:
        return 0


def eval_disp_avg(pred_disps, path, disp_num=None, moving_masks=None):
    error, error_noc, error_occ, error_move, error_static, error_rate = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num = len(pred_disps)
    for i, pred_disp in enumerate(pred_disps):
        if disp_num is not None:
            gt_disp = sm.imread(path + "/disp_occ_" + str(disp_num) + "/" +
                                str(i).zfill(6) + "_10.png", -1)
            gt_disp_noc = sm.imread(path + "/disp_noc_" + str(disp_num) + "/" +
                                    str(i).zfill(6) + "_10.png", -1)
        else:
            gt_disp = sm.imread(
                path + "/disp_occ/" + str(i).zfill(6) + "_10.png", -1)
            gt_disp_noc = sm.imread(
                path + "/disp_noc/" + str(i).zfill(6) + "_10.png", -1)

        gt_disp = gt_disp.astype(np.float32) / 256.0
        gt_disp_noc = gt_disp_noc.astype(np.float32) / 256.0

        noc_mask = (gt_disp_noc > 0.0).astype(np.float32)
        valid_mask = (gt_disp > 0.0).astype(np.float32)

        H, W = gt_disp.shape[0:2]

        pred_disp = W * cv2.resize(
            pred_disp, (W, H), interpolation=cv2.INTER_LINEAR)

        epe_map = np.abs(pred_disp - gt_disp)
        error += np.sum(epe_map * valid_mask) / np.sum(valid_mask)

        error_noc += calculate_error_rate(epe_map, gt_disp, noc_mask)

        error_occ += calculate_error_rate(epe_map, gt_disp,
                                          valid_mask - noc_mask)

        error_rate += calculate_error_rate(epe_map, gt_disp, valid_mask)

        if moving_masks:
            move_mask = moving_masks[i]

            error_move += calculate_error_rate(epe_map, gt_disp,
                                               valid_mask * move_mask)
            error_static += calculate_error_rate(epe_map, gt_disp, valid_mask *
                                                 (1.0 - move_mask))

    if moving_masks:
        result = "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".format(
            'epe', 'noc_rate', 'occ_rate', 'move_rate', 'static_rate',
            'err_rate')
        result += "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f} \n".format(
            error / num, error_noc / num, error_occ / num, error_move / num,
            error_static / num, error_rate / num)
        return result
    else:
        result = "{:>10}, {:>10}, {:>10}, {:>10} \n".format(
            'epe', 'noc_rate', 'occ_rate', 'err_rate')
        result += "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f} \n".format(
            error / num, error_noc / num, error_occ / num, error_rate / num)
        return result
