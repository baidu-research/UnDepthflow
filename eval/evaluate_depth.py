from evaluation_utils import *


# Adopted from https://github.com/mrharicot/monodepth
def load_depths(pred_disp_org, gt_path, eval_occ):
    gt_disparities = load_gt_disp_kitti(gt_path, eval_occ)
    gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(
        gt_disparities, pred_disp_org)

    return gt_depths, pred_depths, gt_disparities, pred_disparities_resized


def process_depth(gt_depth, pred_depth, gt_disp, min_depth, max_depth):
    mask = gt_disp > 0
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    gt_depth[gt_depth < min_depth] = min_depth
    gt_depth[gt_depth > max_depth] = max_depth

    return gt_depth, pred_depth, mask


def eval_depth(gt_depths,
               pred_depths,
               gt_disparities,
               pred_disparities_resized,
               min_depth=1e-3,
               max_depth=80):
    num_samples = len(pred_depths)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        gt_depth, pred_depth, mask = process_depth(
            gt_depth, pred_depth, gt_disparities[i], min_depth, max_depth)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[
            i] = compute_errors(gt_depth[mask], pred_depth[mask])

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        pred_disp = pred_disparities_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3,
                                    (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(
    ), a2.mean(), a3.mean(), d1_all.mean()
