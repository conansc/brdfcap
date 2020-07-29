import logging
import rawpy
import numpy as np
import exifread
import glob
import os


"""
 © Sebastian Cucerca
"""


def get_ldr_vals(path, raw_ext, pts_ids_list, white_lvl, nb_corr):
    """
    TODO
    """

    #
    file_cnt = len(glob.glob(os.path.join(path, "0", "*." + raw_ext)))
    folder_cnt = glob.glob(path + "/*/")
    folder_cnt = [f for f in folder_cnt if os.path.isdir(f)]
    folder_cnt = len(folder_cnt) - 1

    #
    ldr_vals_list = []
    for pts_ids in pts_ids_list:
        ldr_vals_list.append(np.zeros((file_cnt, pts_ids.shape[0]), dtype=np.float16))

    #
    exp_times = np.zeros(file_cnt, dtype=np.float16)

    #
    logging.info("Read in LDR images")

    #
    for j in range(folder_cnt):
        for i in range(file_cnt):

            src_img_path = os.path.join(path, str(j), str(i) + "." + raw_ext)

            meta_data = exifread.process_file(open(src_img_path, 'rb'))
            exp_time = meta_data["EXIF ExposureTime"].values[0]
            exp_times[i] = float(exp_time.num / exp_time.den)

            raw_data = rawpy.imread(src_img_path)
            cfa = raw_data.raw_image_visible
            cfa = np.float16(cfa)

            black_lvls = raw_data.black_level_per_channel
            for b in range(len(black_lvls)):
                cfa[raw_data.raw_colors_visible == b] -= black_lvls[b]
                cfa[raw_data.raw_colors_visible == b] /= white_lvl - black_lvls[b]

            cfa = np.clip(cfa, 0, 1)

            for k in range(len(pts_ids_list)):
                ldr_vals_list[k][i, :] = ldr_vals_list[k][i, :] + cfa[pts_ids_list[k][:, 0], pts_ids_list[k][:, 1]]

    #
    for i in range(len(ldr_vals_list)):

        #
        ldr_vals_list[i] = ldr_vals_list[i] / folder_cnt

        #
        if nb_corr:
            ldr_vals_list[i] = _correct_ldr_vals(ldr_vals_list[i], exp_times)

    #
    return [ldr_vals_list, exp_times]


def _correct_pair(ldr_vals, exp_times, i, j, pix_thresh):
    """
    TODO
    """

    prev_exp = exp_times[i]
    prev_vals = ldr_vals[i]

    next_exp = exp_times[j]
    next_vals = ldr_vals[j]

    mask1 = np.where(next_vals > pix_thresh[0])
    mask2 = np.where(next_vals < pix_thresh[1])
    mask = np.intersect1d(mask1, mask2)

    if len(mask) == 0 or len(mask) < 100:
        logging.debug("Pair (" + str(i) + "," + str(j) + "): Not enough pixels")
        return ldr_vals

    prev_sel = prev_vals[mask]
    next_sel = next_vals[mask]
    means = prev_sel / next_sel
    actual_mean = np.mean(means)
    target_mean = prev_exp / next_exp
    corr_fac = actual_mean / target_mean

    curr_vals = ldr_vals[j]
    corr_mask = curr_vals < 1
    curr_vals[corr_mask] *= corr_fac
    curr_vals = np.clip(curr_vals, 0, 1)
    ldr_vals[j] = curr_vals

    logging.debug("Pair (" + str(i) + "," + str(j) + "): " +
                   "Actual mean " + str(np.round(actual_mean, 3)) +
                   ", Target mean " + str(np.round(target_mean, 3)) +
                   ", Correction factor " + str(np.round(corr_fac, 3)) +
                   ", Pixel count " + str(len(means)))

    return ldr_vals


def _correct_ldr_vals(ldr_vals, exp_times):
    """
    TODO
    """

    logging.info("Correct LDR images")

    img_cnt = exp_times.shape[0]
    mid = int(img_cnt / 2)

    for i in range(mid, img_cnt - 1, 1):
        ldr_vals = _correct_pair(ldr_vals, exp_times, i, i + 1, [0.1, 0.8])

    for i in range(mid, 0, -1):
        ldr_vals = _correct_pair(ldr_vals, exp_times, i, i - 1, [0.1, 0.49])

    return ldr_vals


def get_n_brightest_pts(ldr_vals, samp_rate):
    """
    TODO
    """

    n = samp_rate
    d = ldr_vals.shape[0]
    filter_ids = np.array([], dtype=np.int)

    for l in range(d):

        curr_vals = ldr_vals[l, :]
        gz_ids = np.where(curr_vals > 0)
        curr_filt_ids = np.setdiff1d(gz_ids, filter_ids)

        if (len(filter_ids) + len(curr_filt_ids)) < n:
            filter_ids = np.concatenate((filter_ids, curr_filt_ids))
        else:
            curr_filt_ill = curr_vals[curr_filt_ids]
            miss_cnt = n - len(filter_ids)
            max_ids = np.argsort(curr_filt_ill)[-miss_cnt:]
            max_ids = curr_filt_ids[max_ids]
            filter_ids = np.concatenate((filter_ids, max_ids))
            break

    if n < len(filter_ids):
        logging.error("Not enough points could be extracted from the LDR images for computing light position.")

    return filter_ids
