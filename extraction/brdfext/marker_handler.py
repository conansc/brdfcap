from brdfext import plotting_handler
import scipy.spatial.distance as ssd
import numpy as np
import logging
import cv2


"""
 © Sebastian Cucerca
"""


def get_patch_img_pts(img, thresh_low, thresh_up, patch_area):
    """
    Computes points within white patch
    :param img: Calibration image
    :param thresh_low: Lower color thresholds
    :param thresh_up: Upper color thresholds
    :param patch_area: Area in image to consider
    :return: List with image coordinates belonging to white patch
    """

    logging.info("Compute image coordinates for normalization patch.")

    thresh_low = np.array(thresh_low)
    thresh_up = np.array(thresh_up)

    mask, _, _ = threshold(img, thresh_low, thresh_up, patch_area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    assert (contours is not None)
    assert (len(contours) > 0)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_cnt_idx = np.argsort(areas)[::-1][:1]

    h, w, _ = img.shape

    full_img = np.zeros((h, w), np.uint8)
    cv2.drawContours(full_img, contours, max_cnt_idx, color=255, thickness=-1)

    sub_img = np.zeros((h, w), np.uint8)
    cv2.drawContours(sub_img, contours, max_cnt_idx, color=255, thickness=40)
    kernel = np.ones((70, 70), np.uint8)
    sub_img = cv2.dilate(sub_img, kernel, iterations=1)

    in_img = full_img.copy()
    nz_ids = np.transpose(np.nonzero(sub_img))
    for idx in nz_ids:
        in_img[idx[0], idx[1]] = 0

    in_pts = np.transpose(np.nonzero(in_img))

    if __debug__:
        cv2.imwrite('debug/patch_mask.png', mask)
        cv2.imwrite('debug/patch_full.png', full_img)
        cv2.imwrite('debug/patch_sub.png', sub_img)
        cv2.imwrite('debug/patch_in.png', in_img)

        aug_img = plotting_handler.draw_highlighted_points(img.copy(), in_pts, inv=True)
        cv2.imwrite('debug/patch_aug.png', aug_img)

    return in_pts


def get_cylinder_pts(lower_corners, upper_corners, marker_pos, marker_length, cyl_rad):
    """
    Compute corresponding coordinates of image (2D) and object points (3D)
    :param lower_corners: Lower corner points
    :param upper_corners: Upper corner points
    :param marker_pos: Marker position on cylinder
    :param marker_length: Size of one side of marker
    :param cyl_rad: Radius of cylinder (without applied sample)
    :return: List with corresponding image and object points
    """

    logging.info("Compute cylinder center")

    img_pts = []
    obj_pts = []

    pos = marker_pos + marker_length/2
    for i in range(upper_corners.shape[0]):
        img_pts.append(upper_corners[i, :])
        curr_obj_pt = _get_obj_point(i, pos, marker_length, cyl_rad)
        obj_pts.append(curr_obj_pt)

    pos = marker_pos - marker_length / 2
    for i in range(lower_corners.shape[0]):
        img_pts.append(lower_corners[i, :])
        curr_obj_pt = _get_obj_point(i, pos, marker_length, cyl_rad)
        obj_pts.append(curr_obj_pt)

    img_pts = np.float32(img_pts)
    obj_pts = np.float32(obj_pts)

    return [img_pts, obj_pts]


def compute_contours(img, marker_area, marker_cnt, marker_thresh_low, marker_thresh_up):
    """
    Compute contour of markers on cylinder in calibration image
    :param img: Calibration image
    :param marker_area:
    :param marker_cnt: Number of markers
    :param marker_thresh_low: Lower color thresholds
    :param marker_thresh_up: Upper colors thresholds
    :return: Contours of markers in calibration image
    """

    thresh_low = np.array(marker_thresh_low)
    thresh_up = np.array(marker_thresh_up)
    marker_cnt = np.array(marker_cnt)
    marker_area = np.array(marker_area)

    [mask, det_img, hsv_img] = threshold(img, thresh_low, thresh_up, marker_area)

    if __debug__:
        cv2.imwrite('debug/mask.png', mask)
        cv2.imwrite('debug/det.png', det_img)
        cv2.imwrite('debug/hsv.png', hsv_img)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    assert (contours is not None)
    assert (len(contours) >= marker_cnt)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    cnt_ids = np.argsort(areas)[::-1][:marker_cnt]
    contour = np.take(contours, cnt_ids)

    return contour


def compute_corners(contours, img_shape):
    """
    Computes corner points of contours
    :param contours: Input contours
    :param img_shape: Shape of calibration image
    :return: List with sorted upper corner points on cylinder and
                       sorted lower corner points on cylinder
    """

    h, w, _ = img_shape

    upper_cnt_pts = []
    lower_cnt_pts = []

    for idx in range(len(contours)):

        cnt_img = np.zeros((h, w), np.uint8)
        cv2.drawContours(cnt_img, contours, idx, color=255, thickness=-1)

        corner_resp = cv2.cornerHarris(cnt_img, 11, 3, 0.04)
        corner_img = np.zeros((h, w), np.uint8)
        corner_img[corner_resp > 0.2 * corner_resp.max()] = 255

        corner_pts = np.transpose(np.nonzero(corner_img))
        corner_pts = np.stack((corner_pts[:, 1], corner_pts[:, 0]), axis=1)

        p1, p2, p3, p4 = _get_max_dist_pts(corner_pts)

        if p1[0] < p2[0]:
            upper_cnt_pts.append(p1)
            lower_cnt_pts.append(p2)
        else:
            upper_cnt_pts.append(p2)
            lower_cnt_pts.append(p1)

        if p3[0] < p4[0]:
            upper_cnt_pts.append(p3)
            lower_cnt_pts.append(p4)
        else:
            upper_cnt_pts.append(p4)
            lower_cnt_pts.append(p3)

    upper_cnt_pts = np.array(upper_cnt_pts)
    lower_cnt_pts = np.array(lower_cnt_pts)

    sorted_upper_cnt_pts = _sort_pts(upper_cnt_pts)
    sorted_lower_cnt_pts = _sort_pts(lower_cnt_pts)

    return [sorted_upper_cnt_pts, sorted_lower_cnt_pts]


def _get_max_dist_pts(pts):
    """
    Computes four points with highest distance to each other
    :param pts: Input points
    :return: List with four points having maximum distance to each other
    """

    dist_vec = ssd.pdist(pts)
    dist_mat = ssd.squareform(dist_vec)

    dist_max_idx = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    p1_idx = dist_max_idx[0]
    p2_idx = dist_max_idx[1]

    p3_idx, p3_dist = _get_farthermost_pt(dist_mat, [p1_idx, p2_idx])
    p4_idx, p4_dist = _get_farthermost_pt(dist_mat, [p1_idx, p2_idx, p3_idx])

    p1 = pts[p1_idx]
    p2 = pts[p2_idx]
    p3 = pts[p3_idx]
    p4 = pts[p4_idx]

    return [p1, p2, p3, p4]


def _get_farthermost_pt(dist_mat, ids):
    """
    Compute farthermost two points
    :param dist_mat: Distance matrix of points
    :param ids: Indices of points to consider
    :return: List with indices of farthermost points and
                       distance between both points
    """

    full_dists = np.zeros(dist_mat.shape[0])

    for idx in ids:
        curr_dists = dist_mat[:, idx]
        full_dists += curr_dists

    max_idx = np.argmax(full_dists)
    max_val = full_dists[max_idx]

    return [max_idx, max_val]


def _sort_pts(pts):
    """
    Sort points along axis with highest deviation
    :param pts: Input points
    :return: Sorted points
    """

    x_diff = np.max(pts[:, 0]) - np.min(pts[:, 0])
    y_diff = np.max(pts[:, 1]) - np.min(pts[:, 1])
    if x_diff > y_diff:
        sort_ids = np.argsort(pts[:, 0])[::-1]
    else:
        sort_ids = np.argsort(pts[:, 1])[::-1]

    sorted_pts = pts[sort_ids]
    return sorted_pts


def _get_obj_point(idx, y, arc, cyl_rad):
    """
    Get coordinates of point (3D) on cylinder hull
    :param idx: Current step
    :param y: Height
    :param arc: Degrees per step
    :param cyl_rad: Radius of cylinder
    :return: Coordinates of point (3D)
    """

    # Derived from the formula (2*pi) * (arc / (2*pi*cyl_rad))
    rad_step = arc / cyl_rad
    rad_curr = rad_step * idx

    x = np.sin(rad_curr) * cyl_rad
    z = np.cos(rad_curr) * cyl_rad

    return [x, y, z]


def threshold(img, lows, ups, area):
    """
    Thresholds areas in image
    :param img: Calibration image
    :param lows: Lower color thresholds
    :param ups: Upper colors thresholds
    :param area: Areas to threshold in image
    :return: List with threshold mask,
                       image used for detection (considering areas) and
                       detection image in HSV space
    """

    det_img = np.zeros(img.shape, dtype=img.dtype)
    ulc = area[0]
    lrc = area[1]
    det_img[ulc[0]:lrc[0], ulc[1]:lrc[1]] = img[ulc[0]:lrc[0], ulc[1]:lrc[1]]

    hsv_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2HSV)

    mask = np.zeros((hsv_img.shape[0:2]), dtype=np.uint8)
    for curr_tl, curr_tu in zip(lows, ups):
        curr_mask = cv2.inRange(hsv_img, curr_tl, curr_tu)
        mask[curr_mask > 0] = 255

    return [mask, det_img, hsv_img]