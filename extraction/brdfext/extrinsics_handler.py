from brdfext import plotting_handler
import numpy as np
import logging
import cv2


"""
 © Sebastian Cucerca
"""


def compute_pose(img, img_pts, obj_pts, cam_mtx, dist_coeffs, refine_steps):
    """
     TODO
    """

    for i in range(refine_steps + 1):
        _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mtx, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        proj_img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist_coeffs)

        error_idx = _get_max_error_idx(img_pts, proj_img_pts)
        _ = _compute_error(img_pts, proj_img_pts)

        img_pts = np.delete(img_pts, error_idx, axis=0)
        obj_pts = np.delete(obj_pts, error_idx, axis=0)

    tvec = np.float32(np.squeeze(tvec).transpose())
    rvec = np.float32(np.squeeze(rvec).transpose())

    logging.info("Tranlation vector [%.2f, %.2f, %.2f]" % (tvec[0], tvec[1], tvec[2]))

    if __debug__:
        aug_img = plotting_handler.draw_highlighted_points(img.copy(), img_pts, [255, 0, 0], 4)
        aug_img = plotting_handler.draw_highlighted_points(aug_img, proj_img_pts, [0, 255, 0], 4)
        cv2.imwrite('debug/geometry.png', aug_img)

    return [tvec, rvec]


def _get_max_error_idx(img_pts, proj_img_pts):
    """
    Finds the corner
    :param img_pts:
    :param proj_img_pts:
    :return: Index of corner point with highest error
    """

    max_error = None
    max_error_idx = None

    assert (len(img_pts) == len(proj_img_pts))

    for i in range(len(img_pts)):
        curr_error = np.linalg.norm(img_pts[i] - proj_img_pts[i])
        if max_error is None or curr_error > max_error:
            max_error = curr_error
            max_error_idx = i

    return max_error_idx


def _compute_error(img_pts, proj_img_pts):
    """
    Computes reprojection error for corner points.
    :param img_pts: Actual reference image points (2D) for corners
    :param proj_img_pts: Reprojected image points (2D) for corners
    :return: Reprojection error value
    """

    error = 0.0
    for img_pt, proj_img_pt in zip(img_pts, proj_img_pts):
        curr_error = np.linalg.norm(img_pt - proj_img_pt)
        error += curr_error
    error /= len(img_pts)

    logging.info("Reprojection error %.2f" % error)

    return error






