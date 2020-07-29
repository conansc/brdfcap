import logging
import numpy as np
import cv2


"""
 © Sebastian Cucerca
"""


def compute(img, cyl_trans, cyl_rot, cam_trans, samp_rad, cyl_height, cam_mtx, dist_coeffs):
    """
     TODO
    """

    logging.info("Computing world points")

    # Compute (inverse) rotation matrix
    rot_mat = np.float32(cv2.Rodrigues(cyl_rot)[0])
    inv_rot_mat = np.linalg.inv(rot_mat)

    # Compute ray origin
    ray_o = np.float32(cam_trans)
    trans_ray_o = inv_rot_mat.dot(ray_o - cyl_trans)

    # Compute image points
    [h, w, _] = img.shape
    pts = np.where(np.full((h, w), True, dtype=bool))
    pts = np.transpose(pts)
    pts = np.flip(pts, axis=1)
    pts = np.float32(pts)

    # Compute ray direction
    pts_cnt = pts.shape[0]
    trans_ray_ds = np.reshape(pts, (pts_cnt, 1, 2))
    trans_ray_ds = cv2.undistortPoints(trans_ray_ds, cam_mtx, dist_coeffs)
    trans_ray_ds = np.squeeze(trans_ray_ds)
    trans_ray_ds = np.hstack((trans_ray_ds, np.ones((pts_cnt, 1), dtype=np.float32)))

    trans_ray_ds = trans_ray_ds - cyl_trans
    trans_ray_ds = np.transpose(trans_ray_ds)
    trans_ray_ds = inv_rot_mat.dot(trans_ray_ds)
    trans_ray_ds = np.transpose(trans_ray_ds)

    trans_ray_ds = trans_ray_ds - trans_ray_o
    trans_ray_ds = trans_ray_ds / np.linalg.norm(trans_ray_ds, axis=1, keepdims=True)

    a_s = trans_ray_ds[:, 0] * trans_ray_ds[:, 0] + trans_ray_ds[:, 2] * trans_ray_ds[:, 2]
    b_s = 2 * (trans_ray_o[0] * trans_ray_ds[:, 0] + trans_ray_o[2] * trans_ray_ds[:, 2])
    c_s = trans_ray_o[0] * trans_ray_o[0] + trans_ray_o[2] * trans_ray_o[2] - samp_rad * samp_rad

    inners = b_s * b_s - 4 * a_s * c_s
    ids = inners > 0
    inners = inners[ids]

    a_s = a_s[ids]
    b_s = b_s[ids]
    pts = pts[ids]
    trans_ray_ds = trans_ray_ds[ids]

    inners = np.sqrt(inners)
    ts = np.minimum((-b_s + inners) / (2 * a_s), (-b_s - inners) / (2 * a_s))

    ids = ts > 0
    ts = ts[ids]
    pts = pts[ids]
    trans_ray_ds = trans_ray_ds[ids]

    trans_ray_ds = trans_ray_ds * np.expand_dims(ts, axis=1)
    trans_ray_ds = trans_ray_ds + trans_ray_o

    ids = np.intersect1d(np.where(trans_ray_ds[:, 1] > (-cyl_height/2)),
                         np.where(trans_ray_ds[:, 1] < (cyl_height/2)))

    trans_ray_ds = trans_ray_ds[ids]
    pts = pts[ids]
    pts = np.flip(pts, axis=1)
    pts = np.int32(pts)

    trans_ray_ds = rot_mat.dot(np.transpose(trans_ray_ds))
    trans_ray_ds = np.transpose(trans_ray_ds)
    trans_ray_ds = trans_ray_ds + cyl_trans

    # Visualize raytracing results and store to image
    if __debug__:
        sampled_img = img.copy()
        color = np.array([0, 0, 255])
        sampled_img[pts[:, 0], pts[:, 1]] = color
        cv2.imwrite('debug/intesections.png', sampled_img)

    #
    return [pts, trans_ray_ds]
