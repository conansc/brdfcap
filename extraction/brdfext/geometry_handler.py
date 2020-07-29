import numpy as np
import logging
import math
import cv2


"""
 © Sebastian Cucerca
"""


def compute_geometric_values(world_pts, cyl_trans, cyl_rot, cam_trans, ori, light_trans):
    """
     TODO
    """

    logging.info("Computing geometric values")

    # Get rotation matrix for cylinder
    rot_mat = np.array(cv2.Rodrigues(cyl_rot)[0])

    # Define two points on cylinder axis in local cylinder space
    local_upper_pt = np.array([0, 1, 0])
    local_lower_pt = np.array([0, 0, 0])

    # Translate axis points to world space
    world_upper_pt = rot_mat.dot(local_upper_pt) + cyl_trans
    world_lower_pt = rot_mat.dot(local_lower_pt) + cyl_trans

    # Define cylinder axis in world coordinate space using two axis points
    world_axis = _normalize_vector(world_upper_pt - world_lower_pt)

    # Compute for every hit point height on cylinder axis
    world_pt_vecs = world_pts - world_lower_pt
    height_lengths = np.dot(world_pt_vecs, world_axis)

    # Compute for every hit point exact position on cylinder axis
    tiled_world_axis = np.tile(world_axis, (height_lengths.shape[0], 1))
    height_vecs = tiled_world_axis * np.expand_dims(height_lengths, axis=1)
    world_slice_centers = height_vecs + world_lower_pt

    # Compute normal vectors for every hit point
    norm_vecs = world_pts - world_slice_centers
    norm_vecs = _normalize_vectors(norm_vecs)

    # Compute the general tangent vector
    gen_tang_vecs = np.cross(tiled_world_axis, norm_vecs)

    # Rotate the tangent vector according to the rotation of the sample
    tang_vecs = _rotate_around_axis(norm_vecs, gen_tang_vecs, ori)

    # Compute outgoing vectors (to camera)
    out_vecs = -world_pts + cam_trans
    out_vecs = _normalize_vectors(out_vecs)

    # Compute incoming vectors (to light)
    in_vecs = -world_pts + light_trans
    in_vecs = _normalize_vectors(in_vecs)

    # Compute half vectors
    half_vecs = in_vecs + out_vecs
    half_vecs *= 0.5
    half_vecs = _normalize_vectors(half_vecs)

    # Compute theta angles
    theta_ins = _angles_between(in_vecs, norm_vecs)
    theta_outs = _angles_between(out_vecs, norm_vecs)
    theta_hs = _angles_between(norm_vecs, half_vecs)
    theta_ds = _angles_between(half_vecs, in_vecs)

    # Compute sum of angles between (in vector <-> normal) and (out vector <-> normal)
    theta_sum = theta_ins + theta_outs

    # Compute full angle between in/out vectors
    theta_diffs = _angles_between(in_vecs, out_vecs)

    # Project vectors on plane with same normal as cylinder hit points
    proj_tang_ns = _project_multiple(tang_vecs, norm_vecs)
    proj_out_ns = _project_multiple(out_vecs, norm_vecs)
    proj_in_ns = _project_multiple(in_vecs, norm_vecs)
    proj_half_ns = _project_multiple(half_vecs, norm_vecs)
    proj_in_hs = _project_multiple(in_vecs, half_vecs)
    bin_vecs = np.cross(norm_vecs, proj_tang_ns)
    proj_bin_hs = _project_multiple(bin_vecs, half_vecs)

    # Compute phi angles
    phi_ins = _angles_between(proj_in_ns, proj_tang_ns, norm_vecs)
    phi_outs = _angles_between(proj_out_ns, proj_tang_ns, norm_vecs)
    phi_hs = _angles_between(proj_half_ns, proj_tang_ns, norm_vecs)
    phi_ds = _angles_between(proj_bin_hs, proj_in_hs, half_vecs)
    phi_diffs = _angles_between(proj_out_ns, proj_in_ns, norm_vecs)

    # Generate value map with all relevant information
    return [theta_sum, theta_diffs, theta_ins, theta_outs, theta_hs, theta_ds, phi_ins, phi_outs, phi_hs, phi_ds,
            phi_diffs, in_vecs, out_vecs, norm_vecs, half_vecs, height_lengths, tang_vecs]


def mirror_max_pt(ills, world_object_points, norm_vecs, out_vecs, heights, light_dist, samp_rad, samp_rate):

    logging.info("Computing light position.")

    max_ids = np.argsort(ills)[-samp_rate:]

    world_object_points = world_object_points[max_ids]
    world_object_point = np.average(world_object_points, axis=0)

    norm_vecs = norm_vecs[max_ids]
    norm_vec = np.average(norm_vecs, axis=0)

    out_vecs = out_vecs[max_ids]
    out_vec = np.average(out_vecs, axis=0)

    in_vec = 2 * np.dot(norm_vec, out_vec) * norm_vec - out_vec
    in_vec = _normalize_vector(in_vec)

    position = world_object_point + in_vec * (light_dist - samp_rad)
    heights = heights[max_ids]
    height = np.average(heights, axis=0)

    return position, height


def _normalize_vector(vec):
    """
     TODO
    """
    denom = np.linalg.norm(vec)
    if denom == 0:
        return np.float32(vec * 0.0)
    return np.float32(vec / denom)


def _normalize_vectors(vecs, dim=1):
    """
     TODO
    """
    norms = np.linalg.norm(vecs, axis=dim, keepdims=True)
    norms[norms == 0] = 1
    normed = vecs / norms
    return normed


def _project_multiple(vecs, norms):
    """
     TODO
    """

    proj = _row_wise_dot(vecs, norms)
    proj = norms * np.expand_dims(proj, axis=1)
    proj = -proj + vecs
    proj = _normalize_vectors(proj)

    return proj


def _angles_between(vecs1, vecs2, norm_vecs=None):
    """
     TODO
    """

    angs = _row_wise_dot(vecs1, vecs2)
    angs = np.clip(angs, -1.0, 1.0)
    angs = np.arccos(angs)

    if norm_vecs is not None:
        tp_vecs = np.cross(vecs1, vecs2)
        tp_inds = _row_wise_dot(norm_vecs, tp_vecs)
        angs[tp_inds < 0] = (2 * np.pi) - angs[tp_inds < 0]

    return angs


def _rotate_around_axis(axes, vecs, deg):
    """
     TODO
    """

    vecs = np.asarray(vecs)
    axes = np.asarray(axes)

    assert(axes.shape == vecs.shape)

    deg = float(deg)
    theta = np.radians(deg)
    axes = _normalize_vectors(axes)

    a = math.cos(theta / 2.0)
    a_s = np.repeat(a, axes.shape[0])
    tmp = -axes * math.sin(theta / 2.0)
    b_s, c_s, d_s = tmp[:, 0], tmp[:, 1], tmp[:, 2]

    aas, bbs, ccs, dds = a_s * a_s, b_s * b_s, c_s * c_s, d_s * d_s
    bcs, ads, acs, abs, bds, cds = b_s * c_s, a_s * d_s, a_s * c_s, a_s * b_s, b_s * d_s, c_s * d_s

    rot_mats = np.array([[aas + bbs - ccs - dds, 2 * (bcs + ads), 2 * (bds - acs)],
                     [2 * (bcs - ads), aas + ccs - bbs - dds, 2 * (cds + abs)],
                     [2 * (bds + acs), 2 * (cds - abs), aas + dds - bbs - ccs]])
    rot_mats = np.swapaxes(rot_mats, 0, 2)
    rot_mats = np.swapaxes(rot_mats, 1, 2)

    vecs = np.expand_dims(vecs, 2)
    rot_vecs = np.matmul(rot_mats, vecs)
    rot_vecs = np.squeeze(rot_vecs)

    return rot_vecs


def _row_wise_dot(vecs1, vecs2):
    """
     TODO
    """
    return np.sum(vecs1 * vecs2, axis=1)


def get_filter(theta_sum, theta_diffs, height_lengths, light_height, ext_height, nc_filt, nc_fac):
    """
     Filter out hit points that are too far away from slice where light hits cylinder
    """

    #
    mask_tmp1 = height_lengths > (light_height - ext_height / 2)
    mask_tmp2 = height_lengths < (light_height + ext_height / 2)
    val_mask = np.logical_and(mask_tmp1, mask_tmp2)

    #
    if nc_filt:
        mask_tmp3 = theta_sum <= (theta_diffs * (1 + nc_fac))
        val_mask = np.logical_and(val_mask, mask_tmp3)

    #
    return val_mask


