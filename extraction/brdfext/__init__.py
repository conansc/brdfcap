from . import parameter_handler
from . import extrinsics_handler
from . import geometry_handler
from . import hdr_handler
from . import intrinsics_handler
from . import io_handler
from . import ldr_handler
from . import marker_handler
from . import plotting_handler
from . import radiometry_handler
from . import raytracing_handler
import numpy as np
import logging
import os

# Define version
__version__ = '1.0.0'
VERISION = __version__


def compute_brdf(samp_path, ref_path, light_path, samp_rot, params_file):

    # Define console output format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(threadName)s %(message)s')

    # Load parameters for current instance of extraction
    params = parameter_handler.get_params(params_file)

    # Compute camera intrinsics
    intrinsics_handler.compute(params.cam_setup, params.raw_ext, params.cb_size, params.square_size)

    # Log
    logging.info("Start processing " + samp_path)

    # Compute light position
    light_pos, light_height = _compute_light_position(light_path, params)

    # Compute value map of white diffuse paper
    ref_brdf_data = _extract_data(ref_path, light_pos, light_height, samp_rot, params)

    # Compute value map of sample
    samp_brdf_data = _extract_data(samp_path, light_pos, light_height, samp_rot, params)

    # Normalize illuminances using diffuse reference values
    ref_cmp_vals = np.transpose(np.vstack([ref_brdf_data[v] for v in params.norm_vars]))
    samp_cmp_vals = np.transpose(np.vstack([samp_brdf_data[v] for v in params.norm_vars]))
    corr_cyl_ills = radiometry_handler.normalize_ills(ref_brdf_data['unnorm_cyl_ills'], ref_cmp_vals,
                                                      ref_brdf_data['cfa'], samp_brdf_data['unnorm_cyl_ills'],
                                                      samp_cmp_vals, samp_brdf_data['cfa'])

    # Compute correction factor from white patches
    patch_fac = radiometry_handler.get_corr_fac(ref_brdf_data['patch_ill'], samp_brdf_data['patch_ill'])

    # Compute absolute BRDF values
    norm_cyl_ills = radiometry_handler.apply_corr_fac(corr_cyl_ills, [params.corr_fac, patch_fac])

    # Add normalized illuminations to BRDF data
    samp_brdf_data['norm_cyl_ills'] = norm_cyl_ills

    # Write BRDF data
    brdf_file_path = os.path.join(samp_path, 'brdf_data.pkl')
    io_handler.save_data(brdf_file_path, samp_brdf_data)

    # Visualize BRDF data
    plotting_handler.plot_brdf(samp_path, "values_unnorm", samp_brdf_data['unnorm_cyl_ills'], samp_brdf_data['theta_ins'],
                               samp_brdf_data['theta_outs'], samp_brdf_data['phi_diffs'], samp_brdf_data['cfa'],
                               ["θi - θo", "Absolute illuminance", samp_path], params.c_names)
    plotting_handler.plot_brdf(samp_path, "values_norm", samp_brdf_data['norm_cyl_ills'], samp_brdf_data['theta_ins'],
                               samp_brdf_data['theta_outs'], samp_brdf_data['phi_diffs'], samp_brdf_data['cfa'],
                               ["θi - θo", "Normalized illuminance", samp_path], params.c_names)


def _compute_light_position(path, params):

    # Try to load precomputed data
    data_file_path = os.path.join(path, 'tmp.pkl')
    data = io_handler.load_data(data_file_path)
    if data is not None:
        return data

    # Load camera intrinsics
    cam_mtx, dist_coeffs = intrinsics_handler.load(params.cam_setup)

    # Load calibration image
    calib_img_path = os.path.join(path, 'calib', '0.' + params.raw_ext)
    calib_img, _ = io_handler.read_raw(calib_img_path, params.white_balance)

    # Compute cylinder extrinsics
    marker_contour = marker_handler.compute_contours(calib_img, params.marker_area, params.marker_cnt,
                                                     params.marker_thresh_low, params.marker_thresh_up)
    upper_corners, lower_corners = marker_handler.compute_corners(marker_contour, calib_img.shape)
    marker_img_pts, marker_obj_pts = marker_handler.get_cylinder_pts(lower_corners, upper_corners, params.marker_pos,
                                                                     params.marker_length, params.cyl_rad)
    cyl_trans, cyl_rot = extrinsics_handler.compute_pose(calib_img, marker_img_pts, marker_obj_pts,
                                                         cam_mtx, dist_coeffs, params.refine_steps)

    # Compute cylinder points
    [cyl_img_pts, cyl_world_pts] = raytracing_handler.compute(calib_img, cyl_trans, cyl_rot, params.cam_trans,
                                                              params.samp_rad, params.cyl_height, cam_mtx, dist_coeffs)

    # Load ldr values of cylinder points
    ldr_vals, exp_times = ldr_handler.get_ldr_vals(path, params.raw_ext, [cyl_img_pts],
                                                   params.white_lvl, params.nb_corr)
    ldr_vals = ldr_vals[0]

    # Filter out brightest points on cylinder
    filter_ids = ldr_handler.get_n_brightest_pts(ldr_vals, params.light_pos_precomp)
    cyl_world_pts = cyl_world_pts[filter_ids]
    ldr_vals = ldr_vals[:, filter_ids]

    # Compute geometric values
    tmp_light_pos = np.array([0, 0, 0])
    [_, _, _, _, _, _, _, _, _, _, _, _, out_vecs, norm_vecs, _,
     height_lengths, _] = geometry_handler.compute_geometric_values(cyl_world_pts, cyl_trans, cyl_rot,
                                                                    params.cam_trans, 0, tmp_light_pos)

    # Compute HDR values
    if params.par_comp:
        cyl_hdr_vals = hdr_handler.compute_hdr_par(ldr_vals, exp_times)
    else:
        cyl_hdr_vals = hdr_handler.compute_hdr_seq(ldr_vals, exp_times)

    # Compute light position
    position, height = geometry_handler.mirror_max_pt(cyl_hdr_vals, cyl_world_pts, norm_vecs, out_vecs,
                                                      height_lengths, params.light_dist, params.samp_rad,
                                                      params.light_pos_samp)

    # Log
    logging.info("Light position [%.2f %.2f %.2f]" % (position[0], position[1], position[2]))
    logging.info("Relative light height %.2f" % height)

    # Compose light data
    light_data = [position, height]

    # Save light data temporarily
    io_handler.save_data(data_file_path, light_data)

    # Return light data
    return light_data


def _extract_data(path, light_trans, light_height, rot, params):

    # Try to load precomputed data
    data_file_path = os.path.join(path, 'tmp.pkl')
    data = io_handler.load_data(data_file_path)
    if data is not None:
        return data

    # Load camera intrinsics
    cam_mtx, dist_coeffs, _ = intrinsics_handler.load(params.cam_setup)

    # Load calibration image
    calib_img_path = os.path.join(path, 'calib', '0.' + params.raw_ext)
    calib_img, cfa_pattern = io_handler.read_raw(calib_img_path, params.white_balance)

    # Compute cylinder extrinsics
    marker_contour = marker_handler.compute_contours(calib_img, params.marker_area, params.marker_cnt,
                                                     params.marker_thresh_low, params.marker_thresh_up)
    upper_corners, lower_corners = marker_handler.compute_corners(marker_contour, calib_img.shape)
    marker_img_pts, marker_obj_pts = marker_handler.get_cylinder_pts(lower_corners, upper_corners,
                                                                     params.marker_pos, params.marker_length,
                                                                     params.cyl_rad)
    cyl_trans, cyl_rot = extrinsics_handler.compute_pose(calib_img, marker_img_pts, marker_obj_pts, cam_mtx,
                                                         dist_coeffs, params.refine_steps)

    # Compute cylinder points
    [cyl_img_pts, cyl_world_pts] = raytracing_handler.compute(calib_img, cyl_trans, cyl_rot, params.cam_trans,
                                                              params.samp_rad, params.cyl_height, cam_mtx, dist_coeffs)

    # Compute geometric values
    [theta_sums, theta_diffs, theta_ins, theta_outs,
     theta_hs, theta_ds, phi_ins, phi_outs, phi_hs, phi_ds,
     phi_diffs, in_vecs, out_vecs, norm_vecs, half_vecs,
     height_lengths, tang_vecs] = geometry_handler.compute_geometric_values(cyl_world_pts, cyl_trans, cyl_rot,
                                                                            params.cam_trans, rot, light_trans)

    # Filter points outside cylinder
    geo_mask = geometry_handler.get_filter(theta_sums, theta_diffs, height_lengths, light_height,
                                           params.ext_height, params.nc_filt, params.nc_fac)

    # Compute normlization patch points
    patch_img_pts = marker_handler.get_patch_img_pts(calib_img, params.patch_thresh_low,
                                                     params.patch_thresh_up, params.patch_area)

    # Load LDR values
    comb_img_pts = [cyl_img_pts[geo_mask, :], patch_img_pts]
    ldr_vals, exp_times = ldr_handler.get_ldr_vals(path, params.raw_ext, comb_img_pts, params.white_lvl, params.nb_corr)
    cyl_ldr_vals = ldr_vals[0]
    patch_ldr_vals = ldr_vals[1]

    # Compute HDR values
    if params.par_comp:
        cyl_hdr_vals = hdr_handler.compute_hdr_par(cyl_ldr_vals, exp_times)
        patch_hdr_vals = hdr_handler.compute_hdr_par(patch_ldr_vals, exp_times)
    else:
        cyl_hdr_vals = hdr_handler.compute_hdr_seq(cyl_ldr_vals, exp_times)
        patch_hdr_vals = hdr_handler.compute_hdr_seq(patch_ldr_vals, exp_times)

    # Compute patch average
    patch_ill = np.mean(patch_hdr_vals)

    # Combine data
    brdf_data = {
        'light_trans': light_trans,
        'samp_rot': rot,
        'cam_trans': params.cam_trans,
        'cam_rot': params.cam_rot,
        'calib_img': calib_img,
        'cyl_trans': cyl_trans,
        'cyl_rot': cyl_rot,
        'img_pts': cyl_img_pts[geo_mask, :],
        'cfa': cfa_pattern[cyl_img_pts[geo_mask, 0], cyl_img_pts[geo_mask, 1]],
        'unnorm_cyl_ills': cyl_hdr_vals,
        'patch_ill': patch_ill,
        'theta_sums': theta_sums[geo_mask],
        'theta_diffs': theta_diffs[geo_mask],
        'theta_ins': theta_ins[geo_mask],
        'theta_outs': theta_outs[geo_mask],
        'theta_hs': theta_hs[geo_mask],
        'theta_ds': theta_ds[geo_mask],
        'phi_ins': phi_ins[geo_mask],
        'phi_outs': phi_outs[geo_mask],
        'phi_hs': phi_hs[geo_mask],
        'phi_ds': phi_ds[geo_mask],
        'phi_diffs': phi_diffs[geo_mask],
        'in_vecs': in_vecs[geo_mask, :],
        'out_vecs': out_vecs[geo_mask, :],
        'norm_vecs': norm_vecs[geo_mask, :],
        'half_vecs': half_vecs[geo_mask, :],
        'height_lengths': height_lengths[geo_mask],
        'tang_vecs': tang_vecs[geo_mask, :]
    }

    # Save computed data
    io_handler.save_data(data_file_path, brdf_data)

    # Return data
    return brdf_data
