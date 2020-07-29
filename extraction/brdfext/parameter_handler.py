import configparser
import numpy as np
import os
import ast


"""
 © Sebastian Cucerca
"""


class Params:
    
    raw_ext = "nef"
    cam_setup = "d750_105mm"

    light_dist = 300
    cyl_rad = 25.1
    cyl_height = 50
    samp_rad = 25.8
    square_size = 25
    cb_size = [9, 6]

    cam_trans = [0, 0, 0]
    cam_rot = [0, 0, 0]
    ext_height = 30
    nc_fac = True
    nc_filt = 0.01
    corr_fac = 1
    nb_corr = True
    light_pos_samp = 30
    light_pos_precomp = 5000

    patch_thresh_up = [[190, 220, 200], [20, 220, 200]]
    patch_thresh_low = [[160, 100, 50], [0, 100, 50]]
    patch_area = [[700, 5200], [3400, 6032]]
    marker_thresh_up = [[20, 255, 255], [190, 255, 255]]
    marker_thresh_low = [[0, 80, 120], [160, 80, 120]]
    marker_area = [[800, 300], [3300, 1400]]
    marker_pos = 68.0
    marker_length = 7.9
    marker_cnt = 3
    refine_steps = 0
    white_balance = [1.3, 0.5, 0.5, 0.5]

    paper_alb = 0.8
    light_int = 1000
    white_lvl = 15520
    par_comp = True

    c_names = ['red', 'green', 'green', 'blue']
    norm_vars = ['theta_ins', 'theta_outs', 'phi_ins', 'phi_outs']
    

def get_params(params_file):
    """
     TODO
    """

    params = Params()

    if not os.path.isfile(params_file):
        return params

    config = configparser.ConfigParser()
    config.read(params_file)

    sections = config.sections()
    params_section = sections[0]

    for key, val in config.items(params_section):

        key = key.lower()

        if not hasattr(params, key):
            continue

        val = ast.literal_eval(val)
        if isinstance(val, list):
            val = np.array(val)

        setattr(params, key, val)

    return params
