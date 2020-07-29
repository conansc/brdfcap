from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging


"""
 © Sebastian Cucerca
"""


def get_patch_factor(ref_mean, samp_mean):
    """
     TODO
    """

    if ref_mean is None:
        ref_mean = samp_mean
    norm_fac = ref_mean / samp_mean
    logging.info("Norm factor is %.3f(%.5f, %.5f)" % (norm_fac, samp_mean, ref_mean))
    return norm_fac


def apply_norm_factor(vals, facs):
    """
     TODO
    """

    new_vals = vals.copy()
    for fac in facs:
        new_vals *= fac
    return new_vals


def normalize_ills(ref_unnorm_ills, ref_cmp_vals, ref_cfa, samp_unnorm_ills, samp_cmp_vals, samp_cfa):
    """
     TODO
    """

    logging.info("Normalizing values")

    samp_norm_ills = np.zeros(samp_unnorm_ills.shape)

    for p in range(4):

        curr_ref_unnorm_ills = ref_unnorm_ills[ref_cfa == p]
        curr_samp_unnorm_ills = samp_unnorm_ills[samp_cfa == p]

        if curr_ref_unnorm_ills.shape[0] == 0 or curr_samp_unnorm_ills.shape[0] == 0:
            continue

        curr_ref_cmps = ref_cmp_vals[ref_cfa == p]
        curr_samp_cmps = samp_cmp_vals[samp_cfa == p]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(curr_ref_cmps)
        distances, indices = nbrs.kneighbors(curr_samp_cmps)

        divisors = curr_ref_unnorm_ills[indices]
        divisors = np.squeeze(divisors)
        divisors[divisors == 0] = 1
        quotients = curr_samp_unnorm_ills / divisors

        samp_norm_ills[samp_cfa == p] = quotients

    return samp_norm_ills
