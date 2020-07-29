from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging


"""
 © Sebastian Cucerca
"""


def get_corr_fac(ref_mean, samp_mean):
    """
    Computes correction factor between two white patches
    :param ref_mean: Value of reference white patch
    :param samp_mean: Value of sample white patch
    :return: Correction factor
    """

    if ref_mean is None:
        ref_mean = samp_mean
    corr_fac = ref_mean / samp_mean
    logging.info("Correction factor is %.3f(%.5f, %.5f)" % (corr_fac, samp_mean, ref_mean))
    return corr_fac


def apply_corr_fac(vals, facs):
    """
    Multiplies illumination values by multiple factors
    :param vals: Illumination values
    :param facs: Factors
    :return: Corrected illumination values
    """

    corr_vals = vals.copy()
    for fac in facs:
        corr_vals *= fac
    return corr_vals


def normalize_ills(ref_unnorm_ills, ref_cmp_vals, ref_cfa, samp_unnorm_ills, samp_cmp_vals, samp_cfa):
    """
    Normalize sample illumination values by dividing by reference illumination values
    with similar geometric properties
    :param ref_unnorm_ills: Unnormalized reference illumination values
    :param ref_cmp_vals: Geometric properties of reference
    :param ref_cfa: CFA values of reference
    :param samp_unnorm_ills: Unnormalized sample illumination values
    :param samp_cmp_vals: Geometric properties of sample
    :param samp_cfa: CFA values of sample
    :return: Normalized sample illuminations
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
