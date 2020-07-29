from contextlib import closing
import multiprocessing as mp
import numpy as np
import logging
import ctypes


"""
 © Sebastian Cucerca
"""


def _to_np_arr(buff):
    """
    Converts a buffer to a 1-dimensional array
    :param mp_arr: Buffer to be converted
    :return: Converted array
    """
    return np.frombuffer(buff.get_obj())


def _init_pool(shared_ldr_vals_, shared_ldr_vals_shape_, shared_nzno_ldr_, shared_nzno_ldr_shape_,
               shared_exp_times_, shared_regs_, shared_regs_shape_):
    """
    Initializes global variables for multiprocessing
    :param shared_ldr_vals_: LDR values
    :param shared_ldr_vals_shape_: Shape of LDR values
    :param shared_nzno_ldr_: Non-zero non-one values
    :param shared_nzno_ldr_shape_: Shape of non-zero non-one values values
    :param shared_exp_times_: Exposure times
    :param shared_regs_: Regression parameters
    :param shared_regs_shape_: Shape of regression parameters
    :return:
    """

    global shared_ldr_vals
    global shared_ldr_vals_shape
    global shared_nzno_ldr
    global shared_nzno_ldr_shape
    global shared_exp_times
    global shared_regs
    global shared_regs_shape
    shared_ldr_vals = shared_ldr_vals_
    shared_ldr_vals_shape = shared_ldr_vals_shape_
    shared_nzno_ldr = shared_nzno_ldr_
    shared_nzno_ldr_shape = shared_nzno_ldr_shape_
    shared_exp_times = shared_exp_times_
    shared_regs = shared_regs_
    shared_regs_shape = shared_regs_shape_


def compute_hdr_par(ldr_vals, exp_times):
    """
    Computes HDR values with parallel computing
    :param ldr_vals: LDR values
    :param exp_times: Exposure times of LDR values
    :return: HDR values
    """

    logging.info("Computing HDR")

    nzno_ldr = np.zeros(ldr_vals.shape, dtype=bool)
    nzno_ldr[np.where((0 < ldr_vals) & (ldr_vals < 1))] = 1
    n = np.unique(nzno_ldr, axis=1)
    n = n == 1
    n = np.split(n, n.shape[1], axis=1)
    regs = np.zeros((ldr_vals.shape[1], 2))

    shared_ldr_vals = mp.Array(ctypes.c_double, np.size(ldr_vals))
    np_shared_filt_ldr_img = _to_np_arr(shared_ldr_vals)
    np_shared_filt_ldr_img[:] = ldr_vals.flatten()

    shared_ldr_vals_shape = mp.Array(ctypes.c_double, np.size(ldr_vals.shape))
    np_shared_filt_ldr_img_shape = _to_np_arr(shared_ldr_vals_shape)
    np_shared_filt_ldr_img_shape[:] = ldr_vals.shape

    shared_nzno_ldr = mp.Array(ctypes.c_double, np.size(nzno_ldr))
    np_shared_nzno_ldr = _to_np_arr(shared_nzno_ldr)
    np_shared_nzno_ldr[:] = nzno_ldr.flatten()

    shared_nzno_ldr_shape = mp.Array(ctypes.c_double, np.size(nzno_ldr.shape))
    np_shared_nzno_ldr_shape = _to_np_arr(shared_nzno_ldr_shape)
    np_shared_nzno_ldr_shape[:] = nzno_ldr.shape

    shared_exp_times = mp.Array(ctypes.c_double, np.size(exp_times))
    np_shared_exp_times = _to_np_arr(shared_exp_times)
    np_shared_exp_times[:] = exp_times

    shared_regs = mp.Array(ctypes.c_double, np.size(regs))
    np_shared_regs = _to_np_arr(shared_regs)
    np_shared_regs[:] = regs.flatten()

    shared_regs_shape = mp.Array(ctypes.c_double, np.size(regs.shape))
    np_shared_regs_shape = _to_np_arr(shared_regs_shape)
    np_shared_regs_shape[:] = regs.shape

    cpu_cnt = int(mp.cpu_count()*0.8)
    with closing(mp.Pool(cpu_cnt, initializer=_init_pool, initargs=(shared_ldr_vals, shared_ldr_vals_shape, shared_nzno_ldr, shared_nzno_ldr_shape, shared_exp_times, shared_regs, shared_regs_shape, ))) as pool:
        pool.map_async(_regression, n)

    pool.join()

    regs = np.reshape(np_shared_regs, regs.shape)
    hdr_vals = np.dot(regs, np.array([1, 1]))

    return hdr_vals


def _regression(n):
    """
    Computes regression for LDR values and samples fitted line at values 1
    :param n: Indices of LDR values to compute
    :return: None
    """

    ldr_vals = _to_np_arr(shared_ldr_vals)
    ldr_vals_shape = np.int32(_to_np_arr(shared_ldr_vals_shape))
    ldr_vals = np.reshape(ldr_vals, ldr_vals_shape)

    nzno_ldr = _to_np_arr(shared_nzno_ldr)
    nzno_ldr_shape = np.int32(_to_np_arr(shared_nzno_ldr_shape))
    nzno_ldr = np.reshape(nzno_ldr, nzno_ldr_shape)

    exp_times = _to_np_arr(shared_exp_times)

    regs = _to_np_arr(shared_regs)
    regs_shape = np.int32(_to_np_arr(shared_regs_shape))
    regs = np.reshape(regs, regs_shape)

    n = np.squeeze(n)

    if np.sum(n) <= 3:
        return

    ldr_mask = np.ones(nzno_ldr.shape[1], dtype=bool)
    for j in range(n.shape[0]):
        ldr_mask = ldr_mask & (nzno_ldr[j, :] == n[j])

    cols = ldr_vals[:, ldr_mask]
    X = exp_times[n]
    Y = np.float32(cols[n, :])
    reg = np.polyfit(X, Y, 1)
    regs[ldr_mask, 0] = reg[0, :]
    regs[ldr_mask, 1] = reg[1, :]


def compute_hdr_seq(ldr_vals, exp_times):
    """
    Computes HDR values sequentially
    :param ldr_vals: LDR values
    :param exp_times: Exposure times of LDR values
    :return: HDR values
    """

    logging.info("Computing HDR")

    regs = np.zeros((ldr_vals.shape[1], 2))

    nzno_ldr = np.zeros(ldr_vals.shape, dtype=bool)
    nzno_ldr[np.where((0 < ldr_vals) & (ldr_vals < 1))] = 1
    n = np.unique(nzno_ldr, axis=1)
    n = n == 1

    for i in range(n.shape[1]):

        if np.sum(n[:, i]) <= 3:
            continue

        ldr_mask = np.ones(nzno_ldr.shape[1], dtype=bool)
        for j in range(n.shape[0]):
            ldr_mask = ldr_mask & (nzno_ldr[j, :] == n[j, i])

        cols = ldr_vals[:, ldr_mask]
        X = exp_times[n[:, i]]
        Y = np.float32(cols[n[:, i], :])
        reg = np.polyfit(X, Y, 1)
        regs[ldr_mask, 0] = reg[0, :]
        regs[ldr_mask, 1] = reg[1, :]

    hdr_vals = np.dot(regs, np.array([1, 1]))

    return hdr_vals

