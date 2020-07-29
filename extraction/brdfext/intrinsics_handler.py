import logging
import cv2
import rawpy
import numpy as np
import os


"""
 © Sebastian Cucerca
"""


def get_cam_params(camera_setup, img_ext, checkerboard_size, square_size):
    """
    TODO
    """

    logging.info("Loading camera matrices.")

    intr_file_path = os.path.join('calibration', camera_setup, 'cam_params.npy')

    if not os.path.isfile(intr_file_path):
        intr_folder_path = os.path.join('calibration', camera_setup)
        [img_paths, _, _] = _get_img_paths(intr_folder_path, img_ext)
        [mtx, dist, new_mtx] = _calibrate(img_paths, checkerboard_size, square_size)
        _write_cam_params('calibration', camera_setup, mtx, dist, new_mtx)

    cam_params = np.load(intr_file_path, allow_pickle=True)
    return cam_params


def _calibrate(img_paths, checkerboard_size, square_size):
    """
    TODO
    """

    logging.info("Calibrating device.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cols = checkerboard_size[0]
    rows = checkerboard_size[1]

    obj_p = np.zeros((rows * cols, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_p *= square_size

    obj_points = []
    img_points = []

    for img_path, idx in zip(img_paths, range(len(img_paths))):

        raw = rawpy.imread(img_path)
        img = raw.postprocess(user_flip=0)
        assert(img is not None)

        logging.info("Searching for checkerboard in image " + str(idx+1) + " out of " + str(len(img_paths)) + ".")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            obj_points.append(obj_p)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)

            #cv2.imwrite(os.path.join('calibration', str(idx) + '.png'), img)

        else:
            logging.warning("Could not find chessboard in image.")

    cv2.destroyAllWindows()

    logging.info("Computing camera parameters.")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None,
                                                       flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    tot_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error
    mean_error = tot_error / len(obj_points)

    logging.info("Average calibration error is " + str(mean_error) + " pixels.")
    logging.info("Successfully calibrated device.")

    return [mtx, dist, new_mtx]


def _write_cam_params(intr_folder, camera_setup, mtx, dist, new_mtx):
    """
    TODO
    """

    logging.info("Writing parameters file.")
    intr_path = os.path.join(intr_folder, camera_setup, intr_folder)
    np.save(intr_path, [mtx, dist, new_mtx])


def _get_img_paths(path, img_ext):

    img_names = [fn for fn in os.listdir(path) if fn.lower().endswith(img_ext.lower())]
    img_paths = [os.path.join(path, fn) for fn in img_names]
    img_paths.sort()
    img_cnt = len(img_paths)

    return [img_paths, img_names, img_cnt]