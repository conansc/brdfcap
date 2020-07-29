from brdfacq.argument_handler import ArgumentHandler
import os
import shutil


class CaptureHandler:

    def __init__(self):

        args = ArgumentHandler.get_args()

        self.calib_aperture = args.calib_aperture  # Aperture for calibration images
        self.calib_iso = args.calib_iso  # ISO for calibration images
        self.calib_exposure = args.calib_exposure  # Exposure for calibration images

        self.est_iso = args.sample_iso  # ISO for estimation images
        self.est_exposures = args.sample_exposures  # Exposures for estimation images
        self.exp_cnt = len(self.est_exposures)  # Exposure count for estimation

        self.calib_folder = args.calib_folder  # Folder for calibration images
        self.file_type = args.file_type  # File type of all taken images
        self.tmp_cap = args.tmp_cap + "." + args.file_type #
        self.tmp_calib = args.tmp_calib + "." + args.file_type #

        self.image_format = args.image_format #
        self.wb_type = args.wb_type #

    def capture_var(self, dst_folder, iteration, saper):

        # Make sure destination path for captured images exists
        sequence_path = os.path.join(dst_folder, iteration)
        CaptureHandler.check_folder(sequence_path)

        # Capture images at different exposures
        for i in range(self.exp_cnt):
            dst_file = os.path.join(sequence_path, str(i) + "." + self.file_type)
            self._capture_img(saper, self.est_iso, self.est_exposures[i])
            CaptureHandler._save_tmp(self.tmp_cap, dst_file)

    def capture_calib(self):

        # Make sure destination path for captured images exists
        self._capture_img(self.calib_aperture, self.calib_iso, self.calib_exposure)
        CaptureHandler._save_tmp(self.tmp_cap, self.tmp_calib)

    def _capture_img(self, aperture, iso, exposure):

        # Make sure there is no old temporary file
        # In that case gphoto stops capturing and asks if it should replace it
        self._clean(self.tmp_cap)

        print("Taking picture with exposure %s, iso %i and aperture %.1f" % (exposure, iso, aperture))

        # Send capturing task to gphoto including all parameters
        task = 'gphoto2 --set-config f-number=' + str(aperture) \
               + ' --set-config iso=' + str(iso) \
               + ' --set-config shutterspeed=' + str(exposure) \
               + ' --set-config whitebalance=' + str(self.wb_type) \
               + ' --capture-image-and-download'
	print(task)
        os.system(task)

    def copy_calib(self, dst_folder):
        print("Copying calibration image.")
        dst_folder = os.path.join(dst_folder, self.calib_folder)
        CaptureHandler.check_folder(dst_folder)
        dst_file = os.path.join(dst_folder, "0." + self.file_type)
        shutil.copy(self.tmp_calib, dst_file)

    def clean_tmp(self):
        print("Cleaning temporary files.")
        CaptureHandler._clean(self.tmp_cap)
        CaptureHandler._clean(self.tmp_calib)

    @staticmethod
    def _clean(file):
        # Make sure there is no old temporary file
        if os.path.isfile(file):
            os.remove(file)

    @staticmethod
    def _save_tmp(temp_file_path, dst_file_path):
        # Move and simultaneously rename temporary file to destination
        print("Moving file from " + temp_file_path + " to " + dst_file_path)
        CaptureHandler._clean(dst_file_path)
        os.rename(temp_file_path, dst_file_path)

    @staticmethod
    def check_folder(path):
        # Check if folder exists
        if not os.path.isdir(path):
            # Create if it does not
            os.makedirs(path)

    @staticmethod
    def format_card():
        task = 'gphoto2 --delete-all-files'
        os.system(task)
