import argparse


class ArgumentHandler:

    @staticmethod
    def get_args():
        
        #
        parser = argparse.ArgumentParser()
        parser.add_argument("-leds", default=[7], type=int, nargs='+', help="List with LEDs to capture sequences for.")
        parser.add_argument("-rotation", default=0, type=int, help="Rotation of the applied sample.")
        
        #
        parser.add_argument("-sample_apertures", default=[11.0], type=float, nargs='+', help="Apertures for sample images.")
        parser.add_argument("-sample_iso", default=320, type=int, help="Iso for sample images.")
        parser.add_argument("-sample_exposures", default=['0.00025', '0.0005', '0.001', '0.002', '0.004', '0.008', '0.0166', '0.0333', '0.0666', '0.125', '0.25', '0.5', '1.0', '2.0', '4.0', '8.0', '15.0', '30.0'], nargs='+', help="Exposures for sample captures.")
        
        #
        parser.add_argument("-take_calibration", default=True, help="Indicates if calibration image should be taken.")
        parser.add_argument("-calib_aperture", default=11.0, type=float, help="Aperture for calibration image.")
        parser.add_argument("-calib_iso", default=3200, type=int, help="Iso for calibration image.")
        parser.add_argument("-calib_exposure", default='30.0', help="Exposure for calibration images.")
        parser.add_argument("-calib_folder", default="calib", help="Folder name for calibration images.");
        
        #
        parser.add_argument("-iterations", default=1, type=int, help="Number of sequences to capture per LED.")
        parser.add_argument("-sample", default=None, help="Name of the applied sample.")
        parser.add_argument("-path", default="data", help="Name of the applied sample.")
	
	#
        parser.add_argument("-format", default=True, help="Formats the card before capturing.")
        parser.add_argument("-ledsoff", default=False, help="Indicates if LEDs should be turned of after capturing process.")
        
        #
        parser.add_argument("-file_type", default="nef", help="File type of all taken images.");
        parser.add_argument("-image_format", default=24, help="Image format for capturing. See gphoto documentation for further details.");
        parser.add_argument("-wb_type", default=8, help="Type of white balancing. See gphoto documentation for further details.");
        parser.add_argument("-tmp_cap", default="capt0000", help="Name of captured image file created by gphoto.")
        parser.add_argument("-tmp_calib", default="calib_tmp", help="Temporary name of calibration image file.")
        parser.add_argument("-lamp_names", default=["50i", "30i", "25i", "45i", "50ii", "40ii", "30ii", "20ii", "15ii", "25ii", "35ii", "45ii", "40iii", "20iii", "15iii", "35iii"], help="Names of the installed lamps")
        parser.add_argument("-lamp_pins", default=[25, 27, 21, 6, 24, 4, 17, 23, 13, 26, 5, 19, 22, 12, 16, 20], help="Pins of installed lamps on Raspberry board.")
        
        #
        args = parser.parse_args()
        
        #
        return args
