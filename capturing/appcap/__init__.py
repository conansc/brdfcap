from appcap.capture_handler import CaptureHandler
from appcap.raspberry_lighting import RaspberryLighting
from appcap.argument_handler import ArgumentHandler
import os
import datetime as dt


def capture():

    # Instantiate needed handlers
    args = ArgumentHandler.get_args()
    rot = args.rotation
    sapers = args.sample_apertures
    leds = args.leds
    iterations = args.iterations
    path = args.path
    format = args.format
    ledsoff = args.ledsoff
    sample = args.sample
    take_calib = args.take_calibration
    if sample is None:
        curr_time = dt.datetime.now().strftime("%d-%m-%y-%H-%M")
        sample = curr_time

    lh = RaspberryLighting()
    ch = CaptureHandler()

    if format:
        CaptureHandler.format_card()

    # Capture calibration image
    if take_calib:
        lh.turn_on_all()
        ch.capture_calib()

    # Do capturing for all given leds and apertures
    for saper in sapers:
        for led_idx in leds:

            # Get angle of current led
            curr_ang = lh.get_angle_for_idx(led_idx)

            # Define destination folder for current material and angle
            dst_folder = "%s_%sdeg_%.1ff_%irot" % (sample, curr_ang, saper, rot)
            dst_path = os.path.join(path, dst_folder)

            # Turn on led for current capturing sequence
            lh.turn_on_idx(led_idx)

            # Capture sequences multiple times
            # s.t. they can be averaged later to get rid of noise
            for i in range(iterations):
                # Capture image sequence for current iterations
                curr_iter = str(i)
                ch.capture_var(dst_path, curr_iter, saper)

            # Copy calibration image for current sample
            if take_calib:
                ch.copy_calib(dst_path)

    # Clean temporary files
    ch.clean_tmp()

    if ledsoff:
        lh.turn_off_all()
    else:
        lh.turn_on_all()
