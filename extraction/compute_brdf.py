import argparse
import brdfext


def get_args():
    """
     Parses arguments passed by user.
     © Sebastian Cucerca
    """

    #
    parser = argparse.ArgumentParser()

    #
    parser.add_argument("-samp_folder", default="data/polyethylen", help="Path to folder with sample files.")
    parser.add_argument("-light_folder", default="data/light", help="Path to folder with light files.")
    parser.add_argument("-ref_folder", default="data/ref", help="Path to folder with reference files.")
    parser.add_argument("-samp_rot", default=.0, help="Rotation of sample on cylinder.")
    parser.add_argument("-params_file", default="config/params.ini", help="Path to ini file cotaining parameters.")

    #
    args = parser.parse_args()

    #
    return args


if __name__ == "__main__":

    # Get arguments
    args = get_args()

    # Compute BRDF
    brdfext.compute_brdf(args.samp_folder, args.ref_folder, args.light_folder, args.samp_rot, args.params_file)
