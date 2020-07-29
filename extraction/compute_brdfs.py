# import numpy as np
# import brdfacqol
# import os
#
# def get_paths(base_folder):
#
#     sample_folders = os.listdir(base_folder)
#     samples = []
#
#     for sample_folder in sample_folders:
#
#         parts = sample_folder.split("_")
#         sample_folder_base = parts[0]
#
#         if Params.LAMP_FOLDER == sample_folder_base or Params.REF_FOLDER == sample_folder_base:
#             continue
#
#         rot_num = parts[3][:-3]
#
#         sample_folder_path = os.path.join(Params.BASE_FOLDER, sample_folder)
#
#         lamp_folder_query = "%s_%s_%s*" % (Params.LAMP_FOLDER, parts[1], parts[2])
#         lamp_folder_query = os.path.join(Params.BASE_FOLDER, lamp_folder_query)
#         lamp_folder_results = glob.glob(lamp_folder_query)
#
#         # (!) Reference has to be computed with same rotation to get correct tangent vectors/phi angles
#         ref_folder_query = "%s_%s_%s_%s*" % (Params.REF_FOLDER, parts[1], parts[2], parts[3])
#         ref_folder_query = os.path.join(Params.BASE_FOLDER, ref_folder_query)
#         ref_folder_results = glob.glob(ref_folder_query)
#
#         if len(lamp_folder_results) == 0 or len(ref_folder_results) == 0:
#             continue
#
#         lamp_folder_path = lamp_folder_results[0]
#         ref_folder_path = ref_folder_results[0]
#
#         samples.append([sample_folder_path, lamp_folder_path, ref_folder_path, rot_num])
#
#     return np.array(samples)
#
#
# if __name__ == "__main__":
#     brdfacqol.compute_brdfs()
#
