# import torch, os, sys
# import numpy as np
# from data.ambi.ambi_parser import readAmbiEGWithMatch
#
# def extract_feats_from_EGs(egs_file_path):
#
#     pairs, _, _, match_number, matches = readAmbiEGWithMatch(egs_file_path)
#
#     cam_feats = dict()
#     for p, pairs in enumerate(pairs):
#
#         match_array = np.asarray(matches[p])
#         p1_feats = match_array[:, :5]
#         p2_feats = match_array[:, 5:]
#
#         p1 = pairs[0]
#         if p1 not in cam_feats:
#             cam_feats[p1] = dict()
#         for j in range(p1_feats.shape[0]):
#             cam_key = int(p1_feats[j, 0])
#             cam_feats[p1][cam_key] = p1_feats[j, -2:]
#
#         p2 = pairs[1]
#         if p2 not in cam_feats:
#             cam_feats[p2] = dict()
#         for j in range(p1_feats.shape[0]):
#             cam_key = int(p2_feats[j, 0])
#             cam_feats[p2][cam_key] = p2_feats[j, -2:]
#
#     return cam_feats
#
# def compute_covis_feats(bundler_file):
#
