import os
import numpy as np


feat_path = "data/tb_input_feature.txt"
mask_path = "data/tb_spatial_mask.txt"

mask_np = np.loadtxt(mask_path)
feat_np = np.loadtxt(feat_path)

np.save("tb_input_feature.npy", feat_np)
np.save("tb_spatial_mask.npy", mask_np)

print("Input feature and mask saved.")