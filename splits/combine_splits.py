import numpy as np
import pdb

grouped_fp = "aud_img_D"
files_to_merge = ["aud_img_E.npy", "aud_img_H.npy", "aud_img_HE.npy"]
paired_per_group = [np.load(pair_file) for pair_file in files_to_merge]
grouped_pairs = np.concatenate(paired_per_group)

# save grouped pairs
np.save(grouped_fp, grouped_pairs)
print("Saved grouped fp: {}".format(grouped_fp))