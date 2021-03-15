# Quick script to remove unavailable files
# Run from directory containing img1 and img2.txt
import pdb
import os
import numpy as np
import nibabel as nib

data_path = os.path.expanduser("~/data/adni/img_64_longitudinal/")

file_idx1 = np.genfromtxt("img1.txt", dtype='str') 
file_idx2 = np.genfromtxt("img2.txt", dtype='str')
n = len(file_idx1)

invalid_indices = []
for i in range(n):
    full_fp1 = os.path.join(data_path, file_idx1[i])
    full_fp2 = os.path.join(data_path, file_idx2[i])

    # load image from nibabel
    # try:
    #     nib.load(full_fp1).get_fdata()
    #     nib.load(full_fp2).get_fdata()
    # except FileNotFoundError as ferr:
    #     print("File not valid at idx {}".format(idx))
    #     invalid_indices.append(i)

    if os.path.isfile(full_fp1) and os.path.isfile(full_fp2):
        # print("i: [{}]/[{}] exists!".format(i, n))
        pass
    else:
        print("i: [{}]/[{}] doesn't exist!".format(i, n))
        invalid_indices.append(i)

cleaned_idx1 = np.delete(file_idx1, invalid_indices)
cleaned_idx2 = np.delete(file_idx2, invalid_indices)

print("Saving cleaned image file paths")
np.savetxt("img1_c.txt", cleaned_idx1, fmt="%s")
np.savetxt("img2_c.txt", cleaned_idx2, fmt="%s")

pdb.set_trace()