import numpy as np
import pdb

file_idx1 = np.genfromtxt("img1_c.txt", dtype='str') 
file_idx2 = np.genfromtxt("img2_c.txt", dtype='str')
bad_adni = np.genfromtxt("bad_adni_files.txt", dtype="str", delimiter=",")

file_idx1_dict = set()
file_idx2_dict = set()

for fp in file_idx1:
    file_idx1_dict.add(fp)
for fp in file_idx2:
    file_idx2_dict.add(fp)

for fp in bad_adni:
    if fp in file_idx1_dict:
        pdb.set_trace()
    if fp in file_idx2_dict:
        pdb.set_trace()

print("No files in bad_adni_files in pairs")